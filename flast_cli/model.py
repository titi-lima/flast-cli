"""FLAST-compatible vectorization, training, and prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import (
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

from .constants import (
    DEFAULT_ALGORITHM,
    DEFAULT_EPS,
    DEFAULT_K,
    DEFAULT_METRIC,
    DEFAULT_SIGMA,
    DEFAULT_WEIGHTS,
    FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from .schemas import LabelRecord
from .workspace import model_artifact_path


@dataclass(frozen=True)
class TrainingParams:
    k: int = DEFAULT_K
    sigma: float = DEFAULT_SIGMA
    eps: float = DEFAULT_EPS
    metric: str = DEFAULT_METRIC
    weights: str = DEFAULT_WEIGHTS
    algorithm: str = DEFAULT_ALGORITHM


@dataclass(frozen=True)
class TrainedArtifacts:
    model: KNeighborsClassifier
    vectorizer: CountVectorizer
    projection: Optional[SparseRandomProjection]
    metadata: dict


def _project_features(
    matrix,
    *,
    eps: float,
    fit: bool,
    projection: Optional[SparseRandomProjection] = None,
):
    if eps <= 0:
        return matrix, None
    if fit:
        dim = int(johnson_lindenstrauss_min_dim(matrix.shape[0], eps=eps))
        dim = max(1, min(dim, matrix.shape[1] or 1))
        projection = SparseRandomProjection(n_components=dim, random_state=0)
        return projection.fit_transform(matrix), projection
    if projection is None:
        raise ValueError("Projection artifact missing for projected inference.")
    return projection.transform(matrix), projection


def train_model(
    *,
    project_root: Path,
    labels: Sequence[LabelRecord],
    feature_texts: Sequence[str],
    params: TrainingParams,
) -> dict:
    labeled = [
        (label, feature)
        for label, feature in zip(labels, feature_texts)
        if label.label in {"flaky", "non_flaky"}
    ]
    if not labeled:
        raise ValueError("No labeled samples available for training.")

    target = np.array([1 if label.label == "flaky" else 0 for label, _ in labeled])
    if np.all(target == 1) or np.all(target == 0):
        raise ValueError(
            "Training requires at least one flaky and one non_flaky labeled test."
        )

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform([feature for _, feature in labeled])
    matrix, projection = _project_features(matrix, eps=params.eps, fit=True)
    effective_k = min(params.k, len(labeled))
    model = KNeighborsClassifier(
        algorithm=params.algorithm,
        metric=params.metric,
        weights=params.weights,
        n_neighbors=effective_k,
        n_jobs=1,
    )
    model.fit(matrix, target)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "params": {
            "k": effective_k,
            "requested_k": params.k,
            "sigma": params.sigma,
            "eps": params.eps,
            "metric": params.metric,
            "weights": params.weights,
            "algorithm": params.algorithm,
        },
        "corpus_size": len(labeled),
        "label_counts": {
            "flaky": int(np.sum(target == 1)),
            "non_flaky": int(np.sum(target == 0)),
        },
    }

    model_path = model_artifact_path(project_root, "model.joblib")
    vectorizer_path = model_artifact_path(project_root, "vectorizer.joblib")
    projection_path = model_artifact_path(project_root, "projection.joblib")
    metadata_path = model_artifact_path(project_root, "metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(projection, projection_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_trained_artifacts(project_root: Path) -> TrainedArtifacts:
    model_path = model_artifact_path(project_root, "model.joblib")
    vectorizer_path = model_artifact_path(project_root, "vectorizer.joblib")
    projection_path = model_artifact_path(project_root, "projection.joblib")
    metadata_path = model_artifact_path(project_root, "metadata.json")
    if not model_path.exists() or not vectorizer_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing trained model artifacts under {model_path.parent}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "Model feature schema version does not match current extractor version."
        )

    return TrainedArtifacts(
        model=joblib.load(model_path),
        vectorizer=joblib.load(vectorizer_path),
        projection=joblib.load(projection_path) if projection_path.exists() else None,
        metadata=metadata,
    )


def predict_scores(
    artifacts: TrainedArtifacts,
    feature_texts: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    matrix = artifacts.vectorizer.transform(feature_texts)
    matrix, _ = _project_features(
        matrix,
        eps=artifacts.metadata["params"]["eps"],
        fit=False,
        projection=artifacts.projection,
    )
    distances, neighbors = artifacts.model.kneighbors(matrix)
    labels = np.asarray(artifacts.model._y)  # populated by sklearn during fit
    sigma = float(artifacts.metadata["params"]["sigma"])
    scores: List[float] = []
    predictions: List[int] = []

    for row_distances, row_neighbors in zip(distances, neighbors):
        phi = 0.0
        psi = 0.0
        for distance, neighbor in zip(row_distances, row_neighbors):
            if artifacts.model.weights == "distance":
                weight = (1.0 / distance) if distance != 0 else float("inf")
            else:
                weight = 1.0
            if labels[neighbor] == 1:
                phi += weight
            else:
                psi += weight

        if phi == float("inf") and psi == float("inf"):
            score = 0.0
        elif psi == float("inf"):
            score = 0.0
        elif phi == float("inf"):
            score = 1.0
        elif (phi + psi) == 0:
            score = 0.0
        else:
            score = phi / (phi + psi)
        scores.append(float(score))
        predictions.append(1 if score >= sigma else 0)

    return np.array(scores), np.array(predictions)
