"""FLAST-compatible vectorization, training, and prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    training_index: List[LabelRecord]


@dataclass(frozen=True)
class NeighborEvidence:
    test_id: str
    label: str
    distance: float
    score_contribution: float
    reason: str
    total_observations: int
    outcome_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "label": self.label,
            "distance": self.distance,
            "score_contribution": self.score_contribution,
            "reason": self.reason,
            "total_observations": self.total_observations,
            "outcome_counts": dict(self.outcome_counts),
        }


@dataclass(frozen=True)
class ExplainedPrediction:
    score: float
    predicted_label: str
    nearest_examples: List[NeighborEvidence]


def _project_features(
    matrix,
    *,
    eps: float,
    fit: bool,
    projection: Optional[SparseRandomProjection] = None,
):
    # This follows flast.flastVectorization: epsilon zero skips the random
    # projection; otherwise use the JL lower bound based on sample count.
    if eps == 0:
        return matrix, None
    if fit:
        dim = int(johnson_lindenstrauss_min_dim(matrix.shape[0], eps=eps))
        # Do not cap this at the vocabulary size.  FLAST intentionally uses
        # the JL dimension as-is, including when it expands the input space.
        projection = SparseRandomProjection(n_components=dim)
        return projection.fit_transform(matrix), projection
    if projection is None:
        raise ValueError("Projection artifact missing for projected inference.")
    return projection.transform(matrix), projection


def _as_flast_matrix(matrix) -> np.ndarray:
    """Match FLAST's dense row representation before fitting or querying kNN."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


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
    matrix = _as_flast_matrix(matrix)
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
    training_index_path = model_artifact_path(project_root, "training_index.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(projection, projection_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training_index_path.write_text(
        json.dumps([label.to_dict() for label, _ in labeled], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_trained_artifacts(project_root: Path) -> TrainedArtifacts:
    model_path = model_artifact_path(project_root, "model.joblib")
    vectorizer_path = model_artifact_path(project_root, "vectorizer.joblib")
    projection_path = model_artifact_path(project_root, "projection.joblib")
    metadata_path = model_artifact_path(project_root, "metadata.json")
    training_index_path = model_artifact_path(project_root, "training_index.json")
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
        training_index=[
            LabelRecord(**item)
            for item in json.loads(training_index_path.read_text(encoding="utf-8"))
        ]
        if training_index_path.exists()
        else [],
    )


def predict_scores(
    artifacts: TrainedArtifacts,
    feature_texts: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    explained = predict_with_explanations(artifacts, feature_texts)
    return (
        np.array([item.score for item in explained]),
        np.array([1 if item.predicted_label == "flaky" else 0 for item in explained]),
    )


def predict_with_explanations(
    artifacts: TrainedArtifacts,
    feature_texts: Sequence[str],
) -> List[ExplainedPrediction]:
    matrix = artifacts.vectorizer.transform(feature_texts)
    matrix, _ = _project_features(
        matrix,
        eps=artifacts.metadata["params"]["eps"],
        fit=False,
        projection=artifacts.projection,
    )
    matrix = _as_flast_matrix(matrix)
    distances, neighbors = artifacts.model.kneighbors(matrix)
    labels = np.asarray(artifacts.model._y)  # populated by sklearn during fit
    sigma = float(artifacts.metadata["params"]["sigma"])
    explained: List[ExplainedPrediction] = []

    for row_distances, row_neighbors in zip(distances, neighbors):
        phi = 0.0
        psi = 0.0
        weighted_neighbors = []
        for distance, neighbor in zip(row_distances, row_neighbors):
            if artifacts.model.weights == "distance":
                weight = (1.0 / distance) if distance != 0 else float("inf")
            else:
                weight = 1.0
            weighted_neighbors.append((float(distance), int(neighbor), float(weight)))
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

        nearest_examples = _neighbor_evidence(
            training_index=artifacts.training_index,
            labels=labels,
            weighted_neighbors=weighted_neighbors,
            phi=phi,
            psi=psi,
        )
        explained.append(
            ExplainedPrediction(
                score=float(score),
                predicted_label="flaky" if score >= sigma else "non_flaky",
                nearest_examples=nearest_examples,
            )
        )

    return explained


def _neighbor_evidence(
    *,
    training_index: Sequence[LabelRecord],
    labels: np.ndarray,
    weighted_neighbors: Sequence[Tuple[float, int, float]],
    phi: float,
    psi: float,
) -> List[NeighborEvidence]:
    if not training_index:
        return []

    denominator = phi + psi
    infinite_neighbors = [
        (distance, neighbor, weight)
        for distance, neighbor, weight in weighted_neighbors
        if weight == float("inf")
    ]
    examples: List[NeighborEvidence] = []
    for distance, neighbor, weight in weighted_neighbors:
        if neighbor >= len(training_index):
            continue
        label_record = training_index[neighbor]
        if infinite_neighbors:
            matching_exact = [item for item in infinite_neighbors if labels[item[1]] == labels[neighbor]]
            contribution = 1.0 / len(matching_exact) if matching_exact else 0.0
        elif denominator == 0:
            contribution = 0.0
        else:
            contribution = weight / denominator
        if labels[neighbor] == 0:
            contribution *= -1.0
        examples.append(
            NeighborEvidence(
                test_id=label_record.test_id,
                label=label_record.label,
                distance=float(distance),
                score_contribution=float(contribution),
                reason=label_record.reason,
                total_observations=label_record.total_observations,
                outcome_counts=dict(label_record.outcome_counts),
            )
        )
    return examples
