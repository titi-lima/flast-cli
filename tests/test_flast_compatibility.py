"""Regression tests for behavior shared with the reference FLAST algorithm."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from flast_cli.features import build_feature_text
from flast_cli.model import (
    TrainedArtifacts,
    TrainingParams,
    _as_flast_matrix,
    _project_features,
    load_trained_artifacts,
    predict_with_explanations,
    train_model,
)
from flast_cli.schemas import FeatureInput, LabelRecord


def test_model_feature_text_is_exactly_the_static_test_source(tmp_path) -> None:
    feature = FeatureInput(
        test_id="tests/test_example.py::TestExample::test_case",
        normalized_test_id="tests/test_example.py::TestExample::test_case",
        project_root=tmp_path,
        file_path=tmp_path / "tests" / "test_example.py",
        class_name="TestExample",
        test_name="test_case",
        source_text="def test_case():\n    assert True\n",
    )

    assert build_feature_text(feature) == feature.source_text


def test_projection_uses_the_uncapped_jl_dimension(monkeypatch) -> None:
    captured = {}

    class Projection:
        def __init__(self, *, n_components):
            captured["n_components"] = n_components

        def fit_transform(self, matrix):
            return matrix

    monkeypatch.setattr("flast_cli.model.johnson_lindenstrauss_min_dim", lambda *_args, **_kwargs: 17)
    monkeypatch.setattr("flast_cli.model.SparseRandomProjection", Projection)

    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    projected, projection = _project_features(matrix, eps=0.3, fit=True)

    assert projected is matrix
    assert isinstance(projection, Projection)
    assert captured["n_components"] == 17


def test_flast_matrix_conversion_is_dense() -> None:
    class SparseLike:
        def toarray(self):
            return [[1, 2], [3, 4]]

    matrix = _as_flast_matrix(SparseLike())

    assert isinstance(matrix, np.ndarray)
    assert matrix.tolist() == [[1, 2], [3, 4]]


class _IdentityVectorizer:
    def transform(self, matrix):
        return matrix


@pytest.mark.parametrize(
    ("training_data", "training_labels", "test_data", "expected"),
    [
        ([[0.0], [2.0]], [0, 1], [[0.0]], "non_flaky"),
        ([[0.0], [0.0]], [0, 1], [[0.0]], "non_flaky"),
        ([[0.0], [2.0]], [1, 0], [[0.0]], "flaky"),
        ([[0.0], [2.0]], [0, 1], [[1.0]], "flaky"),
    ],
)
def test_distance_vote_matches_flast_limit_cases_at_zero_sigma(
    training_data, training_labels, test_data, expected
) -> None:
    model = KNeighborsClassifier(
        algorithm="brute",
        metric="euclidean",
        weights="distance",
        n_neighbors=2,
        n_jobs=1,
    ).fit(np.asarray(training_data), np.asarray(training_labels))
    artifacts = TrainedArtifacts(
        model=model,
        vectorizer=_IdentityVectorizer(),
        projection=None,
        metadata={"params": {"eps": 0, "sigma": 0}},
        training_index=[],
    )

    prediction = predict_with_explanations(artifacts, np.asarray(test_data))[0]

    assert prediction.predicted_label == expected


def test_vectorization_uses_the_complete_known_corpus_before_label_selection(
    tmp_path,
) -> None:
    labels = [
        LabelRecord(
            schema_version=1,
            framework="pytest",
            test_id="flaky",
            project_root=str(tmp_path),
            label="flaky",
            reason="",
            total_observations=3,
            outcome_counts={},
        ),
        LabelRecord(
            schema_version=1,
            framework="pytest",
            test_id="stable",
            project_root=str(tmp_path),
            label="non_flaky",
            reason="",
            total_observations=3,
            outcome_counts={},
        ),
        LabelRecord(
            schema_version=1,
            framework="pytest",
            test_id="unknown",
            project_root=str(tmp_path),
            label="unlabeled",
            reason="",
            total_observations=1,
            outcome_counts={},
        ),
    ]

    metadata = train_model(
        project_root=tmp_path,
        labels=labels,
        feature_texts=["alpha", "beta", "unlabeled_only_token"],
        params=TrainingParams(k=2, eps=0),
    )

    artifacts = load_trained_artifacts(tmp_path)
    assert "unlabeled_only_token" in artifacts.vectorizer.vocabulary_
    assert artifacts.model.n_samples_fit_ == 2
    assert metadata["vectorization_corpus_size"] == 3
