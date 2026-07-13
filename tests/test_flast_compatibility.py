"""Regression tests for behavior shared with the reference FLAST algorithm."""

from __future__ import annotations

import numpy as np

from flast_cli.features import build_feature_text
from flast_cli.model import _as_flast_matrix, _project_features
from flast_cli.schemas import FeatureInput


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
