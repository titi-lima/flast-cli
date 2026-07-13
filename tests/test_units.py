from __future__ import annotations

import json
from pathlib import Path

import pytest

from flast_cli.adapters import PytestAdapter
from flast_cli.features import build_feature_text, detect_static_signals, resolve_feature_input
from flast_cli.labels import aggregate_manifests, derive_labels
from flast_cli.manifests import load_manifest
from flast_cli.model import TrainingParams, load_trained_artifacts, predict_with_explanations, train_model
from flast_cli.schemas import LabelRecord, PredictionRecord, RunManifest, TestOutcome as ManifestOutcome


def test_manifest_schema_rejection(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "framework": "pytest",
                "run_id": "bad",
                "project_root": str(tmp_path),
                "tool": {"name": "pytest", "version": "8.0.0"},
                "collected": [],
                "outcomes": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported manifest schema_version"):
        load_manifest(manifest_path)


def test_derive_labels_handles_flaky_non_flaky_and_unlabeled(tmp_path: Path) -> None:
    manifests = [
        RunManifest(
            schema_version=1,
            framework="pytest",
            run_id="run-1",
            project_root=str(tmp_path),
            tool={"name": "pytest", "version": "8.0.0"},
            collected=[],
            outcomes=[
                ManifestOutcome("tests/test_sample.py::test_flaky", "passed"),
                ManifestOutcome("tests/test_sample.py::test_stable", "passed"),
                ManifestOutcome("tests/test_sample.py::test_skip_only", "skipped"),
            ],
        ),
        RunManifest(
            schema_version=1,
            framework="pytest",
            run_id="run-2",
            project_root=str(tmp_path),
            tool={"name": "pytest", "version": "8.0.0"},
            collected=[],
            outcomes=[
                ManifestOutcome("tests/test_sample.py::test_flaky", "failed"),
                ManifestOutcome("tests/test_sample.py::test_stable", "passed"),
                ManifestOutcome("tests/test_sample.py::test_skip_only", "xfail"),
            ],
        ),
        RunManifest(
            schema_version=1,
            framework="pytest",
            run_id="run-3",
            project_root=str(tmp_path),
            tool={"name": "pytest", "version": "8.0.0"},
            collected=[],
            outcomes=[
                ManifestOutcome("tests/test_sample.py::test_stable", "passed"),
            ],
        ),
    ]

    observations = aggregate_manifests(manifests)
    labels = derive_labels(observations, min_observations=3)
    by_test = {label.test_id: label for label in labels}

    assert by_test["tests/test_sample.py::test_flaky"].label == "flaky"
    assert by_test["tests/test_sample.py::test_stable"].label == "non_flaky"
    assert by_test["tests/test_sample.py::test_skip_only"].label == "unlabeled"


def test_feature_resolution_supports_class_methods_and_parametrized_names(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "tests" / "test_example.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """
class TestExample:
    def test_method(self):
        assert 1 == 1

def test_parametrized(value):
    assert value in {1, 2}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    method_feature = resolve_feature_input(
        "tests/test_example.py::TestExample::test_method", tmp_path
    )
    parametrized_feature = resolve_feature_input(
        "tests/test_example.py::test_parametrized[1]", tmp_path
    )

    assert method_feature.class_name == "TestExample"
    assert "def test_method" in method_feature.source_text
    assert parametrized_feature.test_name == "test_parametrized"
    assert "test_parametrized" in build_feature_text(parametrized_feature)


def test_static_signal_detection_finds_common_flaky_clues(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_example.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """
import os
import random
import time
from pathlib import Path

def test_parametrized_case(value):
    os.environ["TOKEN"] = "changed"
    time.sleep(random.random())
    Path("shared-state.txt").write_text("1")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    feature = resolve_feature_input(
        "tests/test_example.py::test_parametrized_case[1]", tmp_path
    )
    kinds = {item["kind"] for item in detect_static_signals(feature)}

    assert {"randomness", "timing", "filesystem state", "environment mutation", "parametrization"}.issubset(kinds)


def test_static_signal_detection_honors_flast_ignore_comments(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_example.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """
def test_ignored_static_signals():
    # flast:ignore
    time.sleep(1)
    random.random()  # flast:ignore
    Path("shared-state.txt").write_text("1")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    feature = resolve_feature_input(
        "tests/test_example.py::test_ignored_static_signals", tmp_path
    )
    kinds = {item["kind"] for item in detect_static_signals(feature)}

    assert "timing" not in kinds
    assert "randomness" not in kinds
    assert "filesystem state" in kinds


def test_diagnostic_line_resolution_supports_module_and_class_tests(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_example.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """
def test_module_level():
    assert True


class TestExample:
    def test_method(self):
        assert True
""".strip()
        + "\n",
        encoding="utf-8",
    )

    module_prediction = PredictionRecord(
        test_id="tests/test_example.py::test_module_level",
        score=1.0,
        predicted_label="flaky",
        class_name=None,
        test_name="test_module_level",
        file_path=str(test_file),
    )
    method_prediction = PredictionRecord(
        test_id="tests/test_example.py::TestExample::test_method",
        score=1.0,
        predicted_label="flaky",
        class_name="TestExample",
        test_name="test_method",
        file_path=str(test_file),
    )

    adapter = PytestAdapter()
    assert adapter.resolve_prediction_line(module_prediction, tmp_path) == 1
    assert adapter.resolve_prediction_line(method_prediction, tmp_path) == 6


def test_train_model_requires_both_positive_and_negative_labels(tmp_path: Path) -> None:
    labels = [
        LabelRecord(
            schema_version=1,
            framework="pytest",
            test_id="tests/test_only.py::test_one",
            project_root=str(tmp_path),
            label="flaky",
            reason="observed both passed and failed",
            total_observations=3,
            outcome_counts={"passed": 1, "failed": 2},
        )
    ]

    with pytest.raises(ValueError, match="at least one flaky and one non_flaky"):
        train_model(
            project_root=tmp_path,
            labels=labels,
            feature_texts=["test_id: tests/test_only.py::test_one\nsource:\nassert 1"],
            params=TrainingParams(),
        )


def test_predict_with_explanations_maps_neighbors_to_training_labels(tmp_path: Path) -> None:
    labels = [
        LabelRecord(
            schema_version=1,
            framework="pytest",
            test_id="tests/test_flaky.py::test_uses_time",
            project_root=str(tmp_path),
            label="flaky",
            reason="observed both passed and failure-like outcomes",
            total_observations=4,
            outcome_counts={"passed": 2, "failed": 2},
        ),
        LabelRecord(
            schema_version=1,
            framework="pytest",
            test_id="tests/test_stable.py::test_simple_math",
            project_root=str(tmp_path),
            label="non_flaky",
            reason="stable across observed runs",
            total_observations=4,
            outcome_counts={"passed": 4},
        ),
    ]
    train_model(
        project_root=tmp_path,
        labels=labels,
        feature_texts=[
            "test_id: tests/test_flaky.py::test_uses_time\nsource:\nimport time\ntime.sleep(1)",
            "test_id: tests/test_stable.py::test_simple_math\nsource:\nassert 1 + 1 == 2",
        ],
        params=TrainingParams(k=2, eps=0),
    )

    artifacts = load_trained_artifacts(tmp_path)
    assert [item.test_id for item in artifacts.training_index] == [
        "tests/test_flaky.py::test_uses_time",
        "tests/test_stable.py::test_simple_math",
    ]

    explained = predict_with_explanations(
        artifacts,
        ["test_id: tests/test_new.py::test_uses_time\nsource:\nimport time\ntime.sleep(1)"],
    )

    assert explained[0].nearest_examples
    neighbor_ids = {item.test_id for item in explained[0].nearest_examples}
    assert "tests/test_flaky.py::test_uses_time" in neighbor_ids
    assert all(item.reason for item in explained[0].nearest_examples)
