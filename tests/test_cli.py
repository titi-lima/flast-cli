from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _cli_env() -> dict:
    env = os.environ.copy()
    package_root = Path(__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(package_root)
        if not existing
        else str(package_root) + os.pathsep + existing
    )
    return env


def _run_cli(project_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "flast_cli.cli", *args],
        cwd=project_dir,
        env=_cli_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _create_project(project_dir: Path) -> None:
    tests_dir = project_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        """
from pathlib import Path

STATE = Path(__file__).with_name("flaky_state.txt")

def _run_count():
    if not STATE.exists():
        STATE.write_text("0", encoding="utf-8")
    value = int(STATE.read_text(encoding="utf-8"))
    STATE.write_text(str(value + 1), encoding="utf-8")
    return value

def test_flaky_toggle():
    assert _run_count() % 2 == 0

def test_stable_pass():
    assert True
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_init_observe_train_predict_and_report(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)

    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr
    assert (project_dir / ".flast" / "config.toml").exists()
    assert not (project_dir / "pytest.ini").exists()

    result = _run_cli(project_dir, "observe", "pytest", "--runs", "3", "--pytest-arg", "tests")
    assert result.returncode == 0, result.stderr
    observations_path = project_dir / ".flast" / "corpus" / "observations.jsonl"
    labels_path = project_dir / ".flast" / "corpus" / "labels.jsonl"
    assert observations_path.exists()
    assert labels_path.exists()

    labels = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels_by_test = {item["test_id"]: item["label"] for item in labels}
    assert labels_by_test["tests/test_sample.py::test_flaky_toggle"] == "flaky"
    assert labels_by_test["tests/test_sample.py::test_stable_pass"] == "non_flaky"

    result = _run_cli(project_dir, "train")
    assert result.returncode == 0, result.stderr
    assert (project_dir / ".flast" / "models" / "latest" / "model.joblib").exists()

    report_output = project_dir / "prediction-report.json"
    result = _run_cli(
        project_dir,
        "predict",
        "pytest",
        "--format",
        "json",
        "--output",
        str(report_output),
        "--pytest-arg",
        "tests",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["predictions"]
    assert report_output.exists()

    run_id = payload["run_id"]
    result = _run_cli(project_dir, "report", "--run-id", run_id, "--format", "json")
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["run_id"] == run_id

    result = _run_cli(
        project_dir,
        "diagnostics",
        "pytest",
        "--format",
        "json",
        "--framework-arg",
        "tests",
    )
    assert result.returncode == 0, result.stderr
    diagnostics = json.loads(result.stdout)
    assert diagnostics["run_id"]
    assert diagnostics["diagnostics"]
    by_test = {item["test_id"]: item for item in diagnostics["diagnostics"]}
    flaky = by_test["tests/test_sample.py::test_flaky_toggle"]
    assert flaky["predicted_label"] == "flaky"
    assert flaky["file_path"].endswith("tests/test_sample.py")
    assert flaky["test_name"] == "test_flaky_toggle"
    assert flaky["line"] == 12
    assert "score" in flaky["message"]
    assert "explanation" in flaky
    assert flaky["explanation"]["summary"]
    assert flaky["explanation"]["nearest_examples"]
    assert "static_signals" in flaky["explanation"]
    assert flaky["explanation"]["observed_history"]["label"] == "flaky"


def test_observe_reuse_history_merges_imported_manifest(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)

    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr

    first = _run_cli(project_dir, "observe", "pytest", "--runs", "1", "--pytest-arg", "tests")
    assert first.returncode == 0, first.stderr

    existing_manifests = sorted((project_dir / ".flast" / "runs").glob("*/manifest.json"))
    assert existing_manifests
    external_dir = tmp_path / "external-history"
    external_dir.mkdir()
    external_manifest = external_dir / "imported.json"
    external_manifest.write_text(existing_manifests[0].read_text(encoding="utf-8"), encoding="utf-8")

    second = _run_cli(
        project_dir,
        "observe",
        "pytest",
        "--runs",
        "2",
        "--reuse-history",
        str(external_manifest),
        "--pytest-arg",
        "tests",
    )
    assert second.returncode == 0, second.stderr

    labels = [
        json.loads(line)
        for line in (project_dir / ".flast" / "corpus" / "labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    labels_by_test = {item["test_id"]: item["label"] for item in labels}
    assert labels_by_test["tests/test_sample.py::test_flaky_toggle"] == "flaky"


def test_predict_without_model_fails_cleanly(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr

    result = _run_cli(project_dir, "predict", "pytest", "--pytest-arg", "tests")
    assert result.returncode != 0
    assert "Missing trained model artifacts" in result.stderr


def test_diagnostics_without_model_fails_cleanly(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr

    result = _run_cli(project_dir, "diagnostics", "pytest", "--pytest-arg", "tests")
    assert result.returncode != 0
    assert "Missing trained model artifacts" in result.stderr
