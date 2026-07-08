from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flast_cli.constants import SCHEMA_VERSION
from flast_cli.manifests import write_manifest
from flast_cli.schemas import CollectedTest, RunManifest, TestOutcome as ManifestOutcome
from flast_cli.workspace import ensure_workspace, run_manifest_path


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


def test_init_vitest_installs_reporter_and_patches_test_script(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run"}}) + "\n",
        encoding="utf-8",
    )

    result = _run_cli(project_dir, "init", "vitest")
    assert result.returncode == 0, result.stderr
    assert (project_dir / ".flast" / "tools" / "vitest" / "flast-reporter.cjs").exists()
    package = json.loads((project_dir / "package.json").read_text(encoding="utf-8"))
    assert "--reporter=.flast/tools/vitest/flast-reporter.cjs" in package["scripts"]["test"]

    second = _run_cli(project_dir, "init", "vitest")
    assert second.returncode == 0, second.stderr
    package = json.loads((project_dir / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["test"].count("flast-reporter.cjs") == 1


def test_vitest_reporter_writes_manifest_with_vitest4_hook(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    result = _run_cli(project_dir, "init", "vitest")
    assert result.returncode == 0, result.stderr
    reporter = project_dir / ".flast" / "tools" / "vitest" / "flast-reporter.cjs"

    script = """
const Reporter = require(process.argv[1]);
const reporter = new Reporter();
function testCase(name, fullName, state) {
  return {
    name,
    fullName,
    result() {
      return { state };
    },
  };
}
reporter.onTestRunEnd([
  {
    relativeModuleId: "tests/sample.test.js",
    children: {
      allTests() {
        return [
          testCase("flaky toggle", "demo suite > flaky toggle", "failed"),
          testCase("stable pass", "demo suite > stable pass", "passed"),
        ];
      },
    },
  },
]);
"""
    node = subprocess.run(
        ["node", "-e", script, str(reporter)],
        cwd=project_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    assert node.returncode == 0, node.stderr

    manifests = sorted((project_dir / ".flast" / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["collected"] == [
        {"test_id": "tests/sample.test.js::demo suite::flaky toggle"},
        {"test_id": "tests/sample.test.js::demo suite::stable pass"},
    ]
    assert payload["outcomes"] == [
        {"test_id": "tests/sample.test.js::demo suite::flaky toggle", "outcome": "failed"},
        {"test_id": "tests/sample.test.js::demo suite::stable pass", "outcome": "passed"},
    ]


def test_init_junit_installs_platform_listener(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    result = _run_cli(project_dir, "init", "junit")
    assert result.returncode == 0, result.stderr

    listener = project_dir / "src" / "test" / "java" / "flast" / "FlastJUnitListener.java"
    service = (
        project_dir
        / "src"
        / "test"
        / "resources"
        / "META-INF"
        / "services"
        / "org.junit.platform.launcher.TestExecutionListener"
    )
    assert listener.exists()
    assert "class FlastJUnitListener" in listener.read_text(encoding="utf-8")
    assert service.read_text(encoding="utf-8") == "flast.FlastJUnitListener\n"


def test_init_junit_patches_ant_build_to_copy_test_resources(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "build.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<project name="demo" default="test">
    <property name="src.test" value="src/test/java"/>
    <property name="classes.test" value="build/classes/test"/>
    <target name="compile">
        <mkdir dir="${classes.test}"/>
        <javac includeantruntime="false" srcdir="${src.test}" destdir="${classes.test}"/>
    </target>
</project>
""",
        encoding="utf-8",
    )

    result = _run_cli(project_dir, "init", "junit")
    assert result.returncode == 0, result.stderr
    second = _run_cli(project_dir, "init", "junit")
    assert second.returncode == 0, second.stderr

    build_xml = (project_dir / "build.xml").read_text(encoding="utf-8")
    assert '<property name="src.test.resources" value="src/test/resources"/>' in build_xml
    assert '<fileset dir="${src.test.resources}" erroronmissingdir="false"/>' in build_xml
    assert build_xml.count("src.test.resources") == 2


def test_predict_without_model_fails_cleanly(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr

    result = _run_cli(project_dir, "predict", "pytest", "--pytest-arg", "tests")
    assert result.returncode != 0
    assert "Missing trained model artifacts" in result.stderr


def test_auto_before_history_returns_empty_diagnostics(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr

    result = _run_cli(project_dir, "auto", "pytest", "--framework-arg", "tests")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["framework"] == "pytest"
    assert payload["diagnostics"] == []


def test_auto_rebuilds_trains_and_emits_diagnostics_from_accumulated_runs(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr

    for _ in range(3):
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "tests"],
            cwd=project_dir,
            env=_cli_env(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert run.returncode in (0, 1), run.stdout + run.stderr

    result = _run_cli(project_dir, "auto", "pytest", "--framework-arg", "tests")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["diagnostics"]
    assert (project_dir / ".flast" / "models" / "latest" / "model.joblib").exists()


@pytest.mark.parametrize(
    ("framework", "test_id", "stable_test_id", "source"),
    [
        (
            "pytest",
            "tests/test_sample.py::test_flaky_toggle",
            "tests/test_sample.py::test_stable_pass",
            "def test_flaky_toggle():\n    assert True\n\n"
            "def test_stable_pass():\n    assert True\n",
        ),
        (
            "vitest",
            "tests/sample.test.js::demo suite::flaky toggle",
            "tests/sample.test.js::demo suite::stable pass",
            "import { describe, test, expect } from 'vitest';\n"
            "describe('demo suite', () => {\n"
            "  test('flaky toggle', () => {\n"
            "    expect(true).toBe(true);\n"
            "  });\n"
            "  test('stable pass', () => {\n"
            "    expect(true).toBe(true);\n"
            "  });\n"
            "});\n",
        ),
        (
            "junit",
            "src/test/java/com/example/AppTest.java::com.example.AppTest::flakyToggle",
            "src/test/java/com/example/AppTest.java::com.example.AppTest::stablePass",
            "package com.example;\n"
            "import org.junit.jupiter.api.Test;\n"
            "class AppTest {\n"
            "    @Test\n"
            "    void flakyToggle() {\n"
            "        org.junit.jupiter.api.Assertions.assertTrue(true);\n"
            "    }\n"
            "    @Test\n"
            "    void stablePass() {\n"
            "        org.junit.jupiter.api.Assertions.assertTrue(true);\n"
            "    }\n"
            "}\n",
        ),
    ],
)
def test_auto_rebuilds_trains_and_emits_diagnostics_for_each_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    framework: str,
    test_id: str,
    stable_test_id: str,
    source: str,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    source_path = project_dir / test_id.split("::", 1)[0]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    ensure_workspace(project_dir)

    for index, outcome in enumerate(["passed", "failed", "passed"]):
        manifest_path = run_manifest_path(project_dir, f"observed-{index}")
        write_manifest(
            manifest_path,
            RunManifest(
                schema_version=SCHEMA_VERSION,
                framework=framework,
                run_id=f"observed-{index}",
                project_root=str(project_dir),
                tool={"name": framework, "version": "test"},
                collected=[
                    CollectedTest(test_id=test_id),
                    CollectedTest(test_id=stable_test_id),
                ],
                outcomes=[
                    ManifestOutcome(test_id=test_id, outcome=outcome),
                    ManifestOutcome(test_id=stable_test_id, outcome="passed"),
                ],
            ),
        )

    def fake_collection(*, adapter, project_root, framework_args):
        manifest_path = run_manifest_path(project_root, "collection")
        write_manifest(
            manifest_path,
            RunManifest(
                schema_version=SCHEMA_VERSION,
                framework=adapter.framework,
                run_id="collection",
                project_root=str(project_root),
                tool={"name": adapter.framework, "version": "test"},
                collected=[CollectedTest(test_id=test_id)],
                outcomes=[],
            ),
        )
        return manifest_path

    monkeypatch.setattr("flast_cli.cli._run_framework_collection", fake_collection)

    from flast_cli.cli import run_auto

    result = run_auto(
        argparse.Namespace(
            project_dir=str(project_dir),
            framework=framework,
            framework_args=[],
            min_observations=None,
            output=None,
        )
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["framework"] == framework
    assert payload["diagnostics"]
    assert payload["diagnostics"][0]["test_id"] == test_id
    assert (project_dir / ".flast" / "models" / "latest" / "model.joblib").exists()


def test_diagnostics_without_model_fails_cleanly(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    _create_project(project_dir)
    result = _run_cli(project_dir, "init", "pytest")
    assert result.returncode == 0, result.stderr

    result = _run_cli(project_dir, "diagnostics", "pytest", "--pytest-arg", "tests")
    assert result.returncode != 0
    assert "Missing trained model artifacts" in result.stderr
