from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flast_cli.integrations.pytest.plugin import _canonicalize_report, pytest_configure


class _DummyPluginManager:
    def __init__(self) -> None:
        self.registered = []

    def register(self, plugin, name: str) -> None:
        self.registered.append((plugin, name))


class _DummyConfig:
    def __init__(self, option):
        self._option = option
        self.pluginmanager = _DummyPluginManager()

    def getoption(self, name: str):
        assert name == "--flast-manifest"
        return self._option


class _FakeReport:
    def __init__(self, *, when: str, outcome: str, wasxfail=None) -> None:
        self.when = when
        self.outcome = outcome
        self.wasxfail = wasxfail

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"

    @property
    def skipped(self) -> bool:
        return self.outcome == "skipped"


def _plugin_env() -> dict:
    env = os.environ.copy()
    package_root = Path(__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(package_root)
        if not existing
        else str(package_root) + os.pathsep + existing
    )
    return env


def test_pytest_configure_is_noop_without_manifest_option() -> None:
    config = _DummyConfig(None)
    pytest_configure(config)
    assert config.pluginmanager.registered == []


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (_FakeReport(when="call", outcome="passed"), "passed"),
        (_FakeReport(when="call", outcome="failed"), "failed"),
        (_FakeReport(when="setup", outcome="failed"), "error"),
        (_FakeReport(when="call", outcome="skipped"), "skipped"),
        (_FakeReport(when="call", outcome="skipped", wasxfail="reason"), "xfail"),
        (_FakeReport(when="call", outcome="passed", wasxfail="reason"), "xpass"),
    ],
)
def test_canonicalize_report(report, expected: str) -> None:
    assert _canonicalize_report(report) == expected


def test_plugin_writes_manifest_for_collect_only_run(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    tests_dir = project_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n",
        encoding="utf-8",
    )
    manifest_path = project_dir / "manifests" / "collect-only" / "manifest.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "--flast-manifest",
        str(manifest_path),
        "tests",
    ]
    result = subprocess.run(
        cmd,
        cwd=project_dir,
        env=_plugin_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["collected"] == [{"test_id": "tests/test_sample.py::test_sample"}]
    assert payload["outcomes"] == []


def test_plugin_writes_outcomes_for_executed_run(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    tests_dir = project_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_outcomes.py").write_text(
        """
import pytest

def test_pass():
    assert True

def test_fail():
    assert False

@pytest.mark.skip(reason="skip")
def test_skip():
    assert True

@pytest.mark.xfail(reason="known")
def test_xfail():
    assert False
    """.strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path = project_dir / "manifests" / "executed" / "manifest.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--flast-manifest",
        str(manifest_path),
        "tests",
    ]
    result = subprocess.run(
        cmd,
        cwd=project_dir,
        env=_plugin_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outcomes = {item["test_id"]: item["outcome"] for item in payload["outcomes"]}
    assert outcomes["tests/test_outcomes.py::test_pass"] == "passed"
    assert outcomes["tests/test_outcomes.py::test_fail"] == "failed"
    assert outcomes["tests/test_outcomes.py::test_skip"] == "skipped"
    assert outcomes["tests/test_outcomes.py::test_xfail"] == "xfail"
