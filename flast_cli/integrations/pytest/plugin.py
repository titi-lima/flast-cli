"""Thin Pytest plugin that emits FLAST run manifests."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from ...constants import (
    OUTCOME_ERROR,
    OUTCOME_FAILED,
    OUTCOME_PASSED,
    OUTCOME_SKIPPED,
    OUTCOME_XFAIL,
    OUTCOME_XPASS,
    SCHEMA_VERSION,
)
from ...manifests import write_manifest
from ...schemas import CollectedTest, RunManifest, TestOutcome


def _canonicalize_report(report: pytest.TestReport) -> Optional[str]:
    wasxfail = getattr(report, "wasxfail", None)

    if report.when == "setup":
        if report.failed:
            return OUTCOME_ERROR
        if report.skipped:
            return OUTCOME_XFAIL if wasxfail else OUTCOME_SKIPPED
        return None

    if report.when == "call":
        if wasxfail:
            if report.skipped:
                return OUTCOME_XFAIL
            if report.passed or report.failed:
                return OUTCOME_XPASS
        if report.passed:
            return OUTCOME_PASSED
        if report.failed:
            return OUTCOME_FAILED
        if report.skipped:
            return OUTCOME_SKIPPED
        return None

    if report.when == "teardown" and report.failed:
        return OUTCOME_ERROR

    return None


class _ManifestWriterPlugin:
    def __init__(self, config: pytest.Config, manifest_path: Path) -> None:
        self._config = config
        self._manifest_path = manifest_path
        self._collected: List[str] = []
        self._outcomes: Dict[str, str] = {}

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self._collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        outcome = _canonicalize_report(report)
        if outcome is None:
            return
        if outcome == OUTCOME_ERROR:
            self._outcomes[report.nodeid] = outcome
            return
        existing = self._outcomes.get(report.nodeid)
        if existing == OUTCOME_ERROR:
            return
        self._outcomes[report.nodeid] = outcome

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        manifest = RunManifest(
            schema_version=SCHEMA_VERSION,
            framework="pytest",
            run_id=self._manifest_path.parent.name,
            project_root=str(Path(str(session.config.rootpath)).resolve()),
            tool={"name": "pytest", "version": pytest.__version__},
            collected=[CollectedTest(test_id=nodeid) for nodeid in self._collected],
            outcomes=[
                TestOutcome(test_id=test_id, outcome=outcome)
                for test_id, outcome in sorted(self._outcomes.items())
            ],
        )
        write_manifest(self._manifest_path, manifest)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flast")
    group.addoption(
        "--flast-manifest",
        action="store",
        default=None,
        help=argparse.SUPPRESS,
    )


def pytest_configure(config: pytest.Config) -> None:
    manifest_path = config.getoption("--flast-manifest")
    if not manifest_path:
        manifest_path = _auto_manifest_path(config)
    if not manifest_path:
        return
    config.pluginmanager.register(
        _ManifestWriterPlugin(config, Path(manifest_path)),
        "flast-manifest-writer",
    )


def _auto_manifest_path(config: pytest.Config) -> Optional[Path]:
    """Enable regular pytest runs as FLAST observations after `flast init pytest`."""
    try:
        collect_only = bool(config.getoption("--collect-only"))
    except (AssertionError, AttributeError, ValueError):
        collect_only = False
    if collect_only:
        return None

    rootpath = getattr(config, "rootpath", None)
    if rootpath is None:
        return None
    root = Path(str(rootpath)).resolve()
    workspace = root / ".flast"
    config_path = workspace / "config.toml"
    if not config_path.exists():
        return None

    return workspace / "runs" / str(uuid.uuid4()) / "manifest.json"
