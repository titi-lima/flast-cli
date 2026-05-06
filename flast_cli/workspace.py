"""Workspace layout helpers for `.flast/`."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .constants import (
    DEFAULT_ALGORITHM,
    DEFAULT_EPS,
    DEFAULT_FRAMEWORK,
    DEFAULT_K,
    DEFAULT_METRIC,
    DEFAULT_MIN_OBSERVATIONS,
    DEFAULT_RUNS,
    DEFAULT_SIGMA,
    DEFAULT_WEIGHTS,
    FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION,
)

try:  # pragma: no cover - Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]


USAGE_TEXT = """# FLAST workspace

This directory is managed by `flast`.

- `runs/` stores normalized Pytest manifests for observed and predicted runs.
- `corpus/observations.jsonl` stores aggregated per-test outcome history.
- `corpus/labels.jsonl` stores derived labels (`flaky`, `non_flaky`, `unlabeled`).
- `models/latest/` stores the latest trained model artifacts.
- `reports/` stores prediction reports.

Typical flow:

1. `flast observe pytest --runs 10`
2. `flast train`
3. `flast predict pytest`
"""


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    workspace_dir: Path
    config_path: Path
    usage_path: Path
    runs_dir: Path
    corpus_dir: Path
    observations_path: Path
    labels_path: Path
    models_dir: Path
    latest_model_dir: Path
    reports_dir: Path


def workspace_paths(project_root: Path) -> WorkspacePaths:
    workspace_dir = project_root / ".flast"
    return WorkspacePaths(
        root=project_root,
        workspace_dir=workspace_dir,
        config_path=workspace_dir / "config.toml",
        usage_path=workspace_dir / "USAGE.md",
        runs_dir=workspace_dir / "runs",
        corpus_dir=workspace_dir / "corpus",
        observations_path=workspace_dir / "corpus" / "observations.jsonl",
        labels_path=workspace_dir / "corpus" / "labels.jsonl",
        models_dir=workspace_dir / "models",
        latest_model_dir=workspace_dir / "models" / "latest",
        reports_dir=workspace_dir / "reports",
    )


def _default_config_text() -> str:
    return f"""schema_version = {SCHEMA_VERSION}

[workspace]
frameworks = ["{DEFAULT_FRAMEWORK}"]

[observe]
default_runs = {DEFAULT_RUNS}
min_observations = {DEFAULT_MIN_OBSERVATIONS}

[model]
feature_schema_version = {FEATURE_SCHEMA_VERSION}
algorithm = "{DEFAULT_ALGORITHM}"
metric = "{DEFAULT_METRIC}"
weights = "{DEFAULT_WEIGHTS}"
k = {DEFAULT_K}
sigma = {DEFAULT_SIGMA}
eps = {DEFAULT_EPS}
"""


def ensure_workspace(project_root: Path) -> WorkspacePaths:
    paths = workspace_paths(project_root)
    paths.workspace_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.corpus_dir.mkdir(parents=True, exist_ok=True)
    paths.latest_model_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    if not paths.config_path.exists():
        paths.config_path.write_text(_default_config_text(), encoding="utf-8")
    if not paths.usage_path.exists():
        paths.usage_path.write_text(USAGE_TEXT, encoding="utf-8")
    return paths


def load_config(project_root: Path) -> dict:
    paths = ensure_workspace(project_root)
    with paths.config_path.open("rb") as handle:
        return tomllib.load(handle)


def run_manifest_path(project_root: Path, run_id: str) -> Path:
    return ensure_workspace(project_root).runs_dir / run_id / "manifest.json"


def report_path(project_root: Path, run_id: str) -> Path:
    return ensure_workspace(project_root).reports_dir / f"{run_id}.json"


def model_artifact_path(project_root: Path, name: str) -> Path:
    return ensure_workspace(project_root).latest_model_dir / name


def latest_report_path(project_root: Path) -> Path:
    reports = sorted(
        ensure_workspace(project_root).reports_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise FileNotFoundError(
            f"No reports found under {ensure_workspace(project_root).reports_dir}"
        )
    return reports[-1]


def list_workspace_manifests(project_root: Path) -> List[Path]:
    return sorted(ensure_workspace(project_root).runs_dir.glob("*/manifest.json"))


def import_manifest(project_root: Path, source_path: Path, run_id: str) -> Path:
    destination = run_manifest_path(project_root, run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source_path, destination)
    return destination
