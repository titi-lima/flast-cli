"""Corpus rebuilding and manifest import helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .labels import aggregate_manifests, derive_labels
from .manifests import write_labels, write_observations
from .schemas import LabelRecord, ObservationRecord, RunManifest
from .workspace import ensure_workspace, import_manifest, list_workspace_manifests


def iter_manifest_sources(paths: Sequence[str]) -> List[Path]:
    manifest_paths: List[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.is_dir():
            manifest_paths.extend(sorted(candidate.rglob("*.json")))
            continue
        manifest_paths.append(candidate)
    return manifest_paths


def sync_manifest_history(
    *,
    project_root: Path,
    manifests: Iterable[RunManifest],
    source_paths: Iterable[Path],
) -> List[Path]:
    synced: List[Path] = []
    ensure_workspace(project_root)
    for manifest, source_path in zip(manifests, source_paths):
        synced.append(import_manifest(project_root, source_path, manifest.run_id))
    return synced


def rebuild_corpus(
    *,
    project_root: Path,
    manifests: Sequence[RunManifest],
    min_observations: int,
) -> tuple[List[ObservationRecord], List[LabelRecord]]:
    paths = ensure_workspace(project_root)
    observations = aggregate_manifests(manifests)
    labels = derive_labels(observations, min_observations=min_observations)
    write_observations(paths.observations_path, observations)
    write_labels(paths.labels_path, labels)
    return observations, labels
