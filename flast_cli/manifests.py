"""Manifest and JSONL serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .constants import SCHEMA_VERSION
from .schemas import (
    CollectedTest,
    LabelRecord,
    ObservationRecord,
    PredictionRecord,
    PredictionReport,
    RunManifest,
    TestOutcome,
)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: RunManifest) -> None:
    write_json(path, manifest.to_dict())


def load_manifest(path: Path) -> RunManifest:
    payload = read_json(path)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version {version!r} in {path}"
        )
    collected = [CollectedTest(**item) for item in payload.get("collected", [])]
    outcomes = [TestOutcome(**item) for item in payload.get("outcomes", [])]
    return RunManifest(
        schema_version=version,
        framework=payload["framework"],
        run_id=payload["run_id"],
        project_root=payload["project_root"],
        tool=payload.get(
            "tool",
            {
                "name": payload["framework"],
                "version": payload.get("pytest_version", "unknown"),
            },
        ),
        collected=collected,
        outcomes=outcomes,
    )


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_observations(path: Path) -> List[ObservationRecord]:
    return [ObservationRecord(**row) for row in read_jsonl(path)]


def load_labels(path: Path) -> List[LabelRecord]:
    return [LabelRecord(**row) for row in read_jsonl(path)]


def write_observations(path: Path, records: Iterable[ObservationRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def write_labels(path: Path, records: Iterable[LabelRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def write_report(path: Path, report: PredictionReport) -> None:
    write_json(path, report.to_dict())


def load_report(path: Path) -> PredictionReport:
    payload = read_json(path)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported report schema_version {version!r} in {path}")
    predictions = [
        PredictionRecord(**item) for item in payload.get("predictions", [])
    ]
    return PredictionReport(
        schema_version=version,
        run_id=payload["run_id"],
        framework=payload["framework"],
        project_root=payload["project_root"],
        model_metadata=payload.get("model_metadata", {}),
        predictions=predictions,
    )
