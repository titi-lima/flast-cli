"""Typed schemas for manifests, corpus records, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CollectedTest:
    test_id: str

    def to_dict(self) -> Dict[str, str]:
        return {"test_id": self.test_id}


@dataclass(frozen=True)
class TestOutcome:
    test_id: str
    outcome: str

    def to_dict(self) -> Dict[str, str]:
        return {"test_id": self.test_id, "outcome": self.outcome}


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    framework: str
    run_id: str
    project_root: str
    pytest_version: str
    collected: List[CollectedTest] = field(default_factory=list)
    outcomes: List[TestOutcome] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "framework": self.framework,
            "run_id": self.run_id,
            "project_root": self.project_root,
            "pytest_version": self.pytest_version,
            "collected": [item.to_dict() for item in self.collected],
            "outcomes": [item.to_dict() for item in self.outcomes],
        }


@dataclass(frozen=True)
class FeatureInput:
    test_id: str
    normalized_test_id: str
    project_root: Path
    file_path: Optional[Path]
    class_name: Optional[str]
    test_name: Optional[str]
    source_text: str


@dataclass(frozen=True)
class ObservationRecord:
    schema_version: int
    framework: str
    test_id: str
    project_root: str
    total_observations: int
    outcome_counts: Dict[str, int]
    seen_outcomes: List[str]
    run_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "framework": self.framework,
            "test_id": self.test_id,
            "project_root": self.project_root,
            "total_observations": self.total_observations,
            "outcome_counts": dict(self.outcome_counts),
            "seen_outcomes": list(self.seen_outcomes),
            "run_ids": list(self.run_ids),
        }


@dataclass(frozen=True)
class LabelRecord:
    schema_version: int
    framework: str
    test_id: str
    project_root: str
    label: str
    reason: str
    total_observations: int
    outcome_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "framework": self.framework,
            "test_id": self.test_id,
            "project_root": self.project_root,
            "label": self.label,
            "reason": self.reason,
            "total_observations": self.total_observations,
            "outcome_counts": dict(self.outcome_counts),
        }


@dataclass(frozen=True)
class PredictionRecord:
    test_id: str
    score: float
    predicted_label: str
    class_name: Optional[str]
    test_name: Optional[str]
    file_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "score": self.score,
            "predicted_label": self.predicted_label,
            "class_name": self.class_name,
            "test_name": self.test_name,
            "file_path": self.file_path,
        }


@dataclass(frozen=True)
class PredictionReport:
    schema_version: int
    run_id: str
    framework: str
    project_root: str
    model_metadata: Dict[str, Any]
    predictions: List[PredictionRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "framework": self.framework,
            "project_root": self.project_root,
            "model_metadata": dict(self.model_metadata),
            "predictions": [item.to_dict() for item in self.predictions],
        }
