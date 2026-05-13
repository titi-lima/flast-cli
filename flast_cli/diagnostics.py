"""Editor-oriented diagnostics for FLAST predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .schemas import PredictionRecord, PredictionReport


@dataclass(frozen=True)
class DiagnosticRecord:
    test_id: str
    score: float
    predicted_label: str
    class_name: Optional[str]
    test_name: Optional[str]
    file_path: Optional[str]
    line: Optional[int]
    message: str
    explanation: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "test_id": self.test_id,
            "score": self.score,
            "predicted_label": self.predicted_label,
            "class_name": self.class_name,
            "test_name": self.test_name,
            "file_path": self.file_path,
            "line": self.line,
            "message": self.message,
        }
        if self.explanation is not None:
            payload["explanation"] = dict(self.explanation)
        return payload


def diagnostic_message(prediction: PredictionRecord) -> str:
    prefix = (
        "FLAST predicts {label} risk for {test_id} "
        "(score {score:.3f})"
    ).format(
        label=prediction.predicted_label.replace("_", " "),
        test_id=prediction.test_id,
        score=prediction.score,
    )
    explanation = prediction.explanation or {}
    summary = explanation.get("summary")
    if summary:
        return f"{prefix}: {summary}."
    return f"{prefix}."


def build_diagnostics(report: PredictionReport) -> Dict[str, Any]:
    diagnostics = [
        DiagnosticRecord(
            test_id=prediction.test_id,
            score=prediction.score,
            predicted_label=prediction.predicted_label,
            class_name=prediction.class_name,
            test_name=prediction.test_name,
            file_path=prediction.file_path,
            line=prediction.line,
            message=diagnostic_message(prediction),
            explanation=prediction.explanation,
        ).to_dict()
        for prediction in report.predictions
    ]
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "framework": report.framework,
        "project_root": report.project_root,
        "model_metadata": dict(report.model_metadata),
        "diagnostics": diagnostics,
    }
