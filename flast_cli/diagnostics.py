"""Editor-oriented diagnostics for FLAST predictions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "score": self.score,
            "predicted_label": self.predicted_label,
            "class_name": self.class_name,
            "test_name": self.test_name,
            "file_path": self.file_path,
            "line": self.line,
            "message": self.message,
        }


def _find_class(tree: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _find_function(nodes: List[ast.stmt], test_name: str) -> Optional[ast.AST]:
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
            return node
    return None


def resolve_prediction_line(prediction: PredictionRecord) -> Optional[int]:
    if not prediction.file_path or not prediction.test_name:
        return None

    file_path = Path(prediction.file_path)
    if not file_path.exists():
        return None

    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    target_node: Optional[ast.AST] = None
    if prediction.class_name:
        class_node = _find_class(tree, prediction.class_name)
        if class_node is not None:
            target_node = _find_function(class_node.body, prediction.test_name)
    else:
        target_node = _find_function(tree.body, prediction.test_name)

    line = getattr(target_node, "lineno", None)
    return int(line) if line is not None else None


def diagnostic_message(prediction: PredictionRecord) -> str:
    return (
        "FLAST predicts {label} risk for {test_id} "
        "(score {score:.3f})."
    ).format(
        label=prediction.predicted_label.replace("_", " "),
        test_id=prediction.test_id,
        score=prediction.score,
    )


def build_diagnostics(report: PredictionReport) -> Dict[str, Any]:
    diagnostics = [
        DiagnosticRecord(
            test_id=prediction.test_id,
            score=prediction.score,
            predicted_label=prediction.predicted_label,
            class_name=prediction.class_name,
            test_name=prediction.test_name,
            file_path=prediction.file_path,
            line=resolve_prediction_line(prediction),
            message=diagnostic_message(prediction),
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
