"""Pytest adapter for collection and observation runs."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .base import FrameworkAdapter
from ..features import resolve_feature_input
from ..manifests import load_manifest
from ..schemas import FeatureInput, PredictionRecord, RunManifest


class PytestAdapter(FrameworkAdapter):
    framework = "pytest"

    def build_observe_invocation(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        framework_args: Sequence[str],
    ) -> List[str]:
        return self._base_invocation(
            manifest_path=manifest_path,
            framework_args=framework_args,
        )

    def build_collect_invocation(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        framework_args: Sequence[str],
    ) -> List[str]:
        return self._base_invocation(
            manifest_path=manifest_path,
            framework_args=["--collect-only", *framework_args],
        )

    def load_manifest(self, path: Path) -> RunManifest:
        return load_manifest(path)

    def resolve_feature_input(self, test_id: str, project_root: Path) -> FeatureInput:
        return resolve_feature_input(test_id, project_root)

    def resolve_prediction_line(
        self,
        prediction: PredictionRecord,
        project_root: Path,
    ) -> Optional[int]:
        if not prediction.file_path or not prediction.test_name:
            return None

        file_path = Path(prediction.file_path)
        if not file_path.is_absolute():
            file_path = project_root / file_path
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

    def run(
        self,
        *,
        project_root: Path,
        command: Sequence[str],
    ) -> Tuple[int, str, str]:
        env = os.environ.copy()
        package_root = Path(__file__).resolve().parents[2]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(package_root)
            if not existing
            else str(package_root) + os.pathsep + existing
        )
        proc = subprocess.run(
            list(command),
            cwd=str(project_root),
            env=env,
            text=True,
            capture_output=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def _base_invocation(
        self,
        *,
        manifest_path: Path,
        framework_args: Sequence[str],
    ) -> List[str]:
        return [
            sys.executable,
            "-m",
            "pytest",
            "--flast-manifest",
            str(manifest_path),
            *framework_args,
        ]


def _find_class(tree: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _find_function(nodes: Sequence[ast.stmt], test_name: str) -> Optional[ast.AST]:
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
            return node
    return None
