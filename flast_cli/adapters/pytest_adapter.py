"""Pytest adapter for collection and observation runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .base import FrameworkAdapter
from ..features import resolve_feature_input
from ..manifests import load_manifest
from ..schemas import FeatureInput, PredictionRecord, RunManifest
from ..sources import python_module


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
            module = python_module(file_path)
        except (OSError, UnicodeDecodeError):
            return None
        target_node = module.find_function(prediction.class_name, prediction.test_name)
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
