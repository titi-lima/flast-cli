"""Pytest adapter for collection and observation runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .base import FrameworkAdapter
from ..features import resolve_feature_input
from ..manifests import load_manifest
from ..schemas import FeatureInput, RunManifest


class PytestAdapter(FrameworkAdapter):
    framework = "pytest"

    def build_observe_invocation(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        pytest_args: Sequence[str],
    ) -> List[str]:
        return self._base_invocation(manifest_path=manifest_path, pytest_args=pytest_args)

    def build_collect_invocation(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        pytest_args: Sequence[str],
    ) -> List[str]:
        return self._base_invocation(
            manifest_path=manifest_path,
            pytest_args=["--collect-only", *pytest_args],
        )

    def load_manifest(self, path: Path) -> RunManifest:
        return load_manifest(path)

    def resolve_feature_input(self, test_id: str, project_root: Path) -> FeatureInput:
        return resolve_feature_input(test_id, project_root)

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
        pytest_args: Sequence[str],
    ) -> List[str]:
        return [
            sys.executable,
            "-m",
            "pytest",
            "--flast-manifest",
            str(manifest_path),
            *pytest_args,
        ]
