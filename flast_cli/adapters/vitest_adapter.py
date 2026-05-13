"""Vitest adapter for collection and observation runs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .base import FrameworkAdapter
from ..manifests import load_manifest
from ..schemas import FeatureInput, PredictionRecord, RunManifest


class VitestAdapter(FrameworkAdapter):
    framework = "vitest"

    def build_observe_invocation(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        framework_args: Sequence[str],
    ) -> List[str]:
        return self._runner_invocation(
            mode="observe",
            project_root=project_root,
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
        return self._runner_invocation(
            mode="collect",
            project_root=project_root,
            manifest_path=manifest_path,
            framework_args=framework_args,
        )

    def load_manifest(self, path: Path) -> RunManifest:
        return load_manifest(path)

    def resolve_feature_input(self, test_id: str, project_root: Path) -> FeatureInput:
        file_part, symbol_parts, normalized_test_id = parse_vitest_test_id(test_id)
        file_path = (project_root / file_part).resolve()
        test_name = symbol_parts[-1] if symbol_parts else None
        class_name = " > ".join(symbol_parts[:-1]) if len(symbol_parts) >= 2 else None

        if not file_path.exists():
            return FeatureInput(
                test_id=test_id,
                normalized_test_id=normalized_test_id,
                project_root=project_root,
                file_path=None,
                class_name=class_name,
                test_name=test_name,
                source_text=normalized_test_id,
            )

        source = file_path.read_text(encoding="utf-8")
        source_text = _find_vitest_test_source(source, test_name) if test_name else None
        return FeatureInput(
            test_id=test_id,
            normalized_test_id=normalized_test_id,
            project_root=project_root,
            file_path=file_path,
            class_name=class_name,
            test_name=test_name,
            source_text=source_text or source,
        )

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

        source = file_path.read_text(encoding="utf-8")
        return _find_vitest_test_line(source, prediction.test_name)

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

    def _runner_invocation(
        self,
        *,
        mode: str,
        project_root: Path,
        manifest_path: Path,
        framework_args: Sequence[str],
    ) -> List[str]:
        return [
            sys.executable,
            "-m",
            "flast_cli.integrations.vitest.runner",
            "--project-root",
            str(project_root),
            "--manifest",
            str(manifest_path),
            mode,
            "--",
            *framework_args,
        ]


def parse_vitest_test_id(test_id: str) -> Tuple[str, List[str], str]:
    parts = test_id.split("::")
    file_part = parts[0]
    symbol_parts = parts[1:]
    normalized_test_id = "::".join([file_part, *symbol_parts]) if symbol_parts else file_part
    return file_part, symbol_parts, normalized_test_id


def _find_vitest_test_source(source: str, test_name: str) -> Optional[str]:
    start = _find_vitest_test_offset(source, test_name)
    if start is None:
        return None
    end = _find_call_end(source, start)
    return source[start:end].strip() if end is not None else source[start:].strip()


def _find_vitest_test_line(source: str, test_name: str) -> Optional[int]:
    start = _find_vitest_test_offset(source, test_name)
    if start is None:
        return None
    return source.count("\n", 0, start) + 1


def _find_vitest_test_offset(source: str, test_name: str) -> Optional[int]:
    escaped = re.escape(test_name)
    for quote in ("'", '"', "`"):
        pattern = re.compile(
            r"\b(?:it|test)(?:\.\w+)?\s*\(\s*" + re.escape(quote) + escaped + re.escape(quote),
            re.MULTILINE,
        )
        match = pattern.search(source)
        if match:
            return match.start()
    return None


def _find_call_end(source: str, start: int) -> Optional[int]:
    open_index = source.find("(", start)
    if open_index < 0:
        return None
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index in range(open_index, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                if end < len(source) and source[end] == ";":
                    end += 1
                return end
    return None
