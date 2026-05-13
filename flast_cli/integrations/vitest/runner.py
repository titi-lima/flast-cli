"""Vitest wrapper that emits FLAST run manifests."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...constants import (
    OUTCOME_ERROR,
    OUTCOME_FAILED,
    OUTCOME_PASSED,
    OUTCOME_SKIPPED,
    SCHEMA_VERSION,
)
from ...manifests import write_manifest
from ...schemas import CollectedTest, RunManifest, TestOutcome


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flast-vitest-runner")
    parser.add_argument("mode", choices=["collect", "observe"])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("vitest_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    vitest_args = _trim_separator(args.vitest_args)

    if args.mode == "collect":
        return _collect(project_root, manifest_path, vitest_args)
    return _observe(project_root, manifest_path, vitest_args)


def _trim_separator(args: Sequence[str]) -> List[str]:
    return list(args[1:] if args and args[0] == "--" else args)


def _collect(project_root: Path, manifest_path: Path, vitest_args: Sequence[str]) -> int:
    proc = subprocess.run(
        ["npx", "vitest", "list", *vitest_args],
        cwd=str(project_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        collected = [
            CollectedTest(test_id=test_id)
            for test_id in parse_vitest_list_output(proc.stdout, project_root)
        ]
        _write_manifest(
            manifest_path,
            framework="vitest",
            project_root=project_root,
            collected=collected,
            outcomes=[],
        )
    return proc.returncode


def _observe(project_root: Path, manifest_path: Path, vitest_args: Sequence[str]) -> int:
    with tempfile.TemporaryDirectory(prefix="flast-vitest-") as tmp_dir:
        result_path = Path(tmp_dir) / "vitest-results.json"
        proc = subprocess.run(
            [
                "npx",
                "vitest",
                "run",
                *vitest_args,
                "--reporter=json",
                f"--outputFile={result_path}",
            ],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            collected, outcomes = manifest_records_from_json(payload, project_root)
            _write_manifest(
                manifest_path,
                framework="vitest",
                project_root=project_root,
                collected=collected,
                outcomes=outcomes,
            )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def parse_vitest_list_output(stdout: str, project_root: Path) -> List[str]:
    test_ids: List[str] = []
    for raw_line in stdout.splitlines():
        line = _ANSI_RE.sub("", raw_line).strip()
        if not line or " > " not in line:
            continue
        file_part, *symbols = [part.strip() for part in line.split(" > ")]
        if not file_part or not symbols:
            continue
        test_ids.append(_test_id(file_part, symbols, project_root))
    return sorted(dict.fromkeys(test_ids))


def manifest_records_from_json(
    payload: Dict[str, Any],
    project_root: Path,
) -> Tuple[List[CollectedTest], List[TestOutcome]]:
    collected: Dict[str, CollectedTest] = {}
    outcomes: Dict[str, TestOutcome] = {}

    for suite in payload.get("testResults", []):
        file_part = _relative_file_part(str(suite.get("name", "")), project_root)
        for assertion in suite.get("assertionResults", []):
            symbols = [
                str(item)
                for item in assertion.get("ancestorTitles", [])
                if str(item)
            ]
            title = str(assertion.get("title", "")).strip()
            if title:
                symbols.append(title)
            if not file_part or not symbols:
                continue
            test_id = _test_id(file_part, symbols, project_root)
            collected[test_id] = CollectedTest(test_id=test_id)
            outcomes[test_id] = TestOutcome(
                test_id=test_id,
                outcome=_normalize_status(str(assertion.get("status", ""))),
            )

    return (
        [collected[key] for key in sorted(collected)],
        [outcomes[key] for key in sorted(outcomes)],
    )


def _normalize_status(status: str) -> str:
    normalized = status.lower()
    if normalized == "passed":
        return OUTCOME_PASSED
    if normalized == "failed":
        return OUTCOME_FAILED
    if normalized in {"skipped", "pending", "todo"}:
        return OUTCOME_SKIPPED
    return OUTCOME_ERROR


def _test_id(file_part: str, symbols: Sequence[str], project_root: Path) -> str:
    return "::".join([_relative_file_part(file_part, project_root), *symbols])


def _relative_file_part(file_part: str, project_root: Path) -> str:
    path = Path(file_part)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            return path.name
    return path.as_posix()


def _write_manifest(
    manifest_path: Path,
    *,
    framework: str,
    project_root: Path,
    collected: Iterable[CollectedTest],
    outcomes: Iterable[TestOutcome],
) -> None:
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        framework=framework,
        run_id=manifest_path.parent.name,
        project_root=str(project_root),
        tool={"name": "vitest", "version": "unknown"},
        collected=list(collected),
        outcomes=list(outcomes),
    )
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
