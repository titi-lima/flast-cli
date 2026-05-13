"""JUnit Console wrapper that emits FLAST run manifests."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ...constants import OUTCOME_ERROR, OUTCOME_FAILED, OUTCOME_PASSED, OUTCOME_SKIPPED, SCHEMA_VERSION
from ...manifests import write_manifest
from ...schemas import CollectedTest, RunManifest, TestOutcome


@dataclass(frozen=True)
class JUnitTest:
    test_id: str
    file_path: Path
    class_name: str
    method_name: str


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flast-junit-runner")
    parser.add_argument("mode", choices=["collect", "observe"])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("junit_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    junit_args = _trim_separator(args.junit_args)

    if args.mode == "collect":
        tests = discover_junit_tests(project_root)
        _write_manifest(manifest_path, project_root=project_root, tests=tests, outcomes=[])
        return 0
    return _observe(project_root, manifest_path, junit_args)


def discover_junit_tests(project_root: Path) -> List[JUnitTest]:
    tests_dir = project_root / "src" / "test" / "java"
    tests: List[JUnitTest] = []
    for java_file in sorted(tests_dir.rglob("*.java")):
        source = java_file.read_text(encoding="utf-8")
        package_name = _extract_package(source)
        class_name = _extract_class(source)
        qualified_class = f"{package_name}.{class_name}" if package_name else class_name
        relative = java_file.relative_to(project_root).as_posix()
        for method_name in extract_test_methods(source):
            tests.append(
                JUnitTest(
                    test_id=f"{relative}::{qualified_class}::{method_name}",
                    file_path=java_file,
                    class_name=qualified_class,
                    method_name=method_name,
                )
            )
    return tests


def extract_test_methods(source: str) -> List[str]:
    text = _strip_comments(source)
    methods: List[str] = []
    for chunk in re.split(r"@(?:org\.junit\.jupiter\.api\.)?Test\b", text)[1:]:
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)[^{;]*\{", chunk)
        if match:
            methods.append(match.group(1))
    return methods


def _observe(project_root: Path, manifest_path: Path, junit_args: Sequence[str]) -> int:
    compile_result = _compile_project(project_root)
    if compile_result.returncode != 0:
        sys.stdout.write(compile_result.stdout)
        sys.stderr.write(compile_result.stderr)
        return compile_result.returncode

    tests = discover_junit_tests(project_root)
    with tempfile.TemporaryDirectory(prefix="flast-junit-") as tmp_dir:
        reports_dir = Path(tmp_dir) / "reports"
        reports_dir.mkdir()
        jar = _find_console_jar(project_root)
        if jar is None:
            sys.stderr.write(
                "JUnit Console jar not found. Set JUNIT_PLATFORM_CONSOLE_JAR or place "
                "junit-platform-console-standalone.jar under lib/.\n"
            )
            return 2
        class_path = _junit_class_path(project_root)
        cmd = [
            "java",
            "-jar",
            str(jar),
            "--class-path",
            class_path,
            "--scan-class-path",
            "--reports-dir",
            str(reports_dir),
            *junit_args,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        outcomes = parse_junit_xml_reports(reports_dir, tests)
        if outcomes:
            _write_manifest(
                manifest_path,
                project_root=project_root,
                tests=tests,
                outcomes=outcomes,
            )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode


def parse_junit_xml_reports(reports_dir: Path, tests: Sequence[JUnitTest]) -> List[TestOutcome]:
    by_selector = {
        (test.class_name, test.method_name): test.test_id
        for test in tests
    }
    outcomes = {}
    for report in sorted(reports_dir.glob("TEST-*.xml")):
        root = ET.parse(report).getroot()
        for case in root.iter("testcase"):
            class_name = case.attrib.get("classname", "")
            method_name = _normalize_case_method_name(case.attrib.get("name", ""))
            test_id = by_selector.get((class_name, method_name))
            if test_id is None:
                continue
            outcomes[test_id] = TestOutcome(
                test_id=test_id,
                outcome=_xml_case_outcome(case),
            )
    return [outcomes[key] for key in sorted(outcomes)]


def _xml_case_outcome(case: ET.Element) -> str:
    if case.find("skipped") is not None:
        return OUTCOME_SKIPPED
    if case.find("failure") is not None:
        return OUTCOME_FAILED
    if case.find("error") is not None:
        return OUTCOME_ERROR
    return OUTCOME_PASSED


def _normalize_case_method_name(name: str) -> str:
    return re.sub(r"\(.*\)$", "", name)


def _compile_project(project_root: Path) -> subprocess.CompletedProcess:
    if (project_root / "build.xml").exists():
        return subprocess.run(
            ["ant", "compile"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
    if (project_root / "pom.xml").exists():
        return subprocess.run(
            ["mvn", "test-compile"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _find_console_jar(project_root: Path) -> Optional[Path]:
    env_path = os.environ.get("JUNIT_PLATFORM_CONSOLE_JAR")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            return candidate
    candidates = [
        project_root / "lib" / "junit-platform-console-standalone.jar",
        *sorted((project_root / "lib").glob("junit-platform-console-standalone-*.jar")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _junit_class_path(project_root: Path) -> str:
    paths = [
        project_root / "build" / "classes" / "main",
        project_root / "build" / "classes" / "test",
        project_root / "target" / "classes",
        project_root / "target" / "test-classes",
    ]
    return os.pathsep.join(str(path) for path in paths if path.exists())


def _write_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    tests: Iterable[JUnitTest],
    outcomes: Iterable[TestOutcome],
) -> None:
    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        framework="junit",
        run_id=manifest_path.parent.name,
        project_root=str(project_root),
        tool={"name": "junit-platform-console", "version": "unknown"},
        collected=[CollectedTest(test_id=test.test_id) for test in tests],
        outcomes=list(outcomes),
    )
    write_manifest(manifest_path, manifest)


def _trim_separator(args: Sequence[str]) -> List[str]:
    return list(args[1:] if args and args[0] == "--" else args)


def _extract_package(source: str) -> str:
    match = re.search(r"\bpackage\s+([\w.]+)\s*;", source)
    return match.group(1) if match else ""


def _extract_class(source: str) -> str:
    match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", source)
    return match.group(1) if match else "UnknownTestClass"


def _strip_comments(source: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"//.*", " ", no_block)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
