"""Feature extraction for FLAST prediction inputs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .schemas import FeatureInput


@dataclass(frozen=True)
class SignalRule:
    kind: str
    detail: str
    tokens: Tuple[str, ...] = ()
    fragments: Tuple[str, ...] = ()


LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "java": {".java"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx", ".mts", ".cts"},
}


COMMON_SIGNAL_RULES: Tuple[SignalRule, ...] = (
    SignalRule(
        "randomness",
        "uses random, crypto, or UUID generation",
        tokens=("random", "secrets", "uuid", "crypto", "math.random", "randomuuid", "secureRandom"),
        fragments=("Math.random", "randomUUID", "SecureRandom"),
    ),
    SignalRule(
        "time",
        "uses wall-clock or date/time APIs",
        tokens=("time", "datetime", "date", "instant", "clock", "localdate", "localdatetime"),
        fragments=("Date.now", "new Date", "System.currentTimeMillis", "Instant.now", "LocalDate"),
    ),
    SignalRule(
        "timing",
        "uses sleeps, timers, or timing-sensitive waits",
        tokens=("sleep", "settimeout", "setinterval", "waitfor", "awaitility"),
        fragments=("time.sleep", "Thread.sleep", "setTimeout", "setInterval", "waitFor"),
    ),
    SignalRule(
        "filesystem state",
        "uses filesystem or shared local state",
        tokens=("path", "pathlib", "tempfile", "shutil", "fs", "file", "files", "paths"),
        fragments=("Path(", "Files.", "Paths.", "fs.", "readFile", "writeFile"),
    ),
    SignalRule(
        "environment mutation",
        "reads or mutates process environment",
        tokens=("environ", "env", "getenv", "setproperty", "clearproperty"),
        fragments=("os.environ", "process.env", "System.getenv", "System.setProperty"),
    ),
    SignalRule(
        "network",
        "uses network-facing APIs",
        tokens=("requests", "httpx", "socket", "urllib", "fetch", "axios", "http", "https", "okhttp"),
        fragments=("fetch(", "axios.", "requests.", "httpx.", "new URL", "HttpClient"),
    ),
    SignalRule(
        "subprocess",
        "starts subprocesses or external commands",
        tokens=("subprocess", "multiprocessing", "child_process", "exec", "spawn", "processbuilder"),
        fragments=("subprocess.", "child_process", "ProcessBuilder", "Runtime.getRuntime"),
    ),
    SignalRule(
        "concurrency",
        "uses threads, async, or concurrent execution",
        tokens=("threading", "asyncio", "concurrent", "promise", "worker", "thread", "executor"),
        fragments=("Promise.", "Promise<", "CompletableFuture", "ExecutorService", "new Thread"),
    ),
)


LANGUAGE_SIGNAL_RULES = {
    "python": (
        SignalRule(
            "parametrization",
            "pytest parametrization can hide case-specific behavior",
            fragments=("parametrize",),
        ),
    ),
    "java": (
        SignalRule(
            "order dependence",
            "uses static or lifecycle state that can leak between tests",
            fragments=("static ", "@BeforeAll", "@AfterAll", "TestInstance"),
        ),
    ),
    "javascript": (
        SignalRule(
            "mock state",
            "uses test doubles or module mocks that may leak between cases",
            tokens=("vi", "jest", "mock", "spyon"),
            fragments=("vi.mock", "jest.mock", "spyOn"),
        ),
    ),
    "typescript": (
        SignalRule(
            "mock state",
            "uses test doubles or module mocks that may leak between cases",
            tokens=("vi", "jest", "mock", "spyon"),
            fragments=("vi.mock", "jest.mock", "spyOn"),
        ),
    ),
}


def strip_pytest_param_suffix(name: str) -> str:
    bracket_index = name.find("[")
    return name if bracket_index < 0 else name[:bracket_index]


def parse_pytest_test_id(test_id: str) -> Tuple[str, List[str], str]:
    parts = test_id.split("::")
    file_part = parts[0]
    symbol_parts = parts[1:]
    normalized_parts = list(symbol_parts)
    if normalized_parts:
        normalized_parts[-1] = strip_pytest_param_suffix(normalized_parts[-1])
    normalized_test_id = "::".join([file_part] + normalized_parts) if normalized_parts else file_part
    return file_part, symbol_parts, normalized_test_id


def _node_source(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment:
        return segment
    lines = source.splitlines()
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return source
    return "\n".join(lines[start - 1 : end])


def _find_class(tree: ast.Module, class_name: str) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _find_function(nodes: Sequence[ast.stmt], function_name: str) -> Optional[ast.AST]:
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None


def resolve_feature_input(test_id: str, project_root: Path) -> FeatureInput:
    file_part, symbol_parts, normalized_test_id = parse_pytest_test_id(test_id)
    file_path = (project_root / file_part).resolve()
    class_name = symbol_parts[-2] if len(symbol_parts) >= 2 else None
    test_name = strip_pytest_param_suffix(symbol_parts[-1]) if symbol_parts else None

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
    source_text = source
    try:
        tree = ast.parse(source)
        target_node = None
        if class_name:
            class_node = _find_class(tree, class_name)
            if class_node is not None and test_name:
                target_node = _find_function(class_node.body, test_name)
        elif test_name:
            target_node = _find_function(tree.body, test_name)
        if target_node is not None:
            source_text = _node_source(source, target_node)
    except SyntaxError:
        source_text = source

    return FeatureInput(
        test_id=test_id,
        normalized_test_id=normalized_test_id,
        project_root=project_root,
        file_path=file_path,
        class_name=class_name,
        test_name=test_name,
        source_text=source_text,
    )


def build_feature_text(feature: FeatureInput) -> str:
    parts = [f"test_id: {feature.normalized_test_id}"]
    if feature.class_name:
        parts.append(f"class_name: {feature.class_name}")
    if feature.test_name:
        parts.append(f"test_name: {feature.test_name}")
    parts.append("source:")
    parts.append(feature.source_text)
    return "\n".join(parts)


def detect_static_signals(feature: FeatureInput) -> List[Dict[str, str]]:
    """Return lightweight source clues commonly associated with flaky tests."""
    source = feature.source_text
    signals: List[Dict[str, str]] = []

    def add(kind: str, detail: str) -> None:
        if not any(item["kind"] == kind for item in signals):
            signals.append({"kind": kind, "detail": detail})

    language = _feature_language(feature)
    tokens = _source_tokens(source)
    for rule in (*COMMON_SIGNAL_RULES, *LANGUAGE_SIGNAL_RULES.get(language, ())):
        if _matches_rule(rule, source, tokens):
            add(rule.kind, rule.detail)

    if language == "python":
        for kind, detail in _detect_python_ast_signals(source):
            add(kind, detail)

    if feature.test_id != feature.normalized_test_id:
        add("parametrization", "pytest parametrization can hide case-specific behavior")

    return signals


def _feature_language(feature: FeatureInput) -> str:
    path_hint = feature.file_path or Path(feature.normalized_test_id.split("::", 1)[0])
    suffix = path_hint.suffix
    for language, extensions in LANGUAGE_EXTENSIONS.items():
        if suffix in extensions:
            return language
    return "unknown"


def _source_tokens(source: str) -> Set[str]:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_$.]*", source)}
    for token in list(tokens):
        tokens.update(part for part in token.split(".") if part)
    return tokens


def _matches_rule(rule: SignalRule, source: str, tokens: Set[str]) -> bool:
    lowered = source.lower()
    return any(token.lower() in tokens for token in rule.tokens) or any(
        fragment.lower() in lowered for fragment in rule.fragments
    )


def _detect_python_ast_signals(source: str) -> Iterable[Tuple[str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imported_names = set()
    called_names = set()
    assigned_env = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                root = alias.name.split(".")[0]
                imported_names.add(root)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                called_names.add(name)
        elif isinstance(node, ast.Subscript):
            target = _call_name(node.value)
            if target in {"os.environ", "environ"}:
                assigned_env = True

    signals = []
    all_names = imported_names | called_names
    if {"random", "secrets", "uuid"} & all_names:
        signals.append(("randomness", "uses random, crypto, or UUID generation"))
    if {"time", "datetime", "freezegun"} & all_names:
        signals.append(("time", "uses wall-clock or date/time APIs"))
    if "sleep" in {name.rsplit(".", 1)[-1] for name in called_names}:
        signals.append(("timing", "uses sleeps, timers, or timing-sensitive waits"))
    if (
        {"pathlib", "tempfile", "shutil"} & all_names
        or "Path" in called_names
        or any(name.startswith(("Path.", "open", "os.path", "shutil.", "tempfile.")) for name in called_names)
    ):
        signals.append(("filesystem state", "uses filesystem or shared local state"))
    if assigned_env:
        signals.append(("environment mutation", "reads or mutates process environment"))
    if {"requests", "httpx", "socket", "urllib"} & all_names or any(
        name.startswith(("requests.", "httpx.", "socket.")) for name in called_names
    ):
        signals.append(("network", "uses network-facing APIs"))
    if {"subprocess", "multiprocessing"} & all_names or any(
        name.startswith(("subprocess.", "multiprocessing.")) for name in called_names
    ):
        signals.append(("subprocess", "starts subprocesses or external commands"))
    if {"threading", "asyncio", "concurrent"} & all_names or any(
        name.startswith(("threading.", "asyncio.", "concurrent.")) for name in called_names
    ):
        signals.append(("concurrency", "uses threads, async, or concurrent execution"))
    return signals


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None
