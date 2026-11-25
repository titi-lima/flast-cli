"""Feature extraction for FLAST prediction inputs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .schemas import FeatureInput
from .sources import python_module


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


JAVASCRIPT_COMMON_SIGNAL_RULES: Tuple[SignalRule, ...] = (
    SignalRule(
        "randomness",
        "uses random, crypto, or UUID generation",
        tokens=("crypto", "randomuuid"),
        fragments=("Math.random", "crypto.randomUUID", "randomUUID("),
    ),
    SignalRule(
        "time",
        "uses wall-clock or date/time APIs",
        tokens=("date", "performance"),
        fragments=("Date.now", "new Date", "performance.now"),
    ),
    SignalRule(
        "timing",
        "uses sleeps, timers, or timing-sensitive waits",
        tokens=("settimeout", "setinterval", "cleartimeout", "clearinterval"),
        fragments=("setTimeout", "setInterval", "clearTimeout", "clearInterval"),
    ),
    SignalRule(
        "filesystem state",
        "uses filesystem or shared local state",
        tokens=("fs", "node:fs", "readfilesync", "writefilesync", "mkdtempsync", "rmsync"),
        fragments=("fs.", "node:fs", "readFile", "writeFile", "mkdtemp", "rmSync"),
    ),
    SignalRule(
        "environment mutation",
        "reads or mutates process environment",
        tokens=("env",),
        fragments=("process.env",),
    ),
    SignalRule(
        "network",
        "uses network-facing APIs",
        tokens=("fetch", "axios", "http", "https", "undici"),
        fragments=("fetch(", "axios.", "http.", "https.", "undici"),
    ),
    SignalRule(
        "subprocess",
        "starts subprocesses or external commands",
        tokens=("child_process", "exec", "spawn", "execa"),
        fragments=("child_process", "exec(", "spawn(", "execa("),
    ),
    SignalRule(
        "concurrency",
        "uses threads, async, or concurrent execution",
        tokens=("promise", "worker", "worker_threads"),
        fragments=("Promise.", "Promise<", "Worker(", "worker_threads"),
    ),
)


COMMON_SIGNAL_RULES: Mapping[str, Tuple[SignalRule, ...]] = {
    "python": (
        SignalRule(
            "randomness",
            "uses random, crypto, or UUID generation",
            tokens=("random", "secrets", "uuid"),
            fragments=("random.", "secrets.", "uuid."),
        ),
        SignalRule(
            "time",
            "uses wall-clock or date/time APIs",
            tokens=("time", "datetime", "freezegun"),
            fragments=("time.", "datetime.", "freezegun."),
        ),
        SignalRule(
            "timing",
            "uses sleeps, timers, or timing-sensitive waits",
            tokens=("sleep",),
            fragments=("time.sleep",),
        ),
        SignalRule(
            "filesystem state",
            "uses filesystem or shared local state",
            tokens=("pathlib", "tempfile", "shutil"),
            fragments=("Path(", "open(", "os.path", "shutil.", "tempfile."),
        ),
        SignalRule(
            "environment mutation",
            "reads or mutates process environment",
            tokens=("environ", "getenv"),
            fragments=("os.environ", "os.getenv", "monkeypatch.setenv", "monkeypatch.delenv"),
        ),
        SignalRule(
            "network",
            "uses network-facing APIs",
            tokens=("requests", "httpx", "socket", "urllib"),
            fragments=("requests.", "httpx.", "socket.", "urllib."),
        ),
        SignalRule(
            "subprocess",
            "starts subprocesses or external commands",
            tokens=("subprocess", "multiprocessing"),
            fragments=("subprocess.", "multiprocessing."),
        ),
        SignalRule(
            "concurrency",
            "uses threads, async, or concurrent execution",
            tokens=("threading", "asyncio", "concurrent"),
            fragments=("threading.", "asyncio.", "concurrent."),
        ),
    ),
    "java": (
        SignalRule(
            "randomness",
            "uses random, crypto, or UUID generation",
            tokens=("random", "securerandom", "uuid"),
            fragments=("Random(", "SecureRandom", "UUID.randomUUID"),
        ),
        SignalRule(
            "time",
            "uses wall-clock or date/time APIs",
            tokens=("instant", "clock", "localdate", "localdatetime", "date"),
            fragments=("System.currentTimeMillis", "Instant.now", "LocalDate", "LocalDateTime", "new Date"),
        ),
        SignalRule(
            "timing",
            "uses sleeps, timers, or timing-sensitive waits",
            tokens=("sleep", "waitfor", "awaitility"),
            fragments=("Thread.sleep", "await().", ".waitFor("),
        ),
        SignalRule(
            "filesystem state",
            "uses filesystem or shared local state",
            tokens=("files", "paths", "path"),
            fragments=("Files.", "Paths.", "Path.of", "File("),
        ),
        SignalRule(
            "environment mutation",
            "reads or mutates process environment",
            tokens=("getenv", "getproperty", "setproperty", "clearproperty"),
            fragments=("System.getenv", "System.getProperty", "System.setProperty", "System.clearProperty"),
        ),
        SignalRule(
            "network",
            "uses network-facing APIs",
            tokens=("httpclient", "httpurlconnection", "socket", "okhttp"),
            fragments=("HttpClient", "HttpURLConnection", "new URL", "Socket(", "OkHttpClient"),
        ),
        SignalRule(
            "subprocess",
            "starts subprocesses or external commands",
            tokens=("processbuilder", "runtime"),
            fragments=("ProcessBuilder", "Runtime.getRuntime"),
        ),
        SignalRule(
            "concurrency",
            "uses threads, async, or concurrent execution",
            tokens=("thread", "executor", "executors", "completablefuture"),
            fragments=("new Thread", "ExecutorService", "Executors.", "CompletableFuture"),
        ),
    ),
    "javascript": JAVASCRIPT_COMMON_SIGNAL_RULES,
    "typescript": JAVASCRIPT_COMMON_SIGNAL_RULES,
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

    module = python_module(file_path)
    source_text = module.source
    target_node = module.find_function(class_name, test_name) if test_name else None
    if target_node is not None:
        source_text = module.node_source(target_node)

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
    """Return the static-test representation consumed by the FLAST model.

    The reference implementation feeds ``CountVectorizer`` the test source
    directly.  Identifiers and framework metadata are useful for reports, but
    including them here changes the token counts and therefore kNN distances.
    Keep the classifier input limited to source text so its feature space is
    the same as FLAST's.
    """
    return feature.source_text


def detect_static_signals(feature: FeatureInput) -> List[Dict[str, str]]:
    """Return lightweight source clues commonly associated with flaky tests."""
    source = _source_without_static_ignores(feature.source_text)
    signals: List[Dict[str, str]] = []

    def add(kind: str, detail: str) -> None:
        if not any(item["kind"] == kind for item in signals):
            signals.append({"kind": kind, "detail": detail})

    language = _feature_language(feature)
    tokens = _source_tokens(source)
    for rule in COMMON_SIGNAL_RULES.get(language, ()):
        if _matches_rule(rule, source, tokens):
            add(rule.kind, rule.detail)

    if language == "python":
        for kind, detail in _detect_python_ast_signals(source):
            add(kind, detail)

    if feature.test_id != feature.normalized_test_id:
        add("parametrization", "pytest parametrization can hide case-specific behavior")

    return signals


def _source_without_static_ignores(source: str) -> str:
    lines = source.splitlines()
    ignored_lines: Set[int] = set()
    ignore_next = False

    for index, line in enumerate(lines):
        if ignore_next:
            ignored_lines.add(index)
            ignore_next = False

        if "flast:ignore" not in line:
            continue

        ignored_lines.add(index)
        if _is_standalone_ignore_comment(line):
            ignore_next = True

    return "\n".join(
        "" if index in ignored_lines else line
        for index, line in enumerate(lines)
    )


def _is_standalone_ignore_comment(line: str) -> bool:
    before_directive = line.split("flast:ignore", 1)[0].strip()
    return before_directive in {"#", "//", "/*", "*", "<!--"}


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
