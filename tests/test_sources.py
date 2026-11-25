"""Tests for the per-process source cache used by source resolution."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from flast_cli import sources
from flast_cli.adapters import PytestAdapter
from flast_cli.features import resolve_feature_input
from flast_cli.schemas import PredictionRecord

TRICKY_MODULE = '''\
import pytest


@pytest.mark.parametrize("x", [1, 2])
def test_decorated(x):
    assert x  # café — non-ASCII before the end column é


async def test_async():
    value = "üñïçødé"; assert value


def test_formfeed():
    a = 1
    \f
    assert a


class TestOuter:
    def test_method(self):
        assert "x y"

    class TestInner:
        def test_nested(self):
            assert True


class TestOuter:
    def test_method(self):
        assert "second definition"


def test_decorated(x):
    assert "duplicate top-level name"


def test_one_line(): assert True
'''


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fresh_cache():
    sources.clear_cache()
    yield
    sources.clear_cache()


def test_node_source_matches_ast_get_source_segment(tmp_path: Path) -> None:
    path = _write(tmp_path / "tests" / "test_tricky.py", TRICKY_MODULE)
    module = sources.python_module(path)
    source = path.read_text(encoding="utf-8")

    assert module.source == source
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.Assert)):
            assert module.node_source(node) == ast.get_source_segment(source, node)


def test_index_keeps_first_match_semantics(tmp_path: Path) -> None:
    path = _write(tmp_path / "tests" / "test_tricky.py", TRICKY_MODULE)
    module = sources.python_module(path)

    decorated = module.find_function(None, "test_decorated")
    assert "assert x" in module.node_source(decorated)
    method = module.find_function("TestOuter", "test_method")
    assert "second definition" not in module.node_source(method)
    # Nested classes are not looked up, as before.
    assert module.find_function("TestInner", "test_nested") is None
    assert module.find_function(None, "test_missing") is None


def test_resolve_feature_input_extracts_exact_function_source(tmp_path: Path) -> None:
    path = _write(tmp_path / "tests" / "test_tricky.py", TRICKY_MODULE)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    expected = ast.get_source_segment(source, next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
    ))

    feature = resolve_feature_input("tests/test_tricky.py::test_async", tmp_path)

    assert feature.source_text == expected


def test_syntax_error_falls_back_to_whole_file_and_no_line(tmp_path: Path) -> None:
    text = "def test_broken(:\n    pass\n"
    path = _write(tmp_path / "tests" / "test_broken.py", text)

    feature = resolve_feature_input("tests/test_broken.py::test_broken", tmp_path)
    line = PytestAdapter().resolve_prediction_line(
        PredictionRecord(
            test_id="tests/test_broken.py::test_broken",
            score=1.0,
            predicted_label="flaky",
            class_name=None,
            test_name="test_broken",
            file_path=str(path),
        ),
        tmp_path,
    )

    assert feature.source_text == text
    assert line is None


def test_module_is_parsed_once_per_version(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path / "tests" / "test_a.py", "def test_a():\n    assert True\n\ndef test_b():\n    assert True\n")
    calls = []
    real_parse = ast.parse
    monkeypatch.setattr(sources.ast, "parse", lambda *a, **k: calls.append(1) or real_parse(*a, **k))

    for name in ("test_a", "test_b", "test_a"):
        resolve_feature_input(f"tests/test_a.py::{name}", tmp_path)

    assert len(calls) == 1


def test_changed_file_is_reread(tmp_path: Path) -> None:
    path = _write(tmp_path / "tests" / "test_a.py", "def test_a():\n    assert 1\n")
    first = resolve_feature_input("tests/test_a.py::test_a", tmp_path)

    path.write_text("def test_a():\n    assert 'changed and longer'\n", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = resolve_feature_input("tests/test_a.py::test_a", tmp_path)

    assert "assert 1" in first.source_text
    assert "changed and longer" in second.source_text


def test_split_lines_matches_ast_line_splitting() -> None:
    text = "a\r\nb\rc\nd\fe\x85f g\nlast"
    assert sources._split_lines(text) == ("a\r\n", "b\r", "c\n", "d\fe\x85f g\n", "last")
    assert sources._split_lines("") == ()
    assert sources._split_lines("x\n") == ("x\n",)
