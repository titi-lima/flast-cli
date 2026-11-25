"""Per-process cache of test-module sources.

Resolving a test ID to its source used to read and re-parse the test's whole
module once per test, and again for every lookup (training, prediction, and
line resolution).  For suites with large test modules that made source
resolution dominate ``flast auto``.  This module reads each file once and, for
Python, parses it once and indexes its test functions, so every later lookup
is a dictionary access.

Entries are keyed by resolved path plus ``st_mtime_ns`` and ``st_size``; a
file that changes on disk is re-read on the next lookup.
"""

from __future__ import annotations

import ast
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

_MAX_ENTRIES = 512

FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


@dataclass(frozen=True)
class PythonModule:
    source: str
    # None when the module does not parse.
    tree: Optional[ast.Module]
    # Lines with their "\n" terminators, as ast.get_source_segment splits them.
    lines: Tuple[str, ...]
    # (class name or None, function name) -> first matching definition, using
    # the same first-match lookup as before: top-level functions, and functions
    # defined directly in the first top-level class with a given name.
    functions: Dict[Tuple[Optional[str], str], FunctionNode]

    def find_function(self, class_name: Optional[str], name: str) -> Optional[FunctionNode]:
        return self.functions.get((class_name, name))

    def node_source(self, node: ast.AST) -> str:
        """Equivalent to ``ast.get_source_segment`` with the ``_node_source`` fallback."""
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        if None not in (start, end, col, end_col):
            segment = self._segment(start - 1, end - 1, col, end_col)
            if segment:
                return segment
        if start is None or end is None:
            return self.source
        return "\n".join(self.source.splitlines()[start - 1 : end])

    def _segment(self, first: int, last: int, col: int, end_col: int) -> Optional[str]:
        # Column offsets are UTF-8 byte offsets, exactly as in ast.get_source_segment.
        if last >= len(self.lines):
            return None
        if first == last:
            return self.lines[first].encode()[col:end_col].decode()
        head = self.lines[first].encode()[col:].decode()
        tail = self.lines[last].encode()[:end_col].decode()
        return "".join([head, *self.lines[first + 1 : last], tail])


_text_cache: "OrderedDict[Tuple[str, int, int], str]" = OrderedDict()
_python_cache: "OrderedDict[Tuple[str, int, int], PythonModule]" = OrderedDict()


def _key(path: Path) -> Tuple[str, int, int]:
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    return resolved, stat.st_mtime_ns, stat.st_size


def _remember(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _MAX_ENTRIES:
        cache.popitem(last=False)
    return value


def read_source(path: Path) -> str:
    """Return ``path.read_text(encoding="utf-8")``, read at most once per version."""
    key = _key(path)
    cached = _text_cache.get(key)
    if cached is not None:
        _text_cache.move_to_end(key)
        return cached
    return _remember(_text_cache, key, Path(key[0]).read_text(encoding="utf-8"))


def python_module(path: Path) -> PythonModule:
    """Return the cached, parsed and indexed Python module at ``path``.

    Read errors (missing file, undecodable bytes) propagate; a module that
    does not parse yields ``tree=None`` and an empty function index.
    """
    key = _key(path)
    cached = _python_cache.get(key)
    if cached is not None:
        _python_cache.move_to_end(key)
        return cached
    source = read_source(path)
    try:
        tree: Optional[ast.Module] = ast.parse(source)
    except SyntaxError:
        tree = None
    module = PythonModule(
        source=source,
        tree=tree,
        lines=_split_lines(source),
        functions=_index_functions(tree) if tree is not None else {},
    )
    return _remember(_python_cache, key, module)


def clear_cache() -> None:
    _text_cache.clear()
    _python_cache.clear()


_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


def _split_lines(source: str) -> Tuple[str, ...]:
    # Same splitting as ast._splitlines_no_ff: only \r\n, \n and \r end a line
    # (str.splitlines would also split on form feeds and Unicode separators).
    return tuple(match.group(0) for match in _LINE.finditer(source))


def _index_functions(tree: ast.Module) -> Dict[Tuple[Optional[str], str], FunctionNode]:
    index: Dict[Tuple[Optional[str], str], FunctionNode] = {}
    seen_classes = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            index.setdefault((None, node.name), node)
        elif isinstance(node, ast.ClassDef) and node.name not in seen_classes:
            seen_classes.add(node.name)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    index.setdefault((node.name, child.name), child)
    return index
