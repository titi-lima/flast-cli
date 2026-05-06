"""Feature extraction for FLAST prediction inputs."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .schemas import FeatureInput


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
