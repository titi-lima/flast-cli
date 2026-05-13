"""Framework adapter registry."""

from __future__ import annotations

from typing import Dict, Iterable

from .base import FrameworkAdapter
from .junit_adapter import JUnitAdapter
from .pytest_adapter import PytestAdapter
from .vitest_adapter import VitestAdapter


_ADAPTERS: Dict[str, FrameworkAdapter] = {
    JUnitAdapter.framework: JUnitAdapter(),
    PytestAdapter.framework: PytestAdapter(),
    VitestAdapter.framework: VitestAdapter(),
}


def get_adapter(framework: str) -> FrameworkAdapter:
    try:
        return _ADAPTERS[framework]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unsupported framework: {framework}. Supported frameworks: {supported}"
        ) from exc


def supported_frameworks() -> Iterable[str]:
    return sorted(_ADAPTERS)
