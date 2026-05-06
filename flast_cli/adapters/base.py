"""Adapter interfaces for framework integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence

from ..schemas import FeatureInput, RunManifest


class FrameworkAdapter(ABC):
    framework: str

    @abstractmethod
    def build_observe_invocation(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        pytest_args: Sequence[str],
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def build_collect_invocation(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        pytest_args: Sequence[str],
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def load_manifest(self, path: Path) -> RunManifest:
        raise NotImplementedError

    @abstractmethod
    def resolve_feature_input(self, test_id: str, project_root: Path) -> FeatureInput:
        raise NotImplementedError
