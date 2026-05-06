"""Observation aggregation and label derivation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .constants import FAILURE_LIKE_OUTCOMES, SCHEMA_VERSION, SKIP_LIKE_OUTCOMES
from .schemas import LabelRecord, ObservationRecord, RunManifest


@dataclass
class _AccumulatedObservation:
    framework: str
    test_id: str
    project_root: str
    run_ids: set
    counts: Counter


def aggregate_manifests(manifests: Iterable[RunManifest]) -> List[ObservationRecord]:
    buckets: Dict[Tuple[str, str], _AccumulatedObservation] = {}
    for manifest in manifests:
        outcome_by_test = {item.test_id: item.outcome for item in manifest.outcomes}
        for test_id, outcome in outcome_by_test.items():
            key = (manifest.framework, test_id)
            entry = buckets.get(key)
            if entry is None:
                entry = _AccumulatedObservation(
                    framework=manifest.framework,
                    test_id=test_id,
                    project_root=manifest.project_root,
                    run_ids=set(),
                    counts=Counter(),
                )
                buckets[key] = entry
            entry.project_root = manifest.project_root
            entry.run_ids.add(manifest.run_id)
            entry.counts[outcome] += 1

    observations: List[ObservationRecord] = []
    for key in sorted(buckets):
        entry = buckets[key]
        observations.append(
            ObservationRecord(
                schema_version=SCHEMA_VERSION,
                framework=entry.framework,
                test_id=entry.test_id,
                project_root=entry.project_root,
                total_observations=sum(entry.counts.values()),
                outcome_counts=dict(sorted(entry.counts.items())),
                seen_outcomes=sorted(entry.counts),
                run_ids=sorted(entry.run_ids),
            )
        )
    return observations


def derive_labels(
    observations: Iterable[ObservationRecord], *, min_observations: int
) -> List[LabelRecord]:
    labels: List[LabelRecord] = []
    for observation in observations:
        counts = observation.outcome_counts
        seen = set(counts)
        has_pass = "passed" in seen
        has_failure_like = bool(seen & FAILURE_LIKE_OUTCOMES)
        only_skip_like = bool(seen) and seen.issubset(SKIP_LIKE_OUTCOMES)
        if has_pass and has_failure_like:
            label = "flaky"
            reason = "observed both passed and failure-like outcomes"
        elif only_skip_like:
            label = "unlabeled"
            reason = "only skip-like outcomes observed"
        elif observation.total_observations < min_observations:
            label = "unlabeled"
            reason = "insufficient observations"
        else:
            label = "non_flaky"
            reason = "stable across observed runs"
        labels.append(
            LabelRecord(
                schema_version=SCHEMA_VERSION,
                framework=observation.framework,
                test_id=observation.test_id,
                project_root=observation.project_root,
                label=label,
                reason=reason,
                total_observations=observation.total_observations,
                outcome_counts=dict(counts),
            )
        )
    return labels
