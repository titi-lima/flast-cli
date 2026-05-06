"""Human and machine-readable report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .schemas import PredictionReport


def render_report(report: PredictionReport, *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)

    headers = ("score", "label", "test_id")
    rows = [
        (
            f"{item.score:.3f}",
            item.predicted_label,
            item.test_id,
        )
        for item in sorted(report.predictions, key=lambda item: item.score, reverse=True)
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    lines = [
        "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    for row in rows:
        lines.append("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return "\n".join(lines)


def write_optional_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
