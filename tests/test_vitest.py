from __future__ import annotations

from pathlib import Path

from flast_cli.adapters import VitestAdapter, supported_frameworks
from flast_cli.features import detect_static_signals
from flast_cli.integrations.vitest.runner import (
    manifest_records_from_json,
    parse_vitest_list_output,
)


def test_vitest_is_registered() -> None:
    assert "vitest" in set(supported_frameworks())


def test_parse_vitest_list_output_builds_stable_ids(tmp_path: Path) -> None:
    output = """
tests/sample.test.ts > math helpers > adds numbers
tests/sample.test.ts > math helpers > subtracts numbers
"""

    assert parse_vitest_list_output(output, tmp_path) == [
        "tests/sample.test.ts::math helpers::adds numbers",
        "tests/sample.test.ts::math helpers::subtracts numbers",
    ]


def test_vitest_json_report_converts_to_manifest_records(tmp_path: Path) -> None:
    payload = {
        "testResults": [
            {
                "name": str(tmp_path / "tests" / "sample.test.ts"),
                "assertionResults": [
                    {
                        "ancestorTitles": ["math helpers"],
                        "title": "adds numbers",
                        "status": "passed",
                    },
                    {
                        "ancestorTitles": ["math helpers"],
                        "title": "flaky case",
                        "status": "failed",
                    },
                ],
            }
        ]
    }

    collected, outcomes = manifest_records_from_json(payload, tmp_path)

    assert [item.test_id for item in collected] == [
        "tests/sample.test.ts::math helpers::adds numbers",
        "tests/sample.test.ts::math helpers::flaky case",
    ]
    assert {item.test_id: item.outcome for item in outcomes} == {
        "tests/sample.test.ts::math helpers::adds numbers": "passed",
        "tests/sample.test.ts::math helpers::flaky case": "failed",
    }


def test_vitest_feature_and_line_resolution(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "sample.test.ts"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """
import { describe, expect, test } from 'vitest';

describe('math helpers', () => {
  test('adds numbers', () => {
    expect(1 + 1).toBe(2);
  });
});
""".strip()
        + "\n",
        encoding="utf-8",
    )

    adapter = VitestAdapter()
    feature = adapter.resolve_feature_input(
        "tests/sample.test.ts::math helpers::adds numbers",
        tmp_path,
    )

    assert feature.class_name == "math helpers"
    assert feature.test_name == "adds numbers"
    assert "expect(1 + 1)" in feature.source_text
    assert adapter.resolve_prediction_line(
        prediction=_prediction("tests/sample.test.ts::math helpers::adds numbers", test_file),
        project_root=tmp_path,
    ) == 4


def test_vitest_static_signal_detection_uses_js_ts_vocabulary(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "flaky.test.ts"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """
import { test, vi } from 'vitest';
import { writeFileSync } from 'node:fs';

test('uses async shared state', async () => {
  vi.mock('../clock');
  process.env.TOKEN = String(Date.now() + Math.random());
  writeFileSync('state.txt', process.env.TOKEN);
  await new Promise((resolve) => setTimeout(resolve, 10));
});
""".strip()
        + "\n",
        encoding="utf-8",
    )

    feature = VitestAdapter().resolve_feature_input(
        "tests/flaky.test.ts::uses async shared state",
        tmp_path,
    )
    kinds = {item["kind"] for item in detect_static_signals(feature)}

    assert {"randomness", "time", "timing", "filesystem state", "environment mutation", "mock state"}.issubset(kinds)


def _prediction(test_id: str, test_file: Path):
    from flast_cli.schemas import PredictionRecord

    return PredictionRecord(
        test_id=test_id,
        score=1.0,
        predicted_label="flaky",
        class_name="math helpers",
        test_name="adds numbers",
        file_path=str(test_file),
    )
