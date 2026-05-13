from __future__ import annotations

from pathlib import Path

from flast_cli.adapters import VitestAdapter, supported_frameworks
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
