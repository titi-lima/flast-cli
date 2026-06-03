from __future__ import annotations

from pathlib import Path

from flast_cli.adapters import JUnitAdapter, supported_frameworks
from flast_cli.features import detect_static_signals
from flast_cli.integrations.junit.runner import (
    discover_junit_tests,
    extract_test_methods,
    parse_junit_xml_reports,
)
from flast_cli.schemas import PredictionRecord


JAVA_SOURCE = """
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class AppTest {
    @Test
    void stablePass() {
        assertEquals(2, 1 + 1);
    }

    @Test
    void flakyToggle() {
        assertEquals(0, 0);
    }
}
""".strip() + "\n"


def test_junit_is_registered() -> None:
    assert "junit" in set(supported_frameworks())


def test_extract_junit_test_methods() -> None:
    assert extract_test_methods(JAVA_SOURCE) == ["stablePass", "flakyToggle"]


def test_discover_junit_tests(tmp_path: Path) -> None:
    test_file = tmp_path / "src" / "test" / "java" / "com" / "example" / "AppTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(JAVA_SOURCE, encoding="utf-8")

    tests = discover_junit_tests(tmp_path)

    assert [item.test_id for item in tests] == [
        "src/test/java/com/example/AppTest.java::com.example.AppTest::stablePass",
        "src/test/java/com/example/AppTest.java::com.example.AppTest::flakyToggle",
    ]


def test_parse_junit_xml_reports(tmp_path: Path) -> None:
    test_file = tmp_path / "src" / "test" / "java" / "com" / "example" / "AppTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(JAVA_SOURCE, encoding="utf-8")
    tests = discover_junit_tests(tmp_path)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "TEST-com.example.AppTest.xml").write_text(
        """
<testsuite>
  <testcase classname="com.example.AppTest" name="stablePass()"/>
  <testcase classname="com.example.AppTest" name="flakyToggle()">
    <failure message="expected failure"/>
  </testcase>
</testsuite>
""".strip()
        + "\n",
        encoding="utf-8",
    )

    outcomes = parse_junit_xml_reports(reports_dir, tests)

    assert {item.test_id: item.outcome for item in outcomes} == {
        "src/test/java/com/example/AppTest.java::com.example.AppTest::stablePass": "passed",
        "src/test/java/com/example/AppTest.java::com.example.AppTest::flakyToggle": "failed",
    }


def test_junit_feature_and_line_resolution(tmp_path: Path) -> None:
    test_file = tmp_path / "src" / "test" / "java" / "com" / "example" / "AppTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(JAVA_SOURCE, encoding="utf-8")

    adapter = JUnitAdapter()
    test_id = "src/test/java/com/example/AppTest.java::com.example.AppTest::stablePass"
    feature = adapter.resolve_feature_input(test_id, tmp_path)

    assert feature.class_name == "com.example.AppTest"
    assert feature.test_name == "stablePass"
    assert "assertEquals(2, 1 + 1)" in feature.source_text
    assert adapter.resolve_prediction_line(
        PredictionRecord(
            test_id=test_id,
            score=1.0,
            predicted_label="flaky",
            class_name="com.example.AppTest",
            test_name="stablePass",
            file_path=str(test_file),
        ),
        tmp_path,
    ) == 7


def test_junit_static_signal_detection_uses_java_vocabulary(tmp_path: Path) -> None:
    test_file = tmp_path / "src" / "test" / "java" / "com" / "example" / "FlakyTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        """
package com.example;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.time.Instant;
import java.util.UUID;

class FlakyTest {
    static String shared;

    @Test
    void readsClockAndFile() throws Exception {
        shared = UUID.randomUUID().toString() + Instant.now();
        Files.writeString(java.nio.file.Path.of("state.txt"), shared);
        Thread.sleep(10);
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    feature = JUnitAdapter().resolve_feature_input(
        "src/test/java/com/example/FlakyTest.java::com.example.FlakyTest::readsClockAndFile",
        tmp_path,
    )
    kinds = {item["kind"] for item in detect_static_signals(feature)}

    assert {"randomness", "time", "timing", "filesystem state"}.issubset(kinds)
