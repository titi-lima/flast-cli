from pathlib import Path


STATE = Path(__file__).with_name("flaky_state.txt")


def _run_count() -> int:
    if not STATE.exists():
        STATE.write_text("0", encoding="utf-8")
    value = int(STATE.read_text(encoding="utf-8"))
    STATE.write_text(str(value + 1), encoding="utf-8")
    return value


def test_flaky_toggle() -> None:
    assert _run_count() % 2 == 0


def test_stable_pass() -> None:
    assert True
