"""Shared constants for workspace and artifact versions."""

SCHEMA_VERSION = 1
# Version 2 switches the model input from CLI metadata plus source to the raw
# static test source used by the reference FLAST implementation.
FEATURE_SCHEMA_VERSION = 2
DEFAULT_FRAMEWORK = "pytest"
DEFAULT_RUNS = 10
DEFAULT_MIN_OBSERVATIONS = 3

DEFAULT_K = 7
DEFAULT_SIGMA = 0.5
DEFAULT_EPS = 0.3
DEFAULT_METRIC = "cosine"
DEFAULT_WEIGHTS = "distance"
DEFAULT_ALGORITHM = "brute"

OUTCOME_PASSED = "passed"
OUTCOME_FAILED = "failed"
OUTCOME_ERROR = "error"
OUTCOME_SKIPPED = "skipped"
OUTCOME_XFAIL = "xfail"
OUTCOME_XPASS = "xpass"

FAILURE_LIKE_OUTCOMES = {OUTCOME_FAILED, OUTCOME_ERROR, OUTCOME_XPASS}
SKIP_LIKE_OUTCOMES = {OUTCOME_SKIPPED, OUTCOME_XFAIL}
