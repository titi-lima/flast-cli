"""Shared constants for workspace and artifact versions."""

SCHEMA_VERSION = 1
# Version 3 fits vectorization and random projection on the complete known
# corpus before selecting labeled kNN rows, matching FLAST's preprocessing.
FEATURE_SCHEMA_VERSION = 3
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
