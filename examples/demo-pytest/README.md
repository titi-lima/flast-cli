# Demo Pytest Project

This is a minimal Pytest project for demonstrating the `flast` workflow.

It contains:

- `test_flaky_toggle`, which alternates between pass and fail across runs
- `test_stable_pass`, which always passes

Typical demo flow from this directory:

```bash
flast init pytest
flast observe pytest --runs 3 --pytest-arg tests
flast train
flast predict pytest --pytest-arg tests
flast report --latest --format json
```
