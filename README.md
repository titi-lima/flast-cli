# flast-cli

`flast-cli` is a headless wrapper around the FLAST flaky-test detection approach.

It provides:

- a `flast` command-line interface
- a thin Pytest plugin that emits manifests with collected test IDs and outcomes
- a local `.flast/` workspace for observations, labels, models, and reports

## Quick start

```bash
python -m pip install -e .
flast init pytest
flast observe pytest --runs 10
flast train
flast predict pytest
```
