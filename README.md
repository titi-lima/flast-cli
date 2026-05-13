# flast-cli

`flast-cli` is a headless wrapper around the FLAST flaky-test detection approach.

It provides:

- a `flast` command-line interface
- a framework-provider boundary for collecting tests, observing outcomes, and resolving source locations
- thin Pytest, Vitest, and JUnit providers that emit manifests with collected test IDs and outcomes
- a local `.flast/` workspace for observations, labels, models, and reports

## Quick start

```bash
python -m pip install -e .
flast init pytest
flast observe pytest --runs 10
flast train
flast predict pytest
```

Vitest projects use the same flow with `vitest`:

```bash
flast init vitest
flast observe vitest --runs 10 --framework-arg tests/sample.test.js
flast train
flast predict vitest --framework-arg tests/sample.test.js
```

Ant/JUnit 5 projects can use `junit` when the JUnit Platform Console standalone
jar is available under `lib/`:

```bash
flast init junit
flast observe junit --runs 10
flast train
flast predict junit
```

Provider authors should start with [docs/plugin-protocol.md](docs/plugin-protocol.md).
