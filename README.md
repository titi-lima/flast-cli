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
pytest
pytest
pytest
flast auto pytest
```

After `flast init pytest`, regular pytest runs automatically write observations
into `.flast/runs/` through the pytest plugin. `flast auto pytest` rebuilds the
corpus, retrains when enough labels exist, and emits diagnostics in one step.
The lower-level `observe`, `train`, `predict`, and `diagnostics` commands remain
available for scripts and non-pytest providers.

Vitest projects use the same flow with `vitest`:

```bash
flast init vitest
npm test
npm test
npm test
flast auto vitest --framework-arg tests/sample.test.js
```

`flast init vitest` installs `.flast/tools/vitest/flast-reporter.cjs` and patches
a standard `package.json` `test` script so normal Vitest runs add observations.

Ant/JUnit 5 projects can use `junit` when the JUnit Platform Console standalone
jar is available under `lib/`:

```bash
flast init junit
./mvnw test
./mvnw test
./mvnw test
flast auto junit
```

`flast init junit` installs a JUnit Platform `TestExecutionListener` under
`src/test/java/flast/` and registers it with ServiceLoader under
`src/test/resources/META-INF/services/`.

Provider authors should start with [docs/plugin-protocol.md](docs/plugin-protocol.md).
