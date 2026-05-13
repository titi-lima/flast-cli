# FLAST framework provider protocol

FLAST is split into a language-neutral core and framework providers. The core
owns corpus aggregation, labels, model training, prediction, reports, and VS Code
diagnostic output. A provider owns the framework-specific work needed to produce
the same normalized records for any test runner.

## Provider responsibilities

A provider must be able to:

- collect test IDs without executing tests
- observe one framework run and write a run manifest
- normalize framework outcomes to FLAST outcomes
- resolve a test ID to source text for model features
- resolve a prediction to a source line for editor diagnostics

The first-party `pytest` provider is currently implemented as an in-process
Python adapter plus a thin pytest plugin. Future providers can use the same
adapter interface or wrap an external executable that speaks JSON over files or
stdout.

## Run manifest

Providers write one manifest per collection or observation run:

```json
{
  "schema_version": 1,
  "framework": "pytest",
  "run_id": "uuid",
  "project_root": "/absolute/project/root",
  "tool": {
    "name": "pytest",
    "version": "8.0.0"
  },
  "collected": [
    {"test_id": "tests/test_example.py::test_name"}
  ],
  "outcomes": [
    {"test_id": "tests/test_example.py::test_name", "outcome": "passed"}
  ]
}
```

`tool` is provider-defined metadata. It should at least include `name` and
`version` when the runner exposes them.

## Outcomes

Providers normalize runner-specific statuses to these core outcomes:

- `passed`
- `failed`
- `error`
- `skipped`
- `xfail`
- `xpass`

The core treats `failed`, `error`, and `xpass` as failure-like outcomes. It
treats `skipped` and `xfail` as skip-like outcomes.

## Python adapter interface

In-process providers implement `FrameworkAdapter`:

```python
class FrameworkAdapter:
    framework: str

    def build_observe_invocation(self, *, project_root, manifest_path, framework_args): ...
    def build_collect_invocation(self, *, project_root, manifest_path, framework_args): ...
    def run(self, *, project_root, command): ...
    def load_manifest(self, path): ...
    def resolve_feature_input(self, test_id, project_root): ...
    def resolve_prediction_line(self, prediction, project_root): ...
```

Provider-specific command-line flags should be forwarded through
`--framework-arg`. Existing pytest users can still use `--pytest-arg`; it is an
alias for compatibility.

## VS Code contract

The VS Code extension should remain framework-neutral. It reads:

- `flast.framework`
- `flast.frameworkArgs`
- `flast.languageIds`

Then it shells out to:

```bash
flast init <framework>
flast observe <framework> --framework-arg ...
flast diagnostics <framework> --format json --framework-arg ...
```

The extension consumes only FLAST diagnostics JSON and should not contain
framework-specific parsing logic.

## Vitest provider

The first Vitest provider uses a Python adapter plus a small runner module. It
collects test IDs with `npx vitest list`, observes outcomes with Vitest's JSON
reporter, and emits IDs in this shape:

```text
tests/sample.test.js::suite name::test name
```

Source resolution currently uses lightweight JavaScript/TypeScript test-call
matching for `test(...)` and `it(...)`. A future provider package can replace
that with a TypeScript parser without changing the core FLAST or VS Code
contracts.

## JUnit provider

The first JUnit provider targets JUnit 5 projects that expose tests under
`src/test/java`. It discovers methods annotated with `@Test`, compiles Ant
projects with `ant compile` when `build.xml` exists, and observes outcomes with
the JUnit Platform Console XML reports. Test IDs use this shape:

```text
src/test/java/com/example/AppTest.java::com.example.AppTest::methodName
```

The runner expects the JUnit Platform Console standalone jar at
`lib/junit-platform-console-standalone.jar`, a matching
`lib/junit-platform-console-standalone-*.jar`, or the
`JUNIT_PLATFORM_CONSOLE_JAR` environment variable.
