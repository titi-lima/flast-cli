# Flast Assistant for VS Code

Flast Assistant surfaces FLAST flaky-test predictions as VS Code diagnostics.
It defaults to the `pytest` provider, and the extension host is framework-neutral
so additional providers can plug into the same command flow.

## Usage

1. Run `Flast: Initialize Workspace`.
2. Run your tests normally.
3. Run `Flast: Refresh Diagnostics`.

Initialization enables the selected framework integration to capture regular
test runs as observations. Pytest uses its plugin entry point, Vitest uses a
local reporter installed under `.flast/tools/vitest/`, and JUnit uses a JUnit
Platform listener registered through ServiceLoader. Refresh rebuilds the corpus,
retrains when enough labels exist, and updates diagnostics in one command.

The extension first looks for a bundled `flast-cli/` directory, then for the
local repository layout used during extension development, and finally falls
back to the `flast` command on `PATH`. Set `flast.executablePath` only when you
need to override that resolver.

Diagnostics are advisory risk hints. Refresh is manual by default.

## Settings

- `flast.executablePath`: optional command or path used to override FLAST resolution.
- `flast.projectDir`: project root. Defaults to the first workspace folder.
- `flast.framework`: framework provider used by FLAST commands.
- `flast.frameworkArgs`: arguments forwarded to the selected framework provider.
- `flast.languageIds`: VS Code language IDs that can trigger on-save refreshes.
- `flast.refreshMode`: `manual` or `onSave`.
- `flast.warningThreshold`: minimum score for warning diagnostics.
- `flast.diagnosticSeverity`: `warning` or `information`.

For Vitest projects, set `flast.framework` to `vitest` and include JavaScript or
TypeScript language IDs in `flast.languageIds`, for example `javascript`,
`typescript`, `javascriptreact`, or `typescriptreact`.

For JUnit projects, set `flast.framework` to `junit`; the default language IDs
already include `java`.
