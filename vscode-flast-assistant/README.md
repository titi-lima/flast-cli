# Flast Assistant for VS Code

Flast Assistant surfaces FLAST flaky-test predictions as VS Code diagnostics for
pytest projects.

## Usage

1. Install `flast-cli` in the Python environment used by your project.
2. Run `Flast: Initialize Workspace`.
3. Run `Flast: Observe Test Runs`.
4. Run `Flast: Train Model`.
5. Run `Flast: Refresh Diagnostics`.

Diagnostics are advisory risk hints. FLAST runs pytest collection during
prediction, so refresh is manual by default.

## Settings

- `flast.executablePath`: command or path used to invoke Flast.
- `flast.projectDir`: project root. Defaults to the first workspace folder.
- `flast.pytestArgs`: arguments forwarded to pytest.
- `flast.refreshMode`: `manual` or `onSave`.
- `flast.warningThreshold`: minimum score for warning diagnostics.
- `flast.diagnosticSeverity`: `warning` or `information`.
