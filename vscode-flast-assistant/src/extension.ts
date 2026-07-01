import * as childProcess from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

type RefreshMode = 'manual' | 'onSave';
type SeveritySetting = 'warning' | 'information';
const DEFAULT_LANGUAGE_IDS = ['python', 'javascript', 'typescript', 'javascriptreact', 'typescriptreact', 'java'];

interface FlastDiagnosticRecord {
  test_id: string;
  score: number;
  predicted_label: string;
  class_name?: string | null;
  test_name?: string | null;
  file_path?: string | null;
  line?: number | null;
  message: string;
  explanation?: {
    summary?: string;
    observed_history?: {
      label: string;
      reason: string;
      total_observations: number;
      outcome_counts: Record<string, number>;
    } | null;
    nearest_examples?: Array<{
      test_id: string;
      label: string;
      distance: number;
      score_contribution: number;
      reason: string;
      total_observations: number;
      outcome_counts: Record<string, number>;
    }>;
    static_signals?: Array<{
      kind: string;
      detail: string;
    }>;
  };
}

interface FlastDiagnosticsPayload {
  diagnostics: FlastDiagnosticRecord[];
}

interface FlastConfig {
  executablePath: string;
  executableArgs: string[];
  pythonPaths: string[];
  projectDir: string;
  framework: string;
  frameworkArgs: string[];
  languageIds: string[];
  refreshMode: RefreshMode;
  warningThreshold: number;
  diagnosticSeverity: SeveritySetting;
}

let diagnosticsCollection: vscode.DiagnosticCollection;
let output: vscode.OutputChannel;
let refreshTimer: NodeJS.Timeout | undefined;
let activeRun: childProcess.ChildProcessWithoutNullStreams | undefined;

export function activate(context: vscode.ExtensionContext): void {
  diagnosticsCollection = vscode.languages.createDiagnosticCollection('flast');
  output = vscode.window.createOutputChannel('Flast');

  context.subscriptions.push(diagnosticsCollection, output);
  context.subscriptions.push(
    vscode.commands.registerCommand('flast.initializeWorkspace', initializeWorkspace),
    vscode.commands.registerCommand('flast.observeTestRuns', observeTestRuns),
    vscode.commands.registerCommand('flast.trainModel', () => runSimpleCommand(['train'], 'Train FLAST model')),
    vscode.commands.registerCommand('flast.refreshDiagnostics', () => refreshDiagnostics()),
    vscode.commands.registerCommand('flast.openLatestReport', openLatestReport),
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      const config = await readConfig();
      if (config.languageIds.includes(document.languageId) && config.refreshMode === 'onSave') {
        scheduleRefresh();
      }
    }),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration('flast')) {
        diagnosticsCollection.clear();
      }
    })
  );
}

export function deactivate(): void {
  if (activeRun) {
    activeRun.kill();
  }
}

async function readConfig(): Promise<FlastConfig> {
  const config = vscode.workspace.getConfiguration('flast');
  const frameworkArgs = config.get<string[]>('frameworkArgs');
  const projectDir = resolveProjectDir(config.get<string>('projectDir', ''));
  const executable = await resolveFlastExecutable(config.get<string>('executablePath', 'flast'), projectDir);
  return {
    executablePath: executable.command,
    executableArgs: executable.args,
    pythonPaths: executable.pythonPaths,
    projectDir,
    framework: config.get<string>('framework', 'pytest'),
    frameworkArgs: frameworkArgs ?? config.get<string[]>('pytestArgs', []),
    languageIds: config.get<string[]>('languageIds', DEFAULT_LANGUAGE_IDS),
    refreshMode: config.get<RefreshMode>('refreshMode', 'onSave'),
    warningThreshold: config.get<number>('warningThreshold', 0.5),
    diagnosticSeverity: config.get<SeveritySetting>('diagnosticSeverity', 'warning')
  };
}

async function resolveFlastExecutable(
  configuredPath: string,
  projectDir: string
): Promise<{ command: string; args: string[]; pythonPaths: string[] }> {
  const trimmed = configuredPath.trim();
  if (trimmed && trimmed !== 'flast') {
    return { command: trimmed, args: [], pythonPaths: [] };
  }

  const extensionRoot = vscode.extensions.getExtension('flast.flast-assistant')?.extensionPath;
  const candidates = [
    extensionRoot ? path.join(extensionRoot, 'flast-cli') : undefined,
    path.resolve(__dirname, '..', '..'),
    path.resolve(__dirname, '..', '..', '..')
  ].filter((candidate): candidate is string => Boolean(candidate));

  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, 'flast_cli', 'cli.py'))) {
      const pythonCommand = await resolvePythonCommand(projectDir);
      return {
        command: pythonCommand[0],
        args: [...pythonCommand.slice(1), '-m', 'flast_cli.cli'],
        pythonPaths: [candidate],
      };
    }
  }

  return { command: 'flast', args: [], pythonPaths: [] };
}

async function resolvePythonCommand(projectDir: string): Promise<string[]> {
  const flastConfig = vscode.workspace.getConfiguration('flast');
  const configuredPythonPath = flastConfig.get<string>('pythonPath', '').trim();
  if (configuredPythonPath) {
    return [configuredPythonPath];
  }

  const nearbyVenvPython = findNearbyVenvPython(projectDir);
  if (nearbyVenvPython) {
    return [nearbyVenvPython];
  }

  const resource = vscode.Uri.file(projectDir);
  const pythonExtension = vscode.extensions.getExtension('ms-python.python');
  if (pythonExtension) {
    try {
      const api = pythonExtension.isActive ? pythonExtension.exports : await pythonExtension.activate();
      const executionDetails = api?.settings?.getExecutionDetails?.(resource);
      const execCommand = executionDetails?.execCommand;
      if (Array.isArray(execCommand) && execCommand.length > 0) {
        return execCommand.map(String);
      }
    } catch (error) {
      output.appendLine(`Unable to read VS Code Python interpreter selection: ${(error as Error).message}`);
    }
  }

  const pythonConfig = vscode.workspace.getConfiguration('python', resource);
  const defaultInterpreter = pythonConfig.get<string>('defaultInterpreterPath', '').trim();
  if (defaultInterpreter) {
    return [defaultInterpreter];
  }
  const legacyInterpreter = pythonConfig.get<string>('pythonPath', '').trim();
  if (legacyInterpreter) {
    return [legacyInterpreter];
  }

  return [process.platform === 'win32' ? 'python' : 'python3'];
}

function findNearbyVenvPython(projectDir: string): string | undefined {
  let current = path.resolve(projectDir);

  while (true) {
    for (const venvName of ['.venv', 'venv']) {
      const executable = process.platform === 'win32'
        ? path.join(current, venvName, 'Scripts', 'python.exe')
        : path.join(current, venvName, 'bin', 'python');
      if (fs.existsSync(executable)) {
        return executable;
      }
    }

    const parent = path.dirname(current);
    if (parent === current) {
      return undefined;
    }
    current = parent;
  }
}

function resolveProjectDir(configuredProjectDir: string): string {
  if (configuredProjectDir.trim()) {
    if (path.isAbsolute(configuredProjectDir)) {
      return configuredProjectDir;
    }
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    return path.resolve(workspaceFolder?.uri.fsPath ?? process.cwd(), configuredProjectDir);
  }

  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    throw new Error('Open a workspace before running FLAST commands.');
  }
  return workspaceFolder.uri.fsPath;
}

function scheduleRefresh(): void {
  if (refreshTimer) {
    clearTimeout(refreshTimer);
  }
  refreshTimer = setTimeout(() => {
    void refreshDiagnostics();
  }, 1000);
}

async function initializeWorkspace(): Promise<void> {
  const config = await readConfig();
  await runSimpleCommand(['init', config.framework], 'Initialize FLAST workspace');
}

async function observeTestRuns(): Promise<void> {
  const config = await readConfig();
  const runs = await vscode.window.showInputBox({
    title: 'Flast: Observe Test Runs',
    prompt: `Number of repeated ${config.framework} runs to observe`,
    value: '10',
    validateInput: (value) => {
      const parsed = Number(value);
      return Number.isInteger(parsed) && parsed > 0 ? undefined : 'Enter a positive integer.';
    }
  });
  if (!runs) {
    return;
  }
  await runSimpleCommand(['observe', config.framework, '--runs', runs], 'Observe FLAST test runs', true);
}

async function runSimpleCommand(baseArgs: string[], title: string, includePytestArgs = false): Promise<void> {
  const config = await readConfig();
  const args = [
    ...baseArgs,
    '--project-dir',
    config.projectDir,
    ...(includePytestArgs ? frameworkArgs(config.frameworkArgs) : [])
  ];
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title, cancellable: true },
    async (_progress, token) => {
      await runFlast(config, args, token);
    }
  );
  await refreshDiagnosticsAfterCommand();
  vscode.window.showInformationMessage(`${title} completed.`);
}

async function refreshDiagnosticsAfterCommand(): Promise<void> {
  try {
    await refreshDiagnostics({ showErrors: false });
  } catch {
    // The output channel already has the diagnostic refresh failure details.
  }
}

async function refreshDiagnostics(options: { showErrors?: boolean } = {}): Promise<void> {
  const config = await readConfig();
  const args = [
    'auto',
    config.framework,
    '--format',
    'json',
    '--project-dir',
    config.projectDir,
    ...frameworkArgs(config.frameworkArgs)
  ];

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Window, title: 'Refreshing FLAST diagnostics', cancellable: true },
    async (_progress, token) => {
      const result = await runFlast(config, args, token, { showErrors: options.showErrors ?? true });
      const payload = parseDiagnostics(result.stdout);
      applyDiagnostics(payload, config);
    }
  );
}

function frameworkArgs(args: string[]): string[] {
  return args.flatMap((arg) => ['--framework-arg', arg]);
}

function runFlast(
  config: FlastConfig,
  args: string[],
  token: vscode.CancellationToken,
  options: { showErrors?: boolean } = {}
): Promise<{ stdout: string; stderr: string }> {
  if (activeRun) {
    activeRun.kill();
    activeRun = undefined;
  }

  const commandArgs = [...config.executableArgs, ...args];
  output.appendLine(`$ ${config.executablePath} ${commandArgs.join(' ')}`);

  return new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
    const env = { ...process.env };
    let pythonPath = '';
    if (config.pythonPaths.length > 0) {
      pythonPath = config.pythonPaths.join(path.delimiter);
      env.PYTHONPATH = env.PYTHONPATH
        ? `${pythonPath}${path.delimiter}${env.PYTHONPATH}`
        : pythonPath;
    }
    output.appendLine(`FLAST project dir: ${config.projectDir}`);
    if (pythonPath) {
      output.appendLine(`FLAST bundled PYTHONPATH entries: ${pythonPath}`);
    }

    const child = childProcess.spawn(config.executablePath, commandArgs, {
      cwd: config.projectDir,
      env,
      shell: process.platform === 'win32'
    });
    activeRun = child;

    let stdout = '';
    let stderr = '';

    const cancellation = token.onCancellationRequested(() => {
      child.kill();
      reject(new Error('FLAST command cancelled.'));
    });

    child.stdout.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      stdout += text;
      output.append(text);
    });
    child.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      stderr += text;
      output.append(text);
    });
    child.on('error', (error) => {
      cancellation.dispose();
      activeRun = undefined;
      reject(error);
    });
    child.on('close', (code) => {
      cancellation.dispose();
      activeRun = undefined;
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      const detail = stderr.trim() || stdout.trim() || `FLAST exited with code ${code}.`;
      reject(new Error(detail));
    });
  }).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    output.show(true);
    if (options.showErrors ?? true) {
      vscode.window.showErrorMessage(`FLAST failed: ${message}`);
    } else {
      output.appendLine(`FLAST diagnostics refresh skipped: ${message}`);
    }
    throw error;
  });
}

function parseDiagnostics(stdout: string): FlastDiagnosticsPayload {
  try {
    const payload = JSON.parse(stdout) as FlastDiagnosticsPayload;
    if (!Array.isArray(payload.diagnostics)) {
      throw new Error('Missing diagnostics array.');
    }
    return payload;
  } catch (error) {
    throw new Error(`FLAST returned invalid diagnostics JSON: ${(error as Error).message}`);
  }
}

function applyDiagnostics(payload: FlastDiagnosticsPayload, config: FlastConfig): void {
  const byFile = new Map<string, vscode.Diagnostic[]>();

  for (const record of payload.diagnostics) {
    if (record.predicted_label !== 'flaky' || !record.file_path || record.score < config.warningThreshold) {
      continue;
    }

    const line = Math.max((record.line ?? 1) - 1, 0);
    const range = new vscode.Range(line, 0, line, Number.MAX_SAFE_INTEGER);
    const diagnostic = new vscode.Diagnostic(
      range,
      record.message,
      config.diagnosticSeverity === 'warning'
        ? vscode.DiagnosticSeverity.Warning
        : vscode.DiagnosticSeverity.Information
    );
    diagnostic.source = 'FLAST';
    diagnostic.code = record.test_id;

    const fileDiagnostics = byFile.get(record.file_path) ?? [];
    fileDiagnostics.push(diagnostic);
    byFile.set(record.file_path, fileDiagnostics);
  }

  diagnosticsCollection.clear();
  for (const [filePath, diagnostics] of byFile) {
    diagnosticsCollection.set(vscode.Uri.file(filePath), diagnostics);
  }
}

async function openLatestReport(): Promise<void> {
  const config = await readConfig();
  const reportsDir = path.join(config.projectDir, '.flast', 'reports');
  if (!fs.existsSync(reportsDir)) {
    vscode.window.showWarningMessage('No FLAST reports directory exists yet.');
    return;
  }

  const reports = fs.readdirSync(reportsDir)
    .filter((name) => name.endsWith('.json'))
    .map((name) => path.join(reportsDir, name))
    .sort((left, right) => fs.statSync(right).mtimeMs - fs.statSync(left).mtimeMs);

  if (reports.length === 0) {
    vscode.window.showWarningMessage('No FLAST reports found.');
    return;
  }

  const document = await vscode.workspace.openTextDocument(vscode.Uri.file(reports[0]));
  await vscode.window.showTextDocument(document);
}
