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
    vscode.workspace.onDidSaveTextDocument((document) => {
      const config = readConfig();
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

function readConfig(): FlastConfig {
  const config = vscode.workspace.getConfiguration('flast');
  const frameworkArgs = config.get<string[]>('frameworkArgs');
  return {
    executablePath: config.get<string>('executablePath', 'flast'),
    projectDir: resolveProjectDir(config.get<string>('projectDir', '')),
    framework: config.get<string>('framework', 'pytest'),
    frameworkArgs: frameworkArgs ?? config.get<string[]>('pytestArgs', []),
    languageIds: config.get<string[]>('languageIds', DEFAULT_LANGUAGE_IDS),
    refreshMode: config.get<RefreshMode>('refreshMode', 'manual'),
    warningThreshold: config.get<number>('warningThreshold', 0.5),
    diagnosticSeverity: config.get<SeveritySetting>('diagnosticSeverity', 'warning')
  };
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
  const config = readConfig();
  await runSimpleCommand(['init', config.framework], 'Initialize FLAST workspace');
}

async function observeTestRuns(): Promise<void> {
  const config = readConfig();
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
  const config = readConfig();
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
  vscode.window.showInformationMessage(`${title} completed.`);
}

async function refreshDiagnostics(): Promise<void> {
  const config = readConfig();
  const args = [
    'diagnostics',
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
      const result = await runFlast(config, args, token);
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
  token: vscode.CancellationToken
): Promise<{ stdout: string; stderr: string }> {
  if (activeRun) {
    activeRun.kill();
    activeRun = undefined;
  }

  output.appendLine(`$ ${config.executablePath} ${args.join(' ')}`);

  return new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
    const child = childProcess.spawn(config.executablePath, args, {
      cwd: config.projectDir,
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
    vscode.window.showErrorMessage(`FLAST failed: ${message}`);
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
  const config = readConfig();
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
