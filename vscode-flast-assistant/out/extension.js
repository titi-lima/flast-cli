"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const childProcess = __importStar(require("child_process"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const vscode = __importStar(require("vscode"));
const DEFAULT_LANGUAGE_IDS = ['python', 'javascript', 'typescript', 'javascriptreact', 'typescriptreact', 'java'];
let diagnosticsCollection;
let output;
let refreshTimer;
let activeRun;
function activate(context) {
    diagnosticsCollection = vscode.languages.createDiagnosticCollection('flast');
    output = vscode.window.createOutputChannel('Flast');
    context.subscriptions.push(diagnosticsCollection, output);
    context.subscriptions.push(vscode.commands.registerCommand('flast.initializeWorkspace', initializeWorkspace), vscode.commands.registerCommand('flast.observeTestRuns', observeTestRuns), vscode.commands.registerCommand('flast.trainModel', () => runSimpleCommand(['train'], 'Train FLAST model')), vscode.commands.registerCommand('flast.refreshDiagnostics', () => refreshDiagnostics()), vscode.commands.registerCommand('flast.openLatestReport', openLatestReport), vscode.workspace.onDidSaveTextDocument((document) => {
        const config = readConfig();
        if (config.languageIds.includes(document.languageId) && config.refreshMode === 'onSave') {
            scheduleRefresh();
        }
    }), vscode.workspace.onDidChangeConfiguration((event) => {
        if (event.affectsConfiguration('flast')) {
            diagnosticsCollection.clear();
        }
    }));
}
function deactivate() {
    if (activeRun) {
        activeRun.kill();
    }
}
function readConfig() {
    const config = vscode.workspace.getConfiguration('flast');
    const frameworkArgs = config.get('frameworkArgs');
    const executable = resolveFlastExecutable(config.get('executablePath', 'flast'));
    return {
        executablePath: executable.command,
        executableArgs: executable.args,
        pythonPath: executable.pythonPath,
        projectDir: resolveProjectDir(config.get('projectDir', '')),
        framework: config.get('framework', 'pytest'),
        frameworkArgs: frameworkArgs ?? config.get('pytestArgs', []),
        languageIds: config.get('languageIds', DEFAULT_LANGUAGE_IDS),
        refreshMode: config.get('refreshMode', 'manual'),
        warningThreshold: config.get('warningThreshold', 0.5),
        diagnosticSeverity: config.get('diagnosticSeverity', 'warning')
    };
}
function resolveFlastExecutable(configuredPath) {
    const trimmed = configuredPath.trim();
    if (trimmed && trimmed !== 'flast') {
        return { command: trimmed, args: [] };
    }
    const extensionRoot = vscode.extensions.getExtension('flast.flast-assistant')?.extensionPath;
    const candidates = [
        extensionRoot ? path.join(extensionRoot, 'flast-cli') : undefined,
        path.resolve(__dirname, '..', '..'),
        path.resolve(__dirname, '..', '..', '..')
    ].filter((candidate) => Boolean(candidate));
    for (const candidate of candidates) {
        if (fs.existsSync(path.join(candidate, 'flast_cli', 'cli.py'))) {
            return {
                command: process.platform === 'win32' ? 'python' : 'python3',
                args: ['-m', 'flast_cli.cli'],
                pythonPath: candidate,
            };
        }
    }
    return { command: 'flast', args: [] };
}
function resolveProjectDir(configuredProjectDir) {
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
function scheduleRefresh() {
    if (refreshTimer) {
        clearTimeout(refreshTimer);
    }
    refreshTimer = setTimeout(() => {
        void refreshDiagnostics();
    }, 1000);
}
async function initializeWorkspace() {
    const config = readConfig();
    await runSimpleCommand(['init', config.framework], 'Initialize FLAST workspace');
    await refreshDiagnostics();
}
async function observeTestRuns() {
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
async function runSimpleCommand(baseArgs, title, includePytestArgs = false) {
    const config = readConfig();
    const args = [
        ...baseArgs,
        '--project-dir',
        config.projectDir,
        ...(includePytestArgs ? frameworkArgs(config.frameworkArgs) : [])
    ];
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title, cancellable: true }, async (_progress, token) => {
        await runFlast(config, args, token);
    });
    vscode.window.showInformationMessage(`${title} completed.`);
}
async function refreshDiagnostics() {
    const config = readConfig();
    const args = [
        'auto',
        config.framework,
        '--format',
        'json',
        '--project-dir',
        config.projectDir,
        ...frameworkArgs(config.frameworkArgs)
    ];
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Window, title: 'Refreshing FLAST diagnostics', cancellable: true }, async (_progress, token) => {
        const result = await runFlast(config, args, token);
        const payload = parseDiagnostics(result.stdout);
        applyDiagnostics(payload, config);
    });
}
function frameworkArgs(args) {
    return args.flatMap((arg) => ['--framework-arg', arg]);
}
function runFlast(config, args, token) {
    if (activeRun) {
        activeRun.kill();
        activeRun = undefined;
    }
    const commandArgs = [...config.executableArgs, ...args];
    output.appendLine(`$ ${config.executablePath} ${commandArgs.join(' ')}`);
    return new Promise((resolve, reject) => {
        const env = { ...process.env };
        if (config.pythonPath) {
            env.PYTHONPATH = env.PYTHONPATH
                ? `${config.pythonPath}${path.delimiter}${env.PYTHONPATH}`
                : config.pythonPath;
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
        child.stdout.on('data', (chunk) => {
            const text = chunk.toString();
            stdout += text;
            output.append(text);
        });
        child.stderr.on('data', (chunk) => {
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
    }).catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        output.show(true);
        vscode.window.showErrorMessage(`FLAST failed: ${message}`);
        throw error;
    });
}
function parseDiagnostics(stdout) {
    try {
        const payload = JSON.parse(stdout);
        if (!Array.isArray(payload.diagnostics)) {
            throw new Error('Missing diagnostics array.');
        }
        return payload;
    }
    catch (error) {
        throw new Error(`FLAST returned invalid diagnostics JSON: ${error.message}`);
    }
}
function applyDiagnostics(payload, config) {
    const byFile = new Map();
    for (const record of payload.diagnostics) {
        if (record.predicted_label !== 'flaky' || !record.file_path || record.score < config.warningThreshold) {
            continue;
        }
        const line = Math.max((record.line ?? 1) - 1, 0);
        const range = new vscode.Range(line, 0, line, Number.MAX_SAFE_INTEGER);
        const diagnostic = new vscode.Diagnostic(range, record.message, config.diagnosticSeverity === 'warning'
            ? vscode.DiagnosticSeverity.Warning
            : vscode.DiagnosticSeverity.Information);
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
async function openLatestReport() {
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
//# sourceMappingURL=extension.js.map