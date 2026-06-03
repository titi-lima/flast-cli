const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

function outcomeFromResult(result) {
  const state = String((result && (result.state || result.status)) || "").toLowerCase();
  if (state === "pass" || state === "passed") return "passed";
  if (state === "fail" || state === "failed") return "failed";
  if (state === "skip" || state === "skipped" || state === "todo") return "skipped";
  return "error";
}

function relativeFile(filePath, projectRoot) {
  if (!filePath) return "";
  const absolute = path.isAbsolute(filePath) ? filePath : path.resolve(projectRoot, filePath);
  const realRoot = fs.realpathSync(projectRoot);
  const realFile = fs.existsSync(absolute) ? fs.realpathSync(absolute) : path.resolve(absolute);
  return path.relative(realRoot, realFile).split(path.sep).join("/");
}

function taskName(task) {
  return String(task && (task.name || task.title || task.id) || "").trim();
}

function testIdFor(task, ancestors, projectRoot) {
  const file = relativeFile(task.file && task.file.filepath || task.filepath || task.file || "", projectRoot);
  const names = [...ancestors, taskName(task)].filter(Boolean);
  return [file, ...names].filter(Boolean).join("::");
}

function walk(task, ancestors, projectRoot, collected, outcomes) {
  const children = task && (task.tasks || task.children || task.suite || []);
  const type = String((task && task.type) || "").toLowerCase();

  if (type === "test" || task && task.result && !children.length) {
    const testId = testIdFor(task, ancestors, projectRoot);
    if (testId) {
      collected.set(testId, { test_id: testId });
      outcomes.set(testId, {
        test_id: testId,
        outcome: outcomeFromResult(task.result),
      });
    }
    return;
  }

  const nextAncestors = type === "suite" ? [...ancestors, taskName(task)] : ancestors;
  for (const child of children || []) {
    walk(child, nextAncestors, projectRoot, collected, outcomes);
  }
}

class FlastReporter {
  onFinished(files) {
    const projectRoot = fs.realpathSync(process.cwd());
    const workspace = path.join(projectRoot, ".flast");
    if (!fs.existsSync(path.join(workspace, "config.toml"))) return;

    const runId = crypto.randomUUID();
    const manifestPath = path.join(workspace, "runs", runId, "manifest.json");
    const collected = new Map();
    const outcomes = new Map();

    for (const file of files || []) {
      walk(file, [], projectRoot, collected, outcomes);
    }

    fs.mkdirSync(path.dirname(manifestPath), { recursive: true });
    fs.writeFileSync(
      manifestPath,
      JSON.stringify({
        schema_version: 1,
        framework: "vitest",
        run_id: runId,
        project_root: projectRoot,
        tool: { name: "vitest", version: "unknown" },
        collected: [...collected.values()].sort((left, right) => left.test_id.localeCompare(right.test_id)),
        outcomes: [...outcomes.values()].sort((left, right) => left.test_id.localeCompare(right.test_id)),
      }, null, 2) + "\n",
      "utf8"
    );
  }
}

module.exports = FlastReporter;
