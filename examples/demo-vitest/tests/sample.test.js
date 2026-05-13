import { describe, expect, test } from "vitest";
import fs from "node:fs";
import path from "node:path";

const statePath = path.join(import.meta.dirname, "flaky-state.txt");

function nextRunCount() {
  let value = 0;
  if (fs.existsSync(statePath)) {
    value = Number(fs.readFileSync(statePath, "utf8"));
  }
  fs.writeFileSync(statePath, String(value + 1));
  return value;
}

describe("demo suite", () => {
  test("flaky toggle", () => {
    expect(nextRunCount() % 2).toBe(0);
  });

  test("stable pass", () => {
    expect(1 + 1).toBe(2);
  });
});
