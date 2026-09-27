/** Where a program was written: found by who called `ai()`, never by the path of a file. */

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { ai, configure, t, type AnyAIFunction } from "../src/index.ts";
import { definedAt, frameLocation } from "../src/fn.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

const here = fileURLToPath(import.meta.url);

async function loggedProgram(fn: AnyAIFunction): Promise<Record<string, any>> {
  const folder = mkdtempSync(join(tmpdir(), "functai-where-"));
  await fn.using({ router: new FakeRouter(["<result>\nok\n</result>"]) as never, logCalls: folder })("hi");
  const [day] = readdirSync(folder);
  const [file] = readdirSync(join(folder, day!));
  return JSON.parse(readFileSync(join(folder, day!, file!), "utf8").split("\n")[0]!).program;
}

test("a program is placed at the line that called ai()", async () => {
  const fn = ai("placed", { input: { text: t.string() } });
  const line = readFileSync(here, "utf8").split("\n").findIndex((l) => l.includes(`ai("placed"`)) + 1;
  const program = await loggedProgram(fn);
  assert.equal(program.file, here);
  assert.equal(program.line, line);
  assert.equal(program.module, "where.test");
});

test("a program written in a folder that looks like this package's own is still placed there", async () => {
  // bundled, or in a project that happens to be called functai, a path says nothing about whose code it is
  const dir = join(mkdtempSync(join(tmpdir(), "functai-where-")), "functai", "src");
  mkdirSync(dir, { recursive: true });
  const app = join(dir, "app.ts");
  writeFileSync(app, [
    `import { ai, t } from ${JSON.stringify(pathToFileURL(join(here, "..", "..", "src", "index.ts")).href)};`,
    "",
    `export const echo = ai("echo", { input: { text: t.string() } });`,
    "",
  ].join("\n"));
  const { echo } = await import(pathToFileURL(app).href);
  const program = await loggedProgram(echo);
  assert.equal(program.file, app);
  assert.equal(program.line, 3);
  assert.equal(program.module, "app");
});

test("the frame is the caller of the function given, however deep", () => {
  function entry() { return definedAt(entry); }
  function helper() { return entry(); }
  const where = helper();
  assert.equal(where.file, here);
  assert.equal(where.module, "where.test");
  assert.match(readFileSync(here, "utf8").split("\n")[where.line! - 1]!, /return entry\(\)/);
});

test("frame lines are read on every system and in every engine", () => {
  const cases: [string, { file: string; line: number } | null][] = [
    // Windows, V8: backslashes, a drive, spaces and parentheses in the path
    ["    at Object.<anonymous> (C:\\Users\\Ann Lee\\shop (copy)\\main.js:12:7)", { file: "C:\\Users\\Ann Lee\\shop (copy)\\main.js", line: 12 }],
    ["    at C:\\Users\\ann\\shop\\main.js:3:1", { file: "C:\\Users\\ann\\shop\\main.js", line: 3 }],
    ["    at \\\\server\\share\\main.js:5:2", { file: "\\\\server\\share\\main.js", line: 5 }],
    // POSIX, V8
    ["    at async main (/home/ann/shop/main.ts:4:1)", { file: "/home/ann/shop/main.ts", line: 4 }],
    ["    at new Shop (/home/ann/shop/main.ts:9:3)", { file: "/home/ann/shop/main.ts", line: 9 }],
    ["    at file:///home/ann/my%20shop/main.mjs:8:5", { file: "/home/ann/my shop/main.mjs", line: 8 }],
    // Firefox, Safari: name@where; an `@` in the path stays
    ["mood@https://shop.example/assets/app.js:10:3", { file: "https://shop.example/assets/app.js", line: 10 }],
    ["@http://localhost:5173/node_modules/@shop/ui/app.js:2:9", { file: "http://localhost:5173/node_modules/@shop/ui/app.js", line: 2 }],
    // no place: eval, native
    ["    at eval (eval at run (/home/ann/shop/main.js:1:1), <anonymous>:1:1)", null],
    ["    at Array.map (<anonymous>)", null],
    ["    at native", null],
  ];
  for (const [frame, want] of cases) assert.deepEqual(frameLocation(frame), want, frame);
});
