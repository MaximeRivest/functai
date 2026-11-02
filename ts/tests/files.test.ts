/**
 * Files as inputs (contract/functions.md, "Pictures, sounds, documents and
 * files"): Attachments' `att()` results through `functai/files`, offline (a
 * fake router stands in for every provider). Without attachments-dev here,
 * the only test is that a function reading files refuses `files-unavailable`.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { ai, calls, configure, fromManifest, InterfaceError, remote, serve, t, tool, toManifest, withSettings } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

type Rec = Record<string, any>;
const HERE = dirname(fileURLToPath(import.meta.url));
const files = await import("../src/files.ts").catch(() => null);
const attachments = await import("attachments-dev").catch(() => null);
const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const answer = (text: string) => new FakeRouter([], () => `<result>\n${text}\n</result>`);
const parts = (router: FakeRouter, i = 0) => router.requests[i]!.messages.flatMap((m) => m.parts as unknown as Rec[]);
const words = (router: FakeRouter, i = 0) => parts(router, i).filter((p) => p["type"] === "text").map((p) => p["text"]).join("");
const folder = () => mkdtempSync(join(tmpdir(), "functai-files-"));

test("without functai/files, a function saved elsewhere that reads files refuses files-unavailable", () => {
  const manifest = JSON.stringify((JSON.parse(readFileSync(join(HERE, "../../contract/cases/files/01-a-report.json"), "utf8")) as Rec).manifest);
  // a fresh process that never imports functai/files
  const script = `import { fromManifest, FilesUnavailable } from "./src/index.ts";
    try { fromManifest(${manifest}); console.log("loaded"); } catch (e) { console.log(e instanceof FilesUnavailable ? e.code : String(e)); }`;
  const out = execFileSync(process.execPath, ["--conditions=functai-source", "--input-type=module", "-e", script],
    { cwd: join(HERE, ".."), encoding: "utf8" });
  assert.equal(out.trim(), "files-unavailable");
});

if (!files || !attachments) {
  test("files with Attachments (skipped: attachments-dev is not installed here)", { skip: true }, () => {});
} else {
  const { artifacts } = files;
  const { att } = attachments;
  const summarizeDef = (router: FakeRouter, more: Rec = {}) => ai("summarize", {
    description: "The report in three sentences.", input: { report: artifacts({ description: "what to summarize" }) },
    output: t.string(), router, ...more,
  });
  const notes = () => {
    const dir = folder();
    const path = join(dir, "notes.md");
    writeFileSync(path, "Revenue rose 12% in the third quarter.\n");
    return { dir, path };
  };
  const PAGED = [{
    text: "Page one.\n\nPage two.", images: [
      { name: "r-page-1.png", mimetype: "image/png", bytes_b64: PNG, page: 1 },
      { name: "r-page-2.png", mimetype: "image/png", bytes_b64: PNG, page: 2 }],
    audio: [], video: [],
    meta: { source: "r.pdf", kind: "pdf", segments: [
      { kind: "page", label: "page 1", start: 0, end: 9, page: 1 }, { kind: "page", label: "page 2", start: 11, end: 20, page: 2 }] },
  }];

  test("att()'s result is shown as Attachments' parts, and logged as its wire form", async () => {
    const { dir, path } = notes();
    const router = answer("Revenue rose.");
    const summarize = summarizeDef(router, { logCalls: dir });
    const a = await att(path);
    assert.equal(await summarize({ report: a }), "Revenue rose.");
    assert.match(words(router), /<report>\n## .*notes\.md\nRevenue rose 12% in the third quarter\.\n<\/report>/);
    const [record] = calls(summarize, { folder: dir });
    assert.deepEqual((record!["inputs"] as Rec)["report"], JSON.parse(JSON.stringify(a.toWire())));
    // the wire form (a served call, a replay, another language) sends the same request
    await summarize({ report: a.toWire() });
    assert.deepEqual(router.requests[0]!.messages, router.requests[1]!.messages);
  });

  test("each page's picture comes after that page's text, as a picture", async () => {
    const router = answer("ok");
    await summarizeDef(router)({ report: PAGED });
    assert.deepEqual(parts(router).map((p) => p["type"]), ["text", "image", "text", "image", "text"]);
    assert.ok(!words(router).includes(PNG));
  });

  test("no files is (no content)", async () => {
    const router = answer("Nothing to say.");
    await summarizeDef(router)({ report: await att([]) });
    assert.match(words(router), /<report>\n\(no content\)\n<\/report>/);
  });

  test("a file att() could not read refuses the call before anything is sent, naming it and why", async () => {
    const router = answer("x");
    const summarize = summarizeDef(router);
    const dir = folder();
    const failed = await withSettings({}, () => att(join(dir, "scan.pdf")));
    await assert.rejects(summarize({ report: failed }), (e: unknown) => e instanceof InterfaceError && e.code === "interface-input"
      && e.field === "report" && /scan\.pdf/.test(e.message) && /could not be read/.test(e.message));
    await assert.rejects(summarize({ report: [{ text: 3 }] as never }), (e: unknown) => e instanceof InterfaceError && /wire form/.test(e.message));
    await assert.rejects(summarize({ report: "report.pdf" as never }), (e: unknown) => e instanceof InterfaceError && e.code === "interface-input");
    assert.equal(router.requests.length, 0);
  });

  test("files are a whole input: not optional, not in a list or a record, not an output or a tool's input", () => {
    const def = (input: Rec, output: unknown = t.string()) => () => ai("f", { description: "d", input, output } as never);
    assert.throws(def({ report: t.optional(artifacts()) }), (e: unknown) => e instanceof InterfaceError && e.code === "interface-malformed"
      && /att\(\[\]\)/.test(e.message));
    assert.throws(def({ reports: t.list(artifacts()) }), (e: unknown) => e instanceof InterfaceError && e.code === "interface-malformed");
    assert.throws(def({ report: t.withDefault(artifacts(), [] as never) }), (e: unknown) => e instanceof InterfaceError);
    assert.throws(() => t.object({ report: artifacts() }), TypeError);
    assert.throws(def({ q: t.string() }, artifacts()), (e: unknown) => e instanceof InterfaceError && e.code === "interface-malformed");
    assert.throws(() => tool("read", { input: { report: artifacts() } }, () => ""), TypeError);
  });

  test("saved and loaded, a function reading files names the type and sends the same request", async () => {
    const router = answer("ok");
    const summarize = summarizeDef(router);
    const manifest = JSON.parse(JSON.stringify(toManifest(summarize)));
    const node = manifest.nodes[manifest.entry];
    assert.equal(node.interface.inputs[0].type, "Artifacts");
    assert.equal(node.ai.signature.fields.find((f: Rec) => f.name === "report").type, "Artifacts");
    const loaded = fromManifest(manifest);
    assert.equal(loaded.version, summarize.version);
    await summarize({ report: PAGED });
    await withSettings({ router }, () => loaded({ report: PAGED } as never));
    assert.deepEqual(router.requests[0]!.messages, router.requests[1]!.messages);
  });

  test("served and called remotely: att()'s result travels as its wire form, a failed file is refused before it is sent", async () => {
    const router = answer("served");
    const server = await serve(summarizeDef(router), { port: 0 });
    try {
      const program = await remote(`http://127.0.0.1:${(server.address() as { port: number }).port}`);
      const { path } = notes();
      assert.equal(await program({ report: await att(path) } as never), "served");
      assert.match(words(router), /Revenue rose 12%/);
      await assert.rejects(program({ report: await att(join(folder(), "gone.pdf")) } as never),
        (e: unknown) => e instanceof InterfaceError && e.code === "interface-input");
      assert.equal(router.requests.length, 1);
    } finally {
      server.close();
    }
  });

  test("a conversation keeps the files as their wire form and shows them again", async () => {
    const router = answer("ok");
    const ask = ai("ask", { description: "Answer from the report.", input: { report: artifacts(), question: t.string() }, output: t.string(), router });
    const chat = ask.conversation();
    await chat({ report: PAGED, question: "What is on page one?" });
    await chat({ report: await att([]), question: "And page two?" });
    assert.equal(parts(router, 1).filter((p) => p["type"] === "image").length, 2, "the first turn's pages, in the second request");
  });
}
