/**
 * Pictures, sounds, videos and documents as inputs (contract/functions.md,
 * "Pictures, sounds, documents and files"), offline: a fake router stands in
 * for every provider.
 */

import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as lmcc from "lmcc";
import {
  ai, audio, bakeExamples, BakeError, calls, configure, document, fromManifest, image, InterfaceError, t, toManifest, video, withSettings,
} from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

type Rec = Record<string, unknown>;
const PNG_A = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const PNG_B = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
const answer = (text: string) => new FakeRouter([], () => `<result>\n${text}\n</result>`);
const parts = (router: FakeRouter, i = 0) => router.requests[i]!.messages.flatMap((m) => m.parts as unknown as Rec[]);
const media = (router: FakeRouter, i = 0) => parts(router, i).filter((p) => p["type"] !== "text");
const words = (router: FakeRouter, i = 0) => parts(router, i).filter((p) => p["type"] === "text").map((p) => p["text"]).join("");

function folder(): string {
  return mkdtempSync(join(tmpdir(), "functai-media-"));
}

test("t.image() takes lm15's image({ path }): the file's bytes are sent as a picture, and logged as lm15's part data", async () => {
  const dir = folder();
  const png = join(dir, "shelf.png");
  writeFileSync(png, Buffer.from(PNG_A, "base64"));
  const router = answer("4");
  const count = ai("count_items", { description: "How many items are on the shelf?", input: { photo: t.image() }, output: t.integer(),
    router, logCalls: dir });
  assert.equal(await count({ photo: image({ path: png }) }), 4);
  const [sent] = media(router);
  assert.equal(sent!["type"], "image");
  assert.equal(sent!["data"], PNG_A);
  assert.equal(sent!["path"], undefined);
  assert.ok(!words(router).includes(PNG_A), "the bytes are never text");
  const [record] = calls(count, { folder: dir });
  // as Python logs an ImagePart: type and media_type (never lm15-ts's camelCase), the bytes not the name
  assert.deepEqual(record!["inputs"], { photo: { type: "image", media_type: "image/png", data: PNG_A } });
});

test("a picture as lm15's part and as plain data sends the same request", async () => {
  const router = answer("blue");
  const colour = ai("colour", { description: "The main colour.", input: { picture: t.image() }, output: t.string(), router });
  await colour({ picture: image({ data: Buffer.from(PNG_A, "base64"), mediaType: "image/png" }) });
  await colour({ picture: { type: "image", media_type: "image/png", data: PNG_A } });
  await colour({ picture: image({ url: "https://example.com/blue.png" }) });
  assert.deepEqual(router.requests[0]!.messages, router.requests[1]!.messages);
  assert.equal(media(router, 2)[0]!["url"], "https://example.com/blue.png");     // a web address is left for the provider
});

test("a list of pictures sends each one as a picture, in order; an empty list sends none", async () => {
  const router = answer("2");
  const brighter = ai("brighter", { description: "Which picture is brighter? Its position, from 1.",
    input: { pictures: t.list(t.image()) }, output: t.integer(), router });
  await brighter({ pictures: [image({ data: PNG_A, mediaType: "image/png" }), { media_type: "image/png", data: PNG_B }] });
  assert.deepEqual(media(router).map((p) => p["data"]), [PNG_A, PNG_B]);
  assert.ok(![PNG_A, PNG_B].some((b) => words(router).includes(b)));
  await brighter({ pictures: [] });
  assert.deepEqual(media(router, 1), []);
});

test("an optional picture may be left out: the model reads null", async () => {
  const router = answer("A mug.");
  const describeItem = ai("describe_item", { description: "Describe the item for a listing.",
    input: { title: t.string(), photo: t.optional(t.image()) }, output: t.string(), router });
  await describeItem({ title: "A blue mug" });
  assert.deepEqual(media(router), []);
  assert.match(words(router), /<photo>\nnull\n<\/photo>/);
  await describeItem({ title: "A blue mug", photo: image({ data: PNG_A, mediaType: "image/png" }) });
  assert.equal(media(router, 1).length, 1);
});

test("a record holding a picture is refused before anything is sent: no layout shows it, none may write it as text", async () => {
  const router = answer("x");
  const listing = ai("listing", { description: "Describe the item.",
    input: { item: t.object({ title: t.string(), photo: t.image() }) }, output: t.string(), router });
  await assert.rejects(listing({ item: { title: "mug", photo: { media_type: "image/png", data: PNG_A } } }),
    (e: unknown) => lmcc.isRefusal(e) && (e as lmcc.Refusal).code === "no-format");
  assert.equal(router.requests.length, 0);
});

test("a sound, a document and a video are each sent as a part of their kind", async () => {
  const router = answer("Minutes.");
  const minutes = ai("minutes", { description: "Write the minutes, following the agenda.",
    input: { recording: t.audio(), agenda: t.document(), clip: t.video() }, output: t.string(), router });
  await minutes({
    recording: audio({ data: "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=", mediaType: "audio/wav" }),
    agenda: document({ data: "JVBERi0xLjQK", mediaType: "application/pdf" }),
    clip: video({ url: "https://example.com/clip.mp4" }),
  });
  assert.deepEqual(media(router).map((p) => p["type"]), ["audio", "document", "video"]);
});

test("a picture function's version holds its instruction (its sample is a picture by web address)", () => {
  const a = ai("colour", { description: "The main colour.", input: { picture: t.image() }, output: t.string() });
  const b = ai("colour", { description: "The colour of the sky in it.", input: { picture: t.image() }, output: t.string() });
  const refused = lmcc.sha256({ request: "refused:value-invalid" } as lmcc.Json);
  assert.notEqual(a.version, b.version);
  assert.ok(![a.version, b.version].includes(refused));
  const probe = a.probeRequest() as { messages: { parts: Rec[] }[] };
  assert.deepEqual(probe.messages[0]!.parts.find((p) => p["type"] === "image"),
    { type: "image", media_type: "image/png", url: "https://example.com/sample.png" });
});

test("a picture that is not a picture is refused before anything is sent", async () => {
  const router = answer("x");
  const colour = ai("colour", { description: "The main colour.", input: { picture: t.image() }, output: t.string(), router });
  // text is lmcc's to refuse (FunctAI's binding never reads the media keyword: programs.md)
  await assert.rejects(colour({ picture: "shelf.png" as never }), (e: unknown) => lmcc.isRefusal(e) && (e as lmcc.Refusal).code === "value-invalid");
  // bytes in hand have no JSON form: refused at binding, saying how to give a picture
  await assert.rejects(colour({ picture: { media_type: "image/png", data: new Uint8Array([1, 2]) } as never }),
    (e: unknown) => e instanceof InterfaceError && e.code === "interface-input" && /image\(\{ path \}\)/.test(e.message));
  assert.equal(router.requests.length, 0);
});

test("a picture function saved and loaded sends the same request and has the same version", async () => {
  const router = answer("blue");
  const colour = ai("colour", { description: "The main colour.", input: { picture: t.image(), note: t.string() }, output: t.string(), router });
  const loaded = fromManifest(JSON.parse(JSON.stringify(toManifest(colour))));
  assert.equal(loaded.version, colour.version);
  const given = { picture: image({ data: PNG_A, mediaType: "image/png" }), note: "the label" };
  await colour(given);
  await withSettings({ router }, () => loaded(given as never));
  assert.deepEqual(router.requests[0]!.messages, router.requests[1]!.messages);
});

test("a picture in a conversation is kept as data and shown again in the next turn", async () => {
  const router = answer("ok");
  const look = ai("look", { description: "Answer about the picture.", input: { picture: t.optional(t.image()), question: t.string() },
    output: t.string(), router });
  const chat = look.conversation();
  await chat({ picture: image({ data: PNG_A, mediaType: "image/png" }), question: "What is it?" });
  await chat({ question: "And its colour?" });
  assert.equal(media(router, 1).filter((p) => p["data"] === PNG_A).length, 1, "the first turn's picture, in the second request");
});

test("training examples refuse pictures: a generative student reads text", () => {
  const colour = ai("colour", { description: "The main colour.", input: { picture: t.image() }, output: t.string() });
  assert.throws(() => bakeExamples(colour, [{ picture: { media_type: "image/png", data: PNG_A }, result: "blue" }]),
    (e: unknown) => e instanceof BakeError && /images and files are not trained yet/.test(e.message));
});

test("a streamed call sends the picture as a picture", async () => {
  const router = answer("blue");
  const colour = ai("colour", { description: "The main colour.", input: { picture: t.image() }, output: t.string(), router });
  const stream = colour.stream({ picture: image({ data: PNG_A, mediaType: "image/png" }) });
  for await (const _ of stream) { /* drain */ }
  assert.equal(media(router)[0]!["data"], PNG_A);
});
