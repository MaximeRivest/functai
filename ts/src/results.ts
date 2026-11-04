/**
 * What a tool returns, as the model is shown it (contract/tools.md, "What a
 * tool returns"): a text; or, when it returned a part (a picture, a sound, a
 * video, a document: lm15's `image(...)` and the like) or a list holding
 * one, a list of lm15 parts as data (`{ type, media_type, data }`), each
 * text a text part. Anything else is its JSON, as text.
 *
 * ```ts
 * const look = tool("look", { description: "A screenshot of the page.", input: { page: t.string() }, effects: "reads" },
 *   async ({ page }) => [`page ${page}`, image({ data: await shoot(page), mediaType: "image/png" })]);
 * ```
 */

import { Part, isPart } from "@lm15/lm15";
import { builtin } from "./host.ts";
import { writeData } from "./values.ts";

type Rec = Record<string, unknown>;

/** A tool's result in data: text, or lm15 parts in their JSON form. */
export type ToolOutput = string | readonly Rec[];

/** The kinds of lm15 part a tool's result may hold. */
export const RESULT_PARTS = ["text", "image", "audio", "video", "document"] as const;

const isResultPart = (v: unknown): boolean =>
  typeof v === "object" && v !== null && !Array.isArray(v) && isPart(v)
  && (RESULT_PARTS as readonly string[]).includes((v as unknown as Rec)["type"] as string);

/** One part as data; a part read from a file holds the file's bytes, so the result is whole wherever it is read again. */
function partData(v: unknown): Rec {
  const r = v as Rec;
  if (!(RESULT_PARTS as readonly string[]).includes(r["type"] as string)) {
    throw new TypeError(`a tool's result holds text, pictures, sounds, videos and documents; not a ${String(r["type"])} part`);
  }
  const json = (Object.hasOwn(r, "media_type") || r["type"] === "text" && !Object.hasOwn(r, "mediaType") ? { ...r } : Part.toJSON(r as never)) as Rec;
  if (typeof json["path"] === "string" && json["data"] === undefined) {
    const fs = builtin("node:fs");
    if (!fs) throw new TypeError(`a part read from a file (${json["path"]}) needs a file system; give its data`);
    const { path, ...rest } = json;
    return { ...rest, data: fs.readFileSync(path as string).toString("base64") };
  }
  return json;
}

/** What a tool returned, as its result (contract/tools.md, "What a tool returns"). */
export function resultOf(out: unknown): ToolOutput {
  if (typeof out === "string") return out;
  if (isResultPart(out)) return [partData(out)];
  if (Array.isArray(out) && out.some(isResultPart)) {
    return out.map((x) => isResultPart(x) ? partData(x) : { type: "text", text: typeof x === "string" ? x : writeData(x) });
  }
  return writeData(out);                                   // members in the value's order
}

/** A result given as data (a plugin's change, a result given or delivered later): text, or lm15 parts. */
export function resultData(v: unknown, where: string): ToolOutput {
  if (typeof v === "string") return v;
  if (isResultPart(v)) return [partData(v)];
  if (Array.isArray(v) && v.length) {
    return v.map((p) => {
      if (typeof p === "string") return { type: "text", text: p };
      if (isResultPart(p)) return partData(p);
      throw new TypeError(`${where}: a part of a tool's result is text, a picture, a sound, a video or a document (an lm15 part)`);
    });
  }
  throw new TypeError(`${where} is text, or a list of lm15 parts (text, pictures, sounds, videos, documents)`);
}

/** A result's text: itself, or its text parts joined by a new line. */
export function resultText(output: ToolOutput): string {
  if (typeof output === "string") return output;
  return output.filter((p) => p["type"] === "text").map((p) => String(p["text"] ?? "")).join("\n");
}

/** How many pictures a result holds. */
export function picturesIn(output: ToolOutput): number {
  return typeof output === "string" ? 0 : output.filter((p) => p["type"] === "image").length;
}

/** A result that came later (`later(key)`), found when a turn resumes: the `toolResult` hooks have not seen it yet. */
export class Delivered {
  readonly output: ToolOutput;
  constructor(output: ToolOutput) {
    this.output = output;
  }
}

/** What a tool returns when its result comes later: `later(key)`. */
export class Later {
  readonly key: string;
  constructor(key: string) {
    this.key = key;
  }
}

/**
 * What a tool returns when its result comes later (contract/tools.md, "A
 * result that comes later"): it started something that ends elsewhere (a
 * job, a cloud browser, a person's reply) and whoever learns the result
 * delivers it under `key`. Only in a conversation's turn: the turn runs the
 * step's other tool calls, then stops, saved, and waits (`Waiting`);
 * `chat.deliver(key, result)`, from any process, gives the tool its result
 * and the turn goes on. The tool does not run again. Anywhere else nothing
 * could wait for it: `later-unavailable`. `key` names one result in its
 * conversation (put an id in it: `"harvest:8f2c"`).
 *
 * ```ts
 * const startHarvest = tool("start_harvest", { description: "Visit a site in a cloud browser.", input: { url: t.string() },
 *   effects: "changes" }, async ({ url }) => later(`harvest:${(await harvests.start(url)).id}`));
 * // in the webhook, any process:
 * await chat.deliver(`harvest:${id}`, report);
 * ```
 */
export function later(key: string): Later {
  if (typeof key !== "string" || !key.trim()) throw new TypeError("a later result's key is a non-empty text (put an id in it: 'harvest:8f2c')");
  return new Later(key);
}
