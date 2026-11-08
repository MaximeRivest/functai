/**
 * A student a service trains, from its own published files (contract/training.md):
 * how a conversation, pictures included, becomes the exact tokens it is trained
 * on and called with, where its reply starts and ends, and how its generated
 * tokens read back as a reply.
 *
 * ```ts
 * const st = await Student.load("Qwen/Qwen3.5-4B");          // tokenizer.json, the chat template, picture settings
 * const { prompt, full, replyStart } = st.render(messages, pictures, reply);
 * st.readReply(generatedTokens);                              // the reply text the function's layout reads
 * ```
 *
 * Needs `@huggingface/tokenizers` and `@huggingface/jinja` (about 1 MB, no
 * native code): `npm install @huggingface/tokenizers @huggingface/jinja`.
 * Nothing runs a model here: the service does.
 */

import { BakeError } from "./errors.ts";
import { STUDENTS } from "./generated/contract.ts";
import { builtin } from "./host.ts";
import { parseForeign, writeData } from "./values.ts";

type Rec = Record<string, unknown>;

/** A family's picture rule with its numbers (contract/training.md §4). */
export interface PictureRule {
  readonly rule: "grid" | "qwen";
  readonly placeholder: string;
  readonly patch: number;
  readonly merge?: number;
  readonly min_pixels?: number;
  readonly max_pixels?: number;
}

/** A picture as a service takes it: its bytes, size, format and hash. */
export interface Picture {
  readonly data: Uint8Array;
  readonly width: number;
  readonly height: number;
  readonly format: "png" | "jpeg";
  /** `sha256:<hex>` of the bytes. */
  readonly sha256: string;
}

/** Tokens, or a picture standing for `tokens` positions (picture `picture` of the conversation). */
export type Chunk = { readonly text: number[] } | { readonly picture: number; readonly tokens: number };

/** A conversation as a student's tokens (contract/training.md §3). */
export interface Rendered {
  readonly prompt: Chunk[];
  /** With the reply: prompt and reply (and what the template writes after it). */
  readonly full: Chunk[] | null;
  /** Where the reply starts (the prompt's length, a picture counting its positions). */
  readonly replyStart: number;
  /** The template wrote the last reply differently from a generation prompt (§3.2). */
  readonly prefixFallback: boolean;
}

// ------------------------------------------------------------------ pictures

/** A picture's width, height and format from its bytes (PNG's IHDR, a JPEG's first frame header), or null. */
export function pictureSize(b: Uint8Array): { width: number; height: number; format: "png" | "jpeg" } | null {
  const u32 = (i: number) => ((b[i]! << 24) >>> 0) + (b[i + 1]! << 16) + (b[i + 2]! << 8) + b[i + 3]!;
  const u16 = (i: number) => (b[i]! << 8) + b[i + 1]!;
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (b.length >= 24 && png.every((x, i) => b[i] === x)) return { width: u32(16), height: u32(20), format: "png" };
  if (b.length >= 4 && b[0] === 0xff && b[1] === 0xd8) {
    let i = 2;
    while (i + 4 <= b.length) {
      const marker = b[i + 1]!;
      const len = u16(i + 2);
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        if (i + 9 > b.length) return null;
        return { width: u16(i + 7), height: u16(i + 5), format: "jpeg" };
      }
      i += 2 + len;
    }
  }
  return null;
}

/** Round half to even, as Python's `round` (the qwen rule says so). */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** Inkling's rule: ceil(w/P)·ceil(h/P), no resizing. */
export const gridTokens = (w: number, h: number, patch: number): number => Math.ceil(w / patch) * Math.ceil(h / patch);

/** Qwen's rule (smart resize into the pixel range, multiples of patch·merge); null past an aspect ratio of 200. */
export function qwenTokens(w: number, h: number, patch: number, merge: number, minPixels: number, maxPixels: number): number | null {
  const f = patch * merge;
  if (Math.max(w, h) / Math.min(w, h) > 200) return null;
  let hb = roundHalfEven(h / f) * f;
  let wb = roundHalfEven(w / f) * f;
  if (hb * wb > maxPixels) {
    const b = Math.sqrt((h * w) / maxPixels);
    hb = Math.max(f, Math.floor(h / b / f) * f);
    wb = Math.max(f, Math.floor(w / b / f) * f);
  } else if (hb * wb < minPixels) {
    const b = Math.sqrt(minPixels / (h * w));
    hb = Math.ceil((h * b) / f) * f;
    wb = Math.ceil((w * b) / f) * f;
  }
  return (hb / f) * (wb / f);
}

/** How many positions a picture takes under a rule (refused when the rule refuses it). */
export function pictureTokens(rule: PictureRule, w: number, h: number): number {
  if (rule.rule === "grid") return gridTokens(w, h, rule.patch);
  const n = qwenTokens(w, h, rule.patch, rule.merge!, rule.min_pixels!, rule.max_pixels!);
  if (n === null) throw new BakeError("training-picture-format", `a ${w}×${h} picture is too long and thin for this student (its sides' ratio is over 200)`);
  return n;
}

function sha256Hex(data: Uint8Array): string {
  const crypto = builtin("node:crypto");
  if (!crypto) throw new Error("hashing pictures needs node:crypto");
  return crypto.createHash("sha256").update(data).digest("hex");
}

/** A picture from its bytes; refused when it is not a PNG or a JPEG (`training-picture-format`). */
export function picture(data: Uint8Array): Picture {
  const size = pictureSize(data);
  if (!size) throw new BakeError("training-picture-format", "a picture a service trains on is a PNG or a JPEG; convert it (another format, or bytes that are not a picture)");
  return { data, ...size, sha256: `sha256:${sha256Hex(data)}` };
}

const fromBase64 = (s: string): Uint8Array => new Uint8Array(Buffer.from(s, "base64"));

// ------------------------------------------------------------------ conversations

/**
 * An lm15 request (canonical JSON) as a student's conversation
 * (contract/training.md §2): the system text first, a developer message as a
 * system one; a message with only text parts has its text as `content`, one
 * with pictures a list (adjacent text joined, each picture `{type: "image"}`);
 * the pictures, numbered in order, beside.
 */
export function studentConversation(request: Rec): { messages: Rec[]; pictures: Picture[] } {
  const messages: Rec[] = [];
  const pictures: Picture[] = [];
  const system = request["system"];
  if (system !== undefined && system !== null && (typeof system !== "string" || system)) {
    messages.push({ role: "system", content: typeof system === "string" ? system : (system as Rec[]).map((p) => p["text"]).join("") });
  }
  for (const m of request["messages"] as Rec[]) {
    const role = m["role"] as string;
    if (!["user", "assistant", "system", "developer"].includes(role)) throw new BakeError("bake-rows", `a student reads chats; the request has a ${writeData(role)} message (tool calls are not trained yet)`);
    const parts: Rec[] = [];
    for (const p of m["parts"] as Rec[]) {
      if (p["type"] === "text") {
        const last = parts.at(-1);
        if (last && last["type"] === "text") last["text"] = (last["text"] as string) + (p["text"] as string);
        else parts.push({ type: "text", text: p["text"] });
      } else if (p["type"] === "image") {
        if (typeof p["data"] !== "string") throw new BakeError("training-picture-format", "a picture given by its address (url, file id) cannot be trained on: give its bytes");
        pictures.push(picture(fromBase64(p["data"] as string)));
        parts.push({ type: "image" });
      } else {
        throw new BakeError("bake-rows", `a student reads text and pictures; the request carries a ${String(p["type"])} part`);
      }
    }
    const r = role === "developer" ? "system" : role;
    messages.push(parts.every((p) => p["type"] === "text") ? { role: r, content: parts.map((p) => p["text"]).join("") } : { role: r, content: parts });
  }
  return { messages, pictures };
}

// ------------------------------------------------------------------ the student

/** A family of contract/students.json, by the student's id; null when none matches. */
export function familyOf(id: string): (typeof STUDENTS)["families"][number] | null {
  const repo = id.split(":")[0]!;
  return STUDENTS.families.find((f) => f.prefixes.some((p) => repo.startsWith(p))) ?? null;
}

/** An effort as a number: a number in [0, 0.99], or one of Inkling's preset names. */
export function effortOf(effort: number | string | undefined): number {
  if (effort === undefined) return 0;
  if (typeof effort === "number") {
    if (!(effort >= 0 && effort <= 0.99)) throw new RangeError(`an effort is a number from 0 to 0.99, not ${effort}`);
    return effort;
  }
  const v = (STUDENTS.efforts as Rec)[effort];
  if (typeof v !== "number") throw new RangeError(`an effort is a number from 0 to 0.99 or one of ${Object.keys(STUDENTS.efforts).filter((k) => k !== "about").join(", ")}, not ${writeData(effort)}`);
  return v;
}

/** The template's keyword arguments (§1): thinking off, the effort when it takes one. */
export function templateKwargs(template: string, effort = 0): Rec {
  const out: Rec = {};
  if (template.includes("enable_thinking")) out["enable_thinking"] = false;
  if (template.includes("reasoning_effort")) out["reasoning_effort"] = effort;
  return out;
}

const dig = (o: unknown, path: string): unknown => path.split(".").reduce<unknown>((x, k) => (x && typeof x === "object" ? (x as Rec)[k] : undefined), o);

/** A family's picture rule with its numbers read from the student's picture configuration file. */
export function pictureRuleOf(family: NonNullable<ReturnType<typeof familyOf>>, config: Rec): PictureRule {
  const p = family.picture as Rec;
  const num = (key: string) => {
    const v = dig(config, p[key] as string);
    if (typeof v !== "number") throw new BakeError("student-unpublished", `the student's ${String(p["file"])} has no ${String(p[key])}`);
    return v;
  };
  if (p["rule"] === "grid") return { rule: "grid", placeholder: p["placeholder"] as string, patch: num("patch") };
  return { rule: "qwen", placeholder: p["placeholder"] as string, patch: num("patch"), merge: num("merge"), min_pixels: num("min_pixels"), max_pixels: num("max_pixels") };
}

interface Tok {
  encode(text: string, o: { add_special_tokens: boolean }): { ids: number[] };
  decode(ids: number[], o: { skip_special_tokens: boolean }): string;
  token_to_id(t: string): number | undefined;
}

interface Tpl {
  render(ctx: Rec): string;
}

async function libraries(): Promise<{ Tokenizer: new (json: Rec, config: Rec) => Tok; Template: new (t: string) => Tpl }> {
  try {
    const [{ Tokenizer }, { Template }] = await Promise.all([import("@huggingface/tokenizers" as string), import("@huggingface/jinja" as string)]);
    return { Tokenizer, Template };
  } catch {
    throw new Error("a student's tokens are made from its published files with @huggingface/tokenizers and @huggingface/jinja: npm install @huggingface/tokenizers @huggingface/jinja");
  }
}

/** The files a student is made from. */
export interface StudentFiles {
  readonly tokenizer: Rec;
  /** tokenizer.json's text as published (its hash is kept); written from `tokenizer` when not given. */
  readonly tokenizerText?: string;
  readonly tokenizerConfig?: Rec;
  readonly template: string;
  readonly picture?: PictureRule | null;
}

/** Options of `Student.load`. */
export interface StudentOptions {
  /** A folder holding the files (tokenizer.json, tokenizer_config.json, chat_template.jinja, the picture configuration) instead of the Hugging Face Hub. */
  dir?: string;
  /** Inkling's thinking effort while trained and called (default 0: answer directly). */
  effort?: number | string;
  /** Where downloaded files are kept (default ~/.cache/functai/students). */
  cacheDir?: string;
  /** A Hugging Face token (default HF_TOKEN). */
  hfToken?: string;
  /** Read only what is cached (default: HF_HUB_OFFLINE). */
  offline?: boolean;
}

/** A student: its tokenizer, template and picture rule, as the contract says to use them. */
export class Student {
  readonly id: string;
  readonly repo: string;
  readonly template: string;
  readonly kwargs: Rec;
  readonly picture: PictureRule | null;
  readonly lead: string;
  readonly end: string;
  readonly stopToken: number;
  readonly endHead: string;
  readonly templateSha256: string;
  readonly tokenizerSha256: string;
  readonly source: "files";
  readonly #tok: Tok;
  readonly #tpl: Tpl;
  readonly #specials: Set<number>;
  readonly #extra: Rec;

  private constructor(id: string, tok: Tok, tpl: Tpl, files: StudentFiles, effort: number) {
    this.id = id;
    this.repo = id.split(":")[0]!;
    this.template = files.template;
    this.kwargs = templateKwargs(files.template, effort);
    this.picture = files.picture ?? null;
    this.#tok = tok;
    this.#tpl = tpl;
    this.#specials = new Set(((files.tokenizer["added_tokens"] ?? []) as Rec[]).filter((t) => t["special"] === true).map((t) => t["id"] as number));
    this.#extra = {};
    const cfg = files.tokenizerConfig ?? {};
    for (const k of ["bos_token", "eos_token", "unk_token", "pad_token"]) {
      const v = cfg[k];
      if (typeof v === "string") this.#extra[k] = v;
      else if (v && typeof v === "object" && typeof (v as Rec)["content"] === "string") this.#extra[k] = (v as Rec)["content"];
    }
    this.source = "files";
    this.templateSha256 = `sha256:${sha256Hex(new TextEncoder().encode(files.template))}`;
    this.tokenizerSha256 = `sha256:${sha256Hex(new TextEncoder().encode(files.tokenizerText ?? writeData(files.tokenizer)))}`;
    const e = this.#ends();
    this.lead = e.lead;
    this.end = e.end;
    this.stopToken = e.stop;
    this.endHead = e.head;
  }

  /** A student from its files given as data (tests, files kept elsewhere). */
  static async fromFiles(id: string, files: StudentFiles, opts: { effort?: number | string } = {}): Promise<Student> {
    const { Tokenizer, Template } = await libraries();
    const tok = new Tokenizer(files.tokenizer, files.tokenizerConfig ?? {});
    return new Student(id, tok, new Template(files.template), files, effortOf(opts.effort));
  }

  /**
   * A student by the service's id (`Qwen/Qwen3.5-4B`,
   * `thinkingmachines/Inkling-2-Small:peft:262144`), from the Hugging Face
   * repository of its name up to the first `:` (or `dir`), files kept in a
   * cache. Refused (`student-unpublished`) when the files are not published.
   */
  static async load(id: string, opts: StudentOptions = {}): Promise<Student> {
    const repo = id.split(":")[0]!;
    const family = familyOf(id);
    const get = (name: string) => fetchFile(repo, name, opts);
    let tokenizerText: string | null;
    try {
      tokenizerText = await get("tokenizer.json");
    } catch (err) {
      throw new BakeError("student-unpublished", `cannot read ${repo}'s tokenizer.json (${(err as Error).message})`);
    }
    if (tokenizerText === null) {
      const why = family && (family as Rec)["unpublished"] ? ((family as Rec)["unpublished"] as Rec)[repo] : undefined;
      throw new BakeError("student-unpublished", `${repo} has no published tokenizer.json${why ? `: ${String(why)}` : ""}; TypeScript makes a student's tokens from its published files`);
    }
    const cfgText = await get("tokenizer_config.json");
    const tokenizerConfig = cfgText ? parseForeign(cfgText) as Rec : {};
    const template = (await get("chat_template.jinja")) ?? (typeof tokenizerConfig["chat_template"] === "string" ? tokenizerConfig["chat_template"] as string : null);
    if (!template) throw new BakeError("student-unpublished", `${repo} has no chat template: a student is a chat (instruct) model`);
    let pictureRule: PictureRule | null = null;
    if (family) {
      const file = (family.picture as Rec)["file"] as string;
      const text = await get(file);
      if (text) pictureRule = pictureRuleOf(family, parseForeign(text) as Rec);
    }
    return Student.fromFiles(id, { tokenizer: parseForeign(tokenizerText) as Rec, tokenizerText, tokenizerConfig, template, picture: pictureRule }, { effort: opts.effort });
  }

  /** Text's tokens, special tokens read as themselves. */
  encode(text: string): number[] {
    return this.#tok.encode(text, { add_special_tokens: false }).ids;
  }

  /** Tokens' text, special tokens kept. */
  decode(ids: readonly number[]): string {
    return ids.length ? this.#tok.decode([...ids], { skip_special_tokens: false }) : "";
  }

  tokenId(text: string): number | undefined {
    return this.#tok.token_to_id(text);
  }

  #render(messages: readonly Rec[], generation: boolean): string {
    try {
      return this.#tpl.render({ ...this.#extra, messages, add_generation_prompt: generation, ...this.kwargs });
    } catch (err) {            // the template's raise_exception (jinja's own), or a template error
      throw new BakeError("bake-rows", `the student's chat template refused the conversation: ${(err as Error).message}`);
    }
  }

  #ends(): { lead: string; end: string; stop: number; head: string } {
    const s = "\u2063R\u2063";
    const probe = [{ role: "user", content: "x" }];
    const p = this.#render(probe, true);
    const f = this.#render([...probe, { role: "assistant", content: s }], false);
    const at = f.indexOf(s);
    const lead = f.startsWith(p) && at >= p.length ? f.slice(p.length, at) : "";
    const end = f.slice(at + s.length);
    let stop: number | undefined;
    for (const t of this.encode(end)) if (this.#specials.has(t)) stop = t;
    if (stop === undefined) {
      const eos = this.#extra["eos_token"];
      stop = typeof eos === "string" ? this.tokenId(eos) : undefined;
    }
    if (stop === undefined) throw new BakeError("student-unpublished", `${this.id}'s template writes nothing that ends a reply, and its tokenizer has no end-of-sequence token`);
    const stopText = this.decode([stop]);
    const head = end.includes(stopText) ? end.slice(0, end.indexOf(stopText)) : "";
    return { lead, end, stop, head };
  }

  #chunks(ids: readonly number[], pictures: readonly Picture[]): [Chunk[], number] {
    const ph = this.picture ? this.tokenId(this.picture.placeholder) : undefined;
    const out: Chunk[] = [];
    let cur: number[] = [];
    let k = 0;
    for (const t of ids) {
      if (ph !== undefined && t === ph) {
        if (k >= pictures.length) throw new BakeError("training-pictures", `the template wrote more pictures than the conversation has (${pictures.length})`);
        if (cur.length) out.push({ text: cur });
        const pic = pictures[k]!;
        out.push({ picture: k, tokens: pictureTokens(this.picture!, pic.width, pic.height) });
        cur = [];
        k++;
      } else {
        cur.push(t);
      }
    }
    if (cur.length) out.push({ text: cur });
    return [out, k];
  }

  /**
   * A conversation's chunks (contract/training.md §3): the prompt, and with
   * `reply` the whole conversation and where the reply starts. Refused
   * (`training-pictures`) when the pictures and the template's placeholders
   * differ in number, or the student has no picture rule.
   */
  render(messages: readonly Rec[], pictures: readonly Picture[] = [], reply: string | null = null): Rendered {
    if (pictures.length && !this.picture) throw new BakeError("training-pictures", `${this.id} has no picture rule (contract/students.json): it is not trained on pictures here`);
    const promptIds = this.encode(this.#render(messages, true));
    const [prompt, used] = this.#chunks(promptIds, pictures);
    if (used !== pictures.length) throw new BakeError("training-pictures", `the conversation has ${pictures.length} pictures and the template shows ${used}`);
    const replyStart = chunksLength(prompt);
    if (reply === null) return { prompt, full: null, replyStart, prefixFallback: false };
    let fullIds = this.encode(this.#render([...messages, { role: "assistant", content: reply }], false));
    let prefixFallback = false;
    if (fullIds.length < promptIds.length || promptIds.some((t, i) => fullIds[i] !== t)) {
      fullIds = [...promptIds, ...this.encode(reply + this.end)];
      prefixFallback = true;
    }
    const [full, used2] = this.#chunks(fullIds, pictures);
    if (used2 !== pictures.length) throw new BakeError("training-pictures", `the conversation has ${pictures.length} pictures and the template shows ${used2}`);
    return { prompt, full, replyStart, prefixFallback };
  }

  /** The reply text in generated tokens (contract/training.md §5). */
  readReply(generated: readonly number[]): string {
    const ids = generated.at(-1) === this.stopToken ? generated.slice(0, -1) : [...generated];
    let text = this.decode(ids);
    if (this.lead) {
      const at = text.lastIndexOf(this.lead);
      if (at >= 0) text = text.slice(at + this.lead.length);
    }
    if (this.endHead) {
      const at = text.indexOf(this.endHead);
      if (at >= 0) text = text.slice(0, at);
    }
    return text;
  }

  /** What `baked.json` keeps of the template (contract/training.md §11). */
  describe(): Rec {
    return {
      sha256: this.templateSha256, kwargs: { ...this.kwargs }, end: this.end, stop_token_ids: [this.stopToken], source: this.source,
      repo: this.repo, tokenizer_sha256: this.tokenizerSha256, lead: this.lead, picture: this.picture ? { ...this.picture } : null,
    };
  }
}

/** A chunk list's length: a picture counts its positions. */
export const chunksLength = (chunks: readonly Chunk[]): number => chunks.reduce((n, c) => n + ("text" in c ? c.text.length : c.tokens), 0);

// ------------------------------------------------------------------ files

function cacheDir(opts: StudentOptions): string {
  const path = builtin("node:path")!;
  const os = builtin("node:os")!;
  const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
  return opts.cacheDir ?? path.join(env["XDG_CACHE_HOME"] ?? path.join(os.homedir(), ".cache"), "functai", "students");
}

/** A file of a student's repository: from `dir`, the cache, or the Hub (kept in the cache); null when the repository has none. */
async function fetchFile(repo: string, name: string, opts: StudentOptions): Promise<string | null> {
  const fs = builtin("node:fs");
  const path = builtin("node:path");
  if (!fs || !path) throw new Error("reading a student's files needs a file system");
  if (opts.dir) {
    const p = path.join(opts.dir, name);
    return fs.existsSync(p) ? fs.readFileSync(p, "utf8") : null;
  }
  const dir = path.join(cacheDir(opts), ...repo.split("/"));
  const cached = path.join(dir, name);
  const missing = cached + ".missing";
  if (fs.existsSync(cached)) return fs.readFileSync(cached, "utf8");
  const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
  if (fs.existsSync(missing)) return null;
  if (opts.offline ?? (env["HF_HUB_OFFLINE"] === "1")) {
    throw new Error(`${repo}/${name} is not in ${dir} and the Hub is not to be asked (HF_HUB_OFFLINE)`);
  }
  const token = opts.hfToken ?? env["HF_TOKEN"];
  const resp = await fetch(`https://huggingface.co/${repo}/resolve/main/${name}`, { headers: token ? { authorization: `Bearer ${token}` } : {}, redirect: "follow" });
  if (resp.status === 404 || resp.status === 401) {
    fs.mkdirSync(dir, { recursive: true });
    if (name !== "tokenizer.json") fs.writeFileSync(missing, "");          // a repository without this file (the answer is kept)
    return null;
  }
  if (!resp.ok) throw new Error(`the Hub answered ${resp.status} for ${repo}/${name}`);
  const text = await resp.text();
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(cached + ".tmp", text);
  fs.renameSync(cached + ".tmp", cached);
  return text;
}
