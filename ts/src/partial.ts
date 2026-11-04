/**
 * The answer so far (contract/streaming.md, *The answer so far*): read from
 * a call's answer text by the answer's shape, while the model writes it.
 *
 * A text answer is its text. A record or a list is the JSON read so far:
 * complete values; a string still being written as its text so far, unless
 * its shape is a choice (`enum`, `const`), which waits for its closing quote
 * ("med" is no value of `["low", "medium", "high"]`); a number, `true`,
 * `false` or `null` once a character after it ends it. Any other answer
 * (a number, a choice, a picture) is nothing (`undefined`) until the call
 * is done. Provisional: the call's result is the typed, checked value.
 */

type Rec = Record<string, unknown>;

/** What a value of type `T` may be while it is written: members and items may be missing; text and choices are whole or absent. */
export type SoFar<T> =
  T extends string | number | boolean | null | undefined ? T :
  T extends readonly (infer U)[] ? SoFar<U>[] :
  T extends object ? { [K in keyof T]?: SoFar<T[K]> } : T;

const SPACE = " \t\n\r";                        // JSON's white space
const NOTHING: unique symbol = Symbol("nothing");
type Got = unknown | typeof NOTHING;
const HEX = /^[0-9a-fA-F]*$/;
const ESCAPES: Record<string, string> = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };
const NUMBER = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$/;

class NotJSON extends Error {}

const isRec = (x: unknown): x is Rec => typeof x === "object" && x !== null && !Array.isArray(x);
const own = (o: Rec, k: string): unknown => (Object.hasOwn(o, k) ? o[k] : undefined);

/** A shape as the reader follows it: `$ref` followed; an `anyOf` of one shape and null read as that shape; anything else it cannot follow is `{}`. */
function resolve(shape: unknown, root: unknown): Rec {
  const seen = new Set<string>();
  while (isRec(shape) && typeof shape["$ref"] === "string") {
    const ref = shape["$ref"];
    if (seen.has(ref) || !ref.startsWith("#/$defs/") || !isRec(root)) return {};
    seen.add(ref);
    const defs = own(root, "$defs");
    shape = isRec(defs) ? own(defs, ref.slice("#/$defs/".length)) ?? {} : {};
  }
  if (!isRec(shape)) return {};
  const options = shape["anyOf"];
  if (Array.isArray(options)) {
    const rest = options.filter((o) => !(isRec(o) && Object.keys(o).length === 1 && o["type"] === "null"));
    return rest.length === 1 && options.length === 2 ? resolve(rest[0], root) : {};
  }
  return shape;
}

const isChoice = (shape: Rec): boolean => Object.hasOwn(shape, "enum") || Object.hasOwn(shape, "const");

function memberShape(shape: Rec, key: string, root: unknown): Rec {
  const props = shape["properties"];
  if (isRec(props) && Object.hasOwn(props, key)) return resolve(props[key], root);
  const extra = shape["additionalProperties"];
  return isRec(extra) ? resolve(extra, root) : {};
}

function itemShape(shape: Rec, i: number, root: unknown): Rec {
  const prefix = shape["prefixItems"];
  if (Array.isArray(prefix) && i < prefix.length) return resolve(prefix[i], root);
  const items = shape["items"];
  return isRec(items) ? resolve(items, root) : {};
}

/** [text so far, index after, closed] for the string whose quote is s[i]. */
function readString(s: string, i: number): [string, number, boolean] {
  let out = "";
  i += 1;
  while (i < s.length) {
    const c = s[i]!;
    if (c === '"') return [out, i + 1, true];
    if (c !== "\\") {
      out += c;
      i += 1;
      continue;
    }
    if (i + 1 >= s.length) break;                 // an escape still being written
    const e = s[i + 1]!;
    if (Object.hasOwn(ESCAPES, e)) {
      out += ESCAPES[e];
      i += 2;
      continue;
    }
    if (e !== "u") throw new NotJSON("bad escape");
    const digits = s.slice(i + 2, i + 6);
    if (!HEX.test(digits)) throw new NotJSON("bad escape");
    if (digits.length < 4) break;
    const code = parseInt(digits, 16);
    if (code >= 0xd800 && code < 0xdc00) {        // a pair: its second half first
      const rest = s.slice(i + 6, i + 12);
      if (rest.length < 6) {
        if ((rest && !"\\u".startsWith(rest.slice(0, 2))) || !HEX.test(rest.slice(2))) throw new NotJSON("bad escape");
        break;
      }
      if (rest.slice(0, 2) !== "\\u" || !HEX.test(rest.slice(2))) throw new NotJSON("bad escape");
      const low = parseInt(rest.slice(2), 16);
      if (low < 0xdc00 || low >= 0xe000) throw new NotJSON("bad escape");
      out += String.fromCharCode(code, low);
      i += 12;
      continue;
    }
    out += String.fromCharCode(code);
    i += 6;
  }
  return [out, s.length, false];
}

function skip(s: string, i: number): number {
  while (i < s.length && SPACE.includes(s[i]!)) i += 1;
  return i;
}

/** [value or NOTHING, index after, complete]. */
function readValue(s: string, i: number, shape: Rec, root: unknown): [Got, number, boolean] {
  i = skip(s, i);
  if (i >= s.length) return [NOTHING, i, false];
  const c = s[i]!;
  if (c === '"') {
    const [text, after, closed] = readString(s, i);
    if (!closed && isChoice(shape)) return [NOTHING, after, false];   // "med" is no value of a choice
    return [text, after, closed];
  }
  if (c === "{") {
    const obj: Rec = {};
    i += 1;
    for (;;) {
      i = skip(s, i);
      if (i >= s.length) return [obj, i, false];
      if (s[i] === "}") return [obj, i + 1, true];
      if (s[i] === ",") {
        i += 1;
        continue;
      }
      if (s[i] !== '"') throw new NotJSON("a key is a string");
      const [key, afterKey, closed] = readString(s, i);
      i = afterKey;
      if (!closed) return [obj, i, false];
      i = skip(s, i);
      if (i >= s.length) return [obj, i, false];
      if (s[i] !== ":") throw new NotJSON("expected :");
      const [value, after, done] = readValue(s, i + 1, memberShape(shape, key, root), root);
      i = after;
      if (value !== NOTHING) Object.defineProperty(obj, key, { value, enumerable: true, writable: true, configurable: true });
      if (!done) return [obj, i, false];
    }
  }
  if (c === "[") {
    const items: unknown[] = [];
    i += 1;
    for (;;) {
      i = skip(s, i);
      if (i >= s.length) return [items, i, false];
      if (s[i] === "]") return [items, i + 1, true];
      if (s[i] === ",") {
        i += 1;
        continue;
      }
      const [value, after, done] = readValue(s, i, itemShape(shape, items.length, root), root);
      i = after;
      if (value !== NOTHING) items.push(value);
      if (!done) return [items, i, false];
    }
  }
  let j = i;
  while (j < s.length && !(",]}" + SPACE).includes(s[j]!)) j += 1;
  if (j >= s.length) return [NOTHING, j, false];   // 1 may become 12
  const word = s.slice(i, j);
  if (word === "true") return [true, j, true];
  if (word === "false") return [false, j, true];
  if (word === "null") return [null, j, true];
  if (!NUMBER.test(word)) throw new NotJSON(`not JSON: ${word}`);
  return [Number(word), j, true];
}

function kind(shape: unknown): "text" | "json" | "none" {
  const s = resolve(shape, shape);
  const t = s["type"];
  const types = new Set((Array.isArray(t) ? t : [t]).filter((x) => x !== "null"));
  if (types.size === 1 && types.has("string") && !isChoice(s)) return "text";
  if (types.size > 0 && [...types].every((x) => x === "object" || x === "array")) return "json";
  return "none";
}

/**
 * The JSON a text says so far, read by `shape` when given (a choice waits
 * for its closing quote; a record starts with `{`, a list with `[`); a
 * fenced block is read after its first line break. `undefined` when the text
 * is not the start of what the shape gives.
 */
export function partialJson(text: string, shape?: unknown): unknown {
  const trim = (x: string) => x.replace(/^[ \t\n\r]+|[ \t\n\r]+$/g, "");
  let s = trim(text);
  if (s.startsWith("```")) {                       // a fenced block: read after its first line break
    const nl = s.indexOf("\n");
    s = nl === -1 ? "" : s.slice(nl + 1);
    const end = s.replace(/[ \t\n\r]+$/, "");
    if (end.endsWith("```")) s = end.slice(0, -3);
    s = trim(s);
  }
  const top = shape === undefined ? {} : resolve(shape, shape);
  const t = top["type"];
  let opens = (Array.isArray(t) ? t : [t]).map((x) => (x === "object" ? "{" : x === "array" ? "[" : "")).join("");
  if (!opens) opens = "[{";
  if (!s || !opens.includes(s[0]!)) return undefined;
  try {
    const [value] = readValue(s, 0, top, shape);
    return value === NOTHING ? undefined : value;
  } catch (err) {
    if (err instanceof NotJSON) return undefined;
    throw err;
  }
}

/**
 * The answer so far, read from its text by the answer's shape (its
 * interface's last output): the text for a text answer, the JSON read so far
 * for a record or a list, `undefined` for any other answer and when the text
 * is not the start of what the shape gives.
 */
export function answerSoFar(text: string, shape: unknown): unknown {
  const k = kind(shape);
  if (k === "text") return text;
  if (k === "none") return undefined;
  return partialJson(text, shape);
}
