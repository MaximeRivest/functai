/**
 * Shapes: the JSON Schema of each input and output (contract/functions.md,
 * "A definition"). Write them with `t` (no dependency), with zod 4, or as
 * plain JSON Schema. The same data type has the same shape in every
 * language, so zod's JSON Schema is read the way Python writes the type.
 */

import * as lmcc from "lmcc";
import { InterfaceError, type InterfaceNote } from "./interface.ts";
import { carriesFiles, FILES } from "./files-binding.ts";
import type { AudioInput, BinaryInput, DocumentInput, ImageInput, VideoInput } from "./media.ts";
import { oneLine, withChoices } from "./text.ts";
import { copyData, entriesOf, setOwn } from "./values.ts";

type JsonObject = Record<string, unknown>;

/** A JSON Schema carrying the static type of its values (erased at run time). */
export type Shape<T = unknown> = lmcc.TypedShape<T>;

/** A zod 4 schema, as far as this package needs one (no dependency on zod). */
export interface ZodLike<T = unknown> {
  readonly _zod: { readonly output: T };
  toJSONSchema(params?: { io?: "input" | "output" }): JsonObject;
}

/**
 * A schema of any library that implements Standard JSON Schema
 * (standardschema.dev): zod 4.2+, valibot, arktype, ... Its output type is the
 * field's type; its JSON Schema, the field's shape.
 */
export interface StandardSchemaLike<T = unknown, In = unknown> {
  readonly "~standard": {
    readonly vendor: string;
    readonly types?: { readonly input: In; readonly output: T };
    readonly validate: (value: unknown) => StandardResult | Promise<StandardResult>;
    readonly jsonSchema?: { input(options: { target: string }): JsonObject };
  };
}

/** What a Standard Schema's `validate` gives: the parsed value, or the issues. */
export type StandardResult = { readonly value: unknown; readonly issues?: undefined }
  | { readonly issues: readonly { readonly message: string; readonly path?: readonly unknown[] }[] };

/**
 * A field with more said about it: its words (`desc`), whether a caller may
 * leave it out (`optional`: then it takes its shape's `default`, or, for a
 * module, stays out), and whether its values may have no JSON form
 * (`opaque`, modules only).
 */
export interface FieldWith<T = unknown> {
  readonly shape: Shape<T> | ZodLike<T> | StandardSchemaLike<T>;
  readonly desc?: string;
  readonly optional?: boolean;
  readonly opaque?: boolean;
}

/** What a field is written as: a shape, a zod or other Standard Schema, or `{ shape, desc?, optional?, opaque? }`. */
export type FieldSpec<T = unknown> = Shape<T> | ZodLike<T> | StandardSchemaLike<T> | FieldWith<T>;

/** What a caller passes for an input field: a Standard Schema's input type (before defaults and transforms), else the value type. */
export type InputValueOf<S> =
  S extends { readonly shape: infer X; readonly optional: true } ? Exclude<InputValueOf<X>, undefined> :
  S extends { readonly shape: infer X } ? InputValueOf<X> :
  S extends { readonly "~standard": { readonly types?: { readonly input: infer T } } } ? Exclude<T, undefined> :
  ValueOf<S>;

/**
 * May a caller leave this input out? A Standard Schema: when it accepts
 * `undefined` (zod's `.optional()`, `.default(...)`; not `.nullable()`, which
 * must be given). A shape: when it allows null (`t.optional(...)`).
 */
export type IsOptional<S> =
  S extends { readonly optional: true } ? true :
  S extends { readonly optional: false } ? false :
  S extends { readonly shape: infer X } ? IsOptional<X> :
  S extends { readonly "~standard": { readonly types?: { readonly input: infer T } } } ? (undefined extends T ? true : false) :
  S extends { readonly default: unknown } ? true :
  S extends lmcc.TypedShape<infer T> ? (unknown extends T ? false : null extends T ? true : false) : false;

/** The value type of a field spec. */
export type ValueOf<S> =
  S extends { readonly "~standard": { readonly types?: { readonly output: infer T } } } ? T :
  S extends ZodLike<infer T> ? T :
  S extends { readonly shape: infer X; readonly optional: true } ? ValueOf<X> | undefined :
  S extends { readonly shape: infer X } ? ValueOf<X> :
  S extends lmcc.TypedShape<infer T> ? (unknown extends T ? unknown : T) : unknown;

const OPAQUE = Symbol.for("functai.opaque");

/** What each described answer of a `t.choice` means, in the choice's order (kept beside the shape, never in it). */
const MEANINGS = Symbol.for("functai.meanings");
type Meanings = readonly (readonly [string, string])[];

/** The described answers a shape carries (`t.choice`, and what wraps one: `t.optional`, `t.list`, `t.withDefault`). */
export function meaningsOf(shape: unknown): Meanings {
  if (typeof shape !== "object" || shape === null) return [];
  return ((shape as Record<symbol, unknown>)[MEANINGS] as Meanings | undefined) ?? [];
}

/** `to`, carrying `from`'s described answers, and its being files (refused anywhere but a whole input). */
function carry<S>(from: unknown, to: S): S {
  const m = meaningsOf(from);
  if (m.length) Object.defineProperty(to, MEANINGS, { value: m });
  if (carriesFiles(from)) Object.defineProperty(to, FILES, { value: true });
  return to;
}

/** Files are a whole input (functions.md): never a record's member or a map's value. */
function noFiles(builder: string, shapes: readonly unknown[]): void {
  if (shapes.some(carriesFiles)) {
    throw new TypeError(`t.${builder}: files (artifacts()) are a whole input of their own, not a member of a record or a map`);
  }
}

/** A media field's shape (lmcc's `{"media": kind}`), with extra keys that add to it (`{ description }`). */
const media = <V>(kind: string, extra: Extra | undefined): Shape<V> =>
  ({ media: kind, ...extraOf(extra, kind, ["media", "type"]) }) as unknown as Shape<V>;

/** The keys of a field written out (`{ shape, desc?, optional?, opaque? }`, what `describe` gives): FunctAI's words, never JSON Schema's. */
const FIELD_KEYS = ["shape", "desc", "optional", "opaque"] as const;

/** A shape given inside another (a member of `t.object`, a list's items, …): a shape, or `describe(shape, words)`. */
export type Member = Shape<unknown> | { readonly shape: Shape<unknown>; readonly desc?: string };

function shown(x: unknown): string {
  if (typeof x === "function") return "a function";
  try {
    return JSON.stringify(x) ?? String(x);
  } catch {
    return typeof x;
  }
}

/** The first key of a written-out field that is not one of its keys, with what was likely meant. */
function strangeFieldKey(spec: object): string | null {
  const k = Object.keys(spec).find((key) => !(FIELD_KEYS as readonly string[]).includes(key));
  if (k === undefined) return null;
  const meant = k === "description" ? ": its words are desc ({ shape, desc }), or the shape's own description (t.string({ description }))" : "";
  return `a field written out is { shape, desc?, optional?, opaque? }, and ${JSON.stringify(k)} is not one of them${meant}`;
}

/**
 * A shape given inside another, as JSON Schema. `describe(shape, words)` is
 * the shape with `description: words` (what R's `described()` and Python's
 * `Field(description=…)` make of a member). What is not a shape is refused,
 * naming where it is: before this, `t.object({ goal: describe(t.string(), "…") })`
 * sent the model `{"shape": …, "desc": …}` as the member's schema.
 */
function member(spec: unknown, builder: string, where: string): Shape<never> {   // `never`: fits any builder's value type; the caller's signature gives the real one
  const fail = (why: string): never => {
    throw new TypeError(`t.${builder}: ${where}${why.startsWith(":") ? "" : " "}${why}`);
  };
  if (isZod(spec) || isStandard(spec)) {
    const vendor = isStandard(spec) ? spec["~standard"].vendor : "zod";
    return fail(`is a ${vendor} schema: inside t.${builder}, give t's shapes (t.string(), describe(t.string(), "words"), …), or write the whole value in ${vendor} (z.object({ … })). Mixed, its checks and transforms would not run`);
  }
  if (isWrapper(spec)) {
    const strange = strangeFieldKey(spec);
    if (strange) return fail(`: ${strange}`);
    if (spec.optional !== undefined) {
      return fail("says optional: whether a caller may leave a value out is said of a whole input, not of a member inside a shape. A member that may be null is t.optional(shape)");
    }
    if (spec.opaque !== undefined) return fail("says opaque: a module's field may hold values with no JSON form, a member inside a shape may not");
    if (spec.desc !== undefined && typeof spec.desc !== "string") return fail(`has desc ${shown(spec.desc)}: its words are text`);
    const inner = member(spec.shape, builder, where);
    return spec.desc === undefined ? inner : carry(inner, { ...(inner as JsonObject), description: spec.desc } as unknown as Shape<never>);
  }
  if (typeof spec !== "object" || spec === null || Array.isArray(spec)) {
    return fail(`is ${shown(spec)}, not a shape: t.string(), t.list(…), describe(t.string(), "words"), or JSON Schema ({ type: "string" })`);
  }
  return spec as Shape<never>;
}

/** Each member of `t.object` or `t.record`, as JSON Schema (what `member` makes of it), in order. */
function members(properties: Record<string, unknown>, builder: string): Record<string, Shape<never>> {
  // describe(...) given whole: its desc is text, which no member's shape is (a member may be named shape or desc)
  const whole = typeof properties === "object" && properties !== null && Object.hasOwn(properties, "shape") && typeof properties["desc"] === "string";
  if (typeof properties !== "object" || properties === null || Array.isArray(properties) || whole) {
    throw new TypeError(`t.${builder}(properties): properties is an object, from each member's name to its shape: t.${builder}({ id: t.integer() })${whole ? `; words about the whole object are describe(t.${builder}({ … }), "words")` : ""}`);
  }
  const out: Record<string, Shape<never>> = {};
  for (const [k, v] of entriesOf(properties)) setOwn(out, k, member(v, builder, `member ${JSON.stringify(k)}`));
  return out;
}

/** A property's shape with its choice's meanings in its `description`, where the model reads them (a record's schema). */
function described(shape: unknown): unknown {
  const m = meaningsOf(shape);
  if (!m.length) return shape;
  const words = typeof (shape as JsonObject)["description"] === "string" ? (shape as JsonObject)["description"] as string : null;
  return { ...(shape as JsonObject), description: withChoices(words, m) };
}


/** Any JSON value. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/** Extra JSON Schema keys a builder takes (`{ description }`, `{ minimum: 0 }`, `{ default: "kind" }`). */
type Extra = Record<string, unknown>;
/** A shape of values `T`; with a `default`, one a caller may leave out (its type says so). */
type Built<E, T> = E extends { readonly default: unknown } ? Shape<T> & { readonly default: T } : Shape<T>;

/**
 * A builder's extra keys, checked: they add to the shape (words, bounds, a
 * default) and never replace what the builder makes (`type`, `items`,
 * `properties`, …), which its value type describes.
 */
function extraOf<E>(extra: E | undefined, builder: string, structural: readonly string[]): E {
  const got = (extra ?? {}) as E & object;
  if (typeof got !== "object" || got === null || Array.isArray(got)) throw new TypeError(`t.${builder}(…, extra): extra is an object of JSON Schema keys`);
  const bad = structural.filter((k) => Object.hasOwn(got, k));
  if (bad.length) throw new TypeError(`t.${builder}: ${bad.join(", ")} is what t.${builder} makes; extra keys add to it (description, bounds, default), never replace it`);
  // a field's own words, given where JSON Schema keys go: the model would read an unknown key, and nothing would do what it says
  const field = FIELD_KEYS.find((k) => Object.hasOwn(got, k));
  if (field === "desc") throw new TypeError(`t.${builder}: desc is not a JSON Schema key; a shape's words are { description: "…" }`);
  if (field === "optional") {
    throw new TypeError(`t.${builder}: optional is not a JSON Schema key; a value that may be null is t.optional(shape), an input a caller may leave out has a default ({ default: … }) or is written { shape, optional: true }`);
  }
  if (field !== undefined) throw new TypeError(`t.${builder}: ${field} is not a JSON Schema key but a field's ({ shape, desc?, optional?, opaque? })`);
  return got;
}

const SCALAR = ["type"];
const LIST = ["type", "items"];
const OBJECT = ["type", "properties", "required", "additionalProperties"];
const RECORD = ["type", "properties", "additionalProperties"];

/** Builders for shapes (lmcc's, plus a map, `optional`, `json` and `opaque`). Each takes extra JSON Schema keys, which add to the shape it makes. */
export const t = {
  ...lmcc.t,
  /** Text: `t.string()`; `t.string({ default: "kind" })` may be left out (and is typed so). */
  string: <const E extends Extra & { readonly default?: string } = {}>(extra?: E): Built<E, string> =>
    lmcc.t.string(extraOf(extra, "string", SCALAR) as never) as never,
  /** An integer; `t.integer({ default: 3 })` may be left out. */
  integer: <const E extends Extra & { readonly default?: number } = {}>(extra?: E): Built<E, number> =>
    lmcc.t.integer(extraOf(extra, "integer", SCALAR) as never) as never,
  /** A number; `t.number({ default: 0.5 })` may be left out. */
  number: <const E extends Extra & { readonly default?: number } = {}>(extra?: E): Built<E, number> =>
    lmcc.t.number(extraOf(extra, "number", SCALAR) as never) as never,
  /** Yes or no; `t.boolean({ default: false })` may be left out. */
  boolean: <const E extends Extra & { readonly default?: boolean } = {}>(extra?: E): Built<E, boolean> =>
    lmcc.t.boolean(extraOf(extra, "boolean", SCALAR) as never) as never,
  /**
   * A list: `t.list(t.string(), { description: "one per line" })`. Words
   * about each item: `t.list(describe(t.string(), "a tag"))`.
   */
  list: <S extends Member>(items: S, extra: Extra = {}): Shape<ValueOf<S>[]> => {
    const shape = member(items, "list", "its items");
    return carry(shape, { ...lmcc.t.list(shape), ...extraOf(extra, "list", LIST) }) as Shape<ValueOf<S>[]>;
  },
  /**
   * An object with these properties, all required: `t.object({ id: t.integer() }, { description })`.
   * Words about a member: `t.object({ goal: describe(t.string(), "the one thing they came to do") })`,
   * the same as `t.string({ description: "…" })`.
   */
  object: <P extends Record<string, Member>>(properties: P, extra: Extra = {}): Shape<{ [K in keyof P]: ValueOf<P[K]> }> => {
    const shapes = members(properties, "object");
    noFiles("object", Object.values(shapes));
    const described_: Record<string, Shape<never>> = {};
    for (const [k, v] of entriesOf(shapes)) setOwn(described_, k, described(v) as Shape<never>);
    return { ...lmcc.t.object(described_), ...extraOf(extra, "object", OBJECT) } as Shape<{ [K in keyof P]: ValueOf<P[K]> }>;
  },
  /** A map from text keys to values: `t.record(t.integer())`, or `t.record(describe(t.integer(), "days"))`. */
  record: <S extends Member>(values: S, extra: Extra = {}): Shape<Record<string, ValueOf<S>>> => {
    const shape = member(values, "record", "its values");
    noFiles("record", [shape]);
    return { type: "object", additionalProperties: described(shape), ...extraOf(extra, "record", RECORD) } as Shape<Record<string, ValueOf<S>>>;
  },
  /** The shape or null; a caller may leave it out, and it is then null. */
  optional: <S extends Member>(shape: S): Shape<ValueOf<S> | null> => {
    const inner = member(shape, "optional", "its shape");
    return carry(inner, lmcc.t.nullable(inner)) as Shape<ValueOf<S> | null>;
  },
  /** The shape or null (lmcc's nullable form), as `t.optional`. */
  nullable: <S extends Member>(shape: S): Shape<ValueOf<S> | null> => {
    const inner = member(shape, "nullable", "its shape");
    return carry(inner, lmcc.t.nullable(inner)) as Shape<ValueOf<S> | null>;
  },
  /** The shape, with the value an input takes when a caller leaves it out: `t.withDefault(t.string(), "kind")`. */
  withDefault: <S extends Member>(shape: S, value: ValueOf<S>): Shape<ValueOf<S>> & { readonly default: ValueOf<S> } => {
    const inner = member(shape, "withDefault", "its shape");
    return carry(inner, { ...inner, default: value }) as Shape<ValueOf<S>> & { readonly default: ValueOf<S> };
  },
  /**
   * One of these answers; with what some of them mean, which the model reads:
   * `t.choice(["low", "high"], { high: "the customer is losing money or time now" })`.
   * The shape is `t.enum`'s, in the list's order; the meanings go into the
   * field's words, written as every language writes them
   * (contract/functions.md, "A choice's answers, described one by one"), or,
   * inside `t.object`, into the property's description.
   */
  choice: <const A extends readonly string[]>(answers: A, meanings: { readonly [K in A[number]]?: string } = {}): Shape<A[number]> => {
    if (!Array.isArray(answers) || !answers.length || answers.some((a) => typeof a !== "string")) {
      throw new TypeError('t.choice(answers, meanings?): answers is a list of strings, t.choice(["yes", "no"])');
    }
    if (typeof meanings !== "object" || meanings === null || Array.isArray(meanings)) {
      throw new TypeError("t.choice(answers, meanings): meanings is an object, from an answer to what it means");
    }
    const twice = answers.find((a, i) => answers.indexOf(a) !== i);
    if (twice !== undefined) throw new TypeError(`t.choice has ${JSON.stringify(twice)} twice`);
    for (const [k, v] of entriesOf(meanings as Record<string, unknown>)) {
      if (!answers.includes(k)) throw new TypeError(`t.choice: ${JSON.stringify(k)} is given a meaning, but it is not one of the answers ${JSON.stringify(answers)}`);
      if (typeof v !== "string") throw new TypeError(`t.choice: what ${JSON.stringify(k)} means is text, not ${JSON.stringify(v)}`);
    }
    const described: [string, string][] = [];
    for (const a of answers) {
      const m = Object.hasOwn(meanings, a) ? oneLine((meanings as Record<string, string>)[a]!) : "";
      if (m) described.push([a, m]);
    }
    const shape = lmcc.t.enum(...answers);
    if (described.length) Object.defineProperty(shape, MEANINGS, { value: Object.freeze(described) });
    return shape as Shape<A[number]>;
  },
  /**
   * A picture: lm15's `image({ path: "shelf.jpg" })` (also `{ url }`,
   * `{ data }`, `{ fileId }`), or the same as data (`{ media_type, data }`).
   * The model is sent the picture itself; a file is read when the call is
   * bound, so a changed file is a new request. `t.list(t.image())` sends
   * each picture, in order; `t.optional(t.image())` may be null.
   */
  image: (extra?: Extra): Shape<ImageInput> => media<ImageInput>("image", extra),
  /** A recording (lm15's `audio(...)`), sent to the model as sound. */
  audio: (extra?: Extra): Shape<AudioInput> => media<AudioInput>("audio", extra),
  /** A video (lm15's `video(...)`), sent to the model as video. */
  video: (extra?: Extra): Shape<VideoInput> => media<VideoInput>("video", extra),
  /** A document given whole (lm15's `document(...)`, a PDF), which providers that read PDFs read themselves. For any file read into text and page pictures, see `artifacts()` in `functai/files`. */
  document: (extra?: Extra): Shape<DocumentInput> => media<DocumentInput>("document", extra),
  /** Bytes of any media type (lm15's `binary(...)`), for a provider that takes them. */
  binary: (extra?: Extra): Shape<BinaryInput> => media<BinaryInput>("binary", extra),
  /** Any JSON value (shape `{}`), with words about it if you like: `t.json({ description: "each input by name" })`. */
  json: (extra: Extra = {}): Shape<Json> => ({ ...extraOf(extra, "json", []) }) as Shape<Json>,
  /**
   * A module's field whose values may have no JSON form (a class instance, a
   * buffer, a data frame): never checked, written in the log as a
   * description. It cannot cross a boundary that needs data (a server, a
   * conversation store). `t.opaque<Buffer>()`.
   */
  opaque: <T = unknown>(): Shape<T> => Object.defineProperty({}, OPAQUE, { value: true }) as Shape<T>,
};

const isWrapper = (spec: unknown): spec is FieldWith => typeof spec === "object" && spec !== null && "shape" in spec
  && !("type" in spec) && !isZod(spec) && !isStandard(spec);

/** Whether a field spec was declared opaque (`t.opaque()`, or `{ shape: {}, opaque: true }`). */
export function isOpaque(spec: unknown): boolean {
  if (isWrapper(spec) && spec.opaque === true) return true;
  const raw = isWrapper(spec) ? spec.shape : spec;
  return typeof raw === "object" && raw !== null && (raw as Record<symbol, unknown>)[OPAQUE] === true;
}

/** Whether a field spec says itself whether it may be left out (`{ shape, optional }`): true, false, or undefined (inferred). */
export function saysOptional(spec: unknown): boolean | undefined {
  return isWrapper(spec) && typeof spec.optional === "boolean" ? spec.optional : undefined;
}

/** A field with words about it: `describe(t.string(), "the customer's own words")`. */
export function describe<S extends Shape | ZodLike | StandardSchemaLike>(shape: S, desc: string): { shape: S; desc: string } {
  return { shape, desc };
}

function isZod(x: unknown): x is ZodLike {
  return typeof x === "object" && x !== null && "_zod" in x && typeof (x as ZodLike).toJSONSchema === "function";
}

function isStandard(x: unknown): x is StandardSchemaLike {
  return (typeof x === "object" || typeof x === "function") && x !== null && "~standard" in x
    && typeof (x as StandardSchemaLike)["~standard"]?.jsonSchema?.input === "function";
}

/** The schema object of a field spec, when it is a Standard Schema (to validate values with). */
export function standardOf(spec: FieldSpec): StandardSchemaLike | null {
  const raw = typeof spec === "object" && spec !== null && "shape" in spec && !isZod(spec) && !isStandard(spec)
    ? (spec as { shape: unknown }).shape : spec;
  return isStandard(raw) ? raw : null;
}

/** Does a shape allow null? */
export function allowsNull(shape: JsonObject): boolean {
  const options = (shape["anyOf"] ?? shape["oneOf"]) as JsonObject[] | undefined;
  if (Array.isArray(options)) return options.some(allowsNull);
  const type = shape["type"];
  return type === "null" || (Array.isArray(type) && type.includes("null"));
}

const SAFE = Number.MAX_SAFE_INTEGER;

/** A schema library's JSON Schema as Python writes the same type: no `$schema`, no safe-integer bounds, nullable as `anyOf`. */
function fromZod(schema: JsonObject): JsonObject {
  const walk = (s: unknown): unknown => {
    if (Array.isArray(s)) return s.map(walk);
    if (typeof s !== "object" || s === null) return s;
    const o: JsonObject = {};
    for (const [k, v] of entriesOf(s as JsonObject)) {
      if (k === "$schema") continue;
      setOwn(o, k, walk(v));                          // a property may be named __proto__: data, not the prototype
    }
    if (o["type"] === "integer" && o["minimum"] === -SAFE && o["maximum"] === SAFE) {
      delete o["minimum"];
      delete o["maximum"];
    }
    if (o["type"] === "object" && Object.hasOwn(o, "additionalProperties") && typeof o["additionalProperties"] === "object"
      && JSON.stringify(o["propertyNames"]) === '{"type":"string"}') {
      delete o["propertyNames"];
    }
    if (Object.hasOwn(o, "const") && !Object.hasOwn(o, "enum")) {
      o["enum"] = [o["const"]];
      delete o["const"];
    }
    if (Array.isArray(o["type"])) {
      const { type, ...rest } = o;
      const types = type as string[];
      const plain = types.filter((x) => x !== "null");
      const options: unknown[] = plain.map((x) => ({ ...rest, type: x }));
      if (types.includes("null")) options.push({ type: "null" });
      return options.length === 1 ? options[0] : { anyOf: options };
    }
    return o;
  };
  return walk(schema) as JsonObject;
}

/**
 * A field spec as `{shape, desc}`: a top-level `description` in the shape
 * becomes the desc. `field`: the interface field it declares (then a spec
 * that is not a shape refuses `interface-malformed`, naming it).
 */
export function readField(spec: FieldSpec, where: string, field?: string): { shape: JsonObject; desc: string | null; files: boolean } {
  let desc: string | null = null;
  let raw: unknown = spec;
  const refuse = (why: string): never => {
    if (field !== undefined) throw new InterfaceError("interface-malformed", field, `${where}: ${why}`);
    throw new TypeError(`${where}: ${why}`);
  };
  if (isWrapper(spec)) {
    const strange = strangeFieldKey(spec);
    if (strange) refuse(strange);
    if (spec.desc !== undefined && spec.desc !== null && typeof spec.desc !== "string") refuse("desc is text");
    desc = spec.desc ?? null;
    raw = spec.shape;
  }
  const meanings = meaningsOf(raw);
  let shape: JsonObject;
  if (isStandard(raw)) shape = fromZod(raw["~standard"].jsonSchema!.input({ target: "draft-2020-12" }));
  else if (isZod(raw)) shape = fromZod(raw.toJSONSchema({ io: "input" }));
  else if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) {
    try {
      // a default given as a function (computed at each call: calls.md, "Versions", "Defaults") is kept as it is
      const computed = typeof (raw as Record<string, unknown>)["default"] === "function" ? (raw as Record<string, unknown>)["default"] : undefined;
      if (computed !== undefined) {
        const { default: _d, ...data } = raw as Record<string, unknown>;
        shape = { ...(copyData(data) as JsonObject), default: computed as never };
      } else shape = copyData(raw) as JsonObject;
    } catch {
      return refuse("a shape is JSON Schema: plain data");
    }
  } else {
    let shown: string;
    try {
      shown = JSON.stringify(raw) ?? String(raw);
    } catch {
      shown = typeof raw;
    }
    return refuse(`expected a shape (t.string(), a Standard Schema such as zod, valibot or arktype, or JSON Schema), not ${shown}`);
  }
  const stray = strayField(shape, "");
  if (stray !== null) {
    return refuse(`${stray || "its shape"} is a field written out ({ shape, desc }, what describe() gives), not JSON Schema. Inside a shape, build it with t.object, t.list, t.record or t.optional, which read describe(), or give the schema its words: { type: "string", description: "…" }`);
  }
  if (typeof shape["description"] === "string") {
    desc = desc ?? (shape["description"] as string);
    delete shape["description"];
  }
  return { shape, desc: withChoices(desc, meanings), files: carriesFiles(raw) };
}

/** JSON Schema's keywords whose value is a schema, a list of schemas, or a map of them (draft 2020-12, and draft 7's `items` list, `additionalItems`, `definitions`). */
const ONE_SCHEMA = ["items", "additionalItems", "additionalProperties", "contains", "not", "if", "then", "else", "propertyNames", "unevaluatedItems", "unevaluatedProperties"];
const SCHEMA_LISTS = ["anyOf", "oneOf", "allOf", "prefixItems", "items"];
const SCHEMA_MAPS = ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"];

/**
 * Where in a JSON Schema a field written out (`{ shape, desc }`) stands in
 * for a schema (`properties.goal`; `""` for the schema itself), or null.
 * `shape` is no JSON Schema keyword, so a schema that has it is one: the
 * model would be sent FunctAI's words as the schema, and any value would
 * fit it. Values (`default`, `enum`, `const`, `examples`) are not walked.
 */
function strayField(schema: unknown, path: string): string | null {
  if (typeof schema !== "object" || schema === null || Array.isArray(schema)) return null;
  const s = schema as JsonObject;
  if (Object.hasOwn(s, "shape")) return path;
  const at = (key: string) => (path ? `${path}.${key}` : key);
  for (const k of ONE_SCHEMA) {
    if (Object.hasOwn(s, k) && !Array.isArray(s[k])) {
      const found = strayField(s[k], at(k));
      if (found !== null) return found;
    }
  }
  for (const k of SCHEMA_LISTS) {
    if (Object.hasOwn(s, k) && Array.isArray(s[k])) {
      const list = s[k] as unknown[];
      for (let i = 0; i < list.length; i++) {
        const found = strayField(list[i], `${at(k)}[${i}]`);
        if (found !== null) return found;
      }
    }
  }
  for (const k of SCHEMA_MAPS) {
    const map = s[k];
    if (Object.hasOwn(s, k) && typeof map === "object" && map !== null && !Array.isArray(map)) {
      for (const [name, sub] of entriesOf(map as JsonObject)) {
        const found = strayField(sub, `${at(k)}.${name}`);
        if (found !== null) return found;
      }
    }
  }
  return null;
}

/**
 * The first place `value` does not fit `shape`, or null. The subset of JSON
 * Schema shapes use: type, enum, anyOf/oneOf, properties, required,
 * additionalProperties, items.
 */
export function misfit(shape: JsonObject, value: unknown, where: string): string | null {
  if (Array.isArray(shape["anyOf"]) || Array.isArray(shape["oneOf"])) {
    const options = (shape["anyOf"] ?? shape["oneOf"]) as JsonObject[];
    return options.some((o) => misfit(o, value, where) === null) ? null : `${where}: ${JSON.stringify(value)} fits none of its options`;
  }
  if (Array.isArray(shape["enum"])) {
    const ok = (shape["enum"] as unknown[]).some((e) => lmcc.jsonEqual(e as lmcc.Json, value as lmcc.Json));
    if (!ok) return `${where}: ${JSON.stringify(value)} is not one of ${JSON.stringify(shape["enum"])}`;
  }
  const type = shape["type"];
  if (typeof type === "string") {
    const actual = value === null ? "null" : Array.isArray(value) ? "array" : typeof value === "bigint" ? "integer" : typeof value;
    const fits = type === actual || (type === "number" && (actual === "number" || actual === "integer"))
      || (type === "integer" && ((actual === "number" && Number.isInteger(value)) || actual === "integer"));
    if (!fits) return `${where}: expected ${type}, got ${JSON.stringify(value)}`;
  }
  if (Array.isArray(value) && typeof shape["items"] === "object" && shape["items"] !== null) {
    for (let i = 0; i < value.length; i++) {
      const p = misfit(shape["items"] as JsonObject, value[i], `${where}[${i}]`);
      if (p) return p;
    }
  }
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    const obj = value as JsonObject;
    const props = (shape["properties"] ?? {}) as Record<string, JsonObject>;
    for (const name of (shape["required"] ?? []) as string[]) {
      if (!Object.hasOwn(obj, name)) return `${where}: missing ${name}`;
    }
    for (const [k, v] of entriesOf(obj)) {
      if (Object.hasOwn(props, k)) {
        const p = misfit(props[k]!, v, `${where}.${k}`);
        if (p) return p;
      } else if (shape["additionalProperties"] === false) {
        return `${where}: unexpected ${k}`;
      } else if (typeof shape["additionalProperties"] === "object" && shape["additionalProperties"] !== null) {
        const p = misfit(shape["additionalProperties"] as JsonObject, v, `${where}.${k}`);
        if (p) return p;
      }
    }
  }
  return null;
}

/**
 * A note a program declares (programs.md, *Notes*): its data's shape (a
 * shape, zod or another Standard Schema), or `{ shape, desc?, latest?,
 * outside? }`. `latest`: only its latest value matters (a stage, a count so
 * far); `outside`: a caller who sees only the program's boundary may see it.
 */
export type NoteSpec = Shape<unknown> | ZodLike | StandardSchemaLike
  | { readonly shape: Shape<unknown> | ZodLike | StandardSchemaLike; readonly desc?: string; readonly latest?: boolean; readonly outside?: boolean };

/** Notes declared in TypeScript (`{ name: NoteSpec }`) as an interface's `notes`, in order; `undefined` when none were declared. */
export function notesOf(declared: Readonly<Record<string, NoteSpec>> | undefined, where: string): InterfaceNote[] | undefined {
  if (declared === undefined) return undefined;
  if (typeof declared !== "object" || declared === null || Array.isArray(declared)) {
    throw new TypeError(`${where}: notes is an object, from a note's name to its shape or { shape, desc?, latest?, outside? }`);
  }
  const out: InterfaceNote[] = [];
  for (const [name, spec] of entriesOf(declared as Record<string, unknown>)) {
    let field: unknown = spec;
    let latest = false;
    let outside = false;
    if (isWrapper(spec)) {
      const { latest: l, outside: o, ...rest } = spec as unknown as Record<string, unknown>;
      if (l !== undefined && typeof l !== "boolean") throw new InterfaceError("interface-malformed", name, `${where}: note ${JSON.stringify(name)}: latest is true or false`);
      if (o !== undefined && typeof o !== "boolean") throw new InterfaceError("interface-malformed", name, `${where}: note ${JSON.stringify(name)}: outside is true or false`);
      if (Object.hasOwn(rest, "optional") || Object.hasOwn(rest, "opaque")) {
        throw new InterfaceError("interface-malformed", name, `${where}: note ${JSON.stringify(name)}: a note is { shape, desc?, latest?, outside? }; its data is JSON, never left out or opaque`);
      }
      latest = l === true;
      outside = o === true;
      field = rest;
    }
    const { shape, desc, files } = readField(field as FieldSpec, `${where}: note ${JSON.stringify(name)}`, name);
    if (files || isOpaque(field)) throw new InterfaceError("interface-malformed", name, `${where}: note ${JSON.stringify(name)}: a note's data is JSON`);
    const note: { name: string; shape: JsonObject; desc?: string; latest?: true; outside?: true } = { name, shape };
    if (desc) note.desc = desc;
    if (latest) note.latest = true;
    if (outside) note.outside = true;
    out.push(note);
  }
  return out;
}
