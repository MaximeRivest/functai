/**
 * Shapes: the JSON Schema of each input and output (contract/functions.md,
 * "A definition"). Write them with `t` (no dependency), with zod 4, or as
 * plain JSON Schema. The same data type has the same shape in every
 * language, so zod's JSON Schema is read the way Python writes the type.
 */

import * as lmcc from "lmcc";
import { InterfaceError } from "./interface.ts";
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

/** `to`, carrying `from`'s described answers. */
function carry<S>(from: unknown, to: S): S {
  const m = meaningsOf(from);
  if (m.length) Object.defineProperty(to, MEANINGS, { value: m });
  return to;
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
  /** A list: `t.list(t.string(), { description: "one per line" })`. */
  list: <T>(items: Shape<T>, extra: Extra = {}): Shape<T[]> => carry(items, { ...lmcc.t.list(items), ...extraOf(extra, "list", LIST) }) as Shape<T[]>,
  /** An object with these properties, all required: `t.object({ id: t.integer() }, { description })`. */
  object: <P extends Record<string, Shape<unknown>>>(properties: P, extra: Extra = {}): ReturnType<typeof lmcc.t.object<P>> =>
    ({ ...lmcc.t.object(Object.fromEntries(entriesOf(properties).map(([k, v]) => [k, described(v)])) as P), ...extraOf(extra, "object", OBJECT) }) as ReturnType<typeof lmcc.t.object<P>>,
  /** A map from text keys to values: `t.record(t.integer())`. */
  record: <V>(values: Shape<V>, extra: Extra = {}): Shape<Record<string, V>> =>
    ({ type: "object", additionalProperties: described(values), ...extraOf(extra, "record", RECORD) }) as Shape<Record<string, V>>,
  /** The shape or null; a caller may leave it out, and it is then null. */
  optional: <T>(shape: Shape<T>): Shape<T | null> => carry(shape, lmcc.t.nullable(shape)),
  /** The shape, with the value an input takes when a caller leaves it out: `t.withDefault(t.string(), "kind")`. */
  withDefault: <T>(shape: Shape<T>, value: T): Shape<T> & { readonly default: T } => carry(shape, { ...shape, default: value }) as Shape<T> & { readonly default: T },
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
  /** Any JSON value (shape `{}`), with words about it if you like: `t.json({ description: "each input by name" })`. */
  json: (extra: Extra = {}): Shape<Json> => ({ ...extra }) as Shape<Json>,
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
export function readField(spec: FieldSpec, where: string, field?: string): { shape: JsonObject; desc: string | null } {
  let desc: string | null = null;
  let raw: unknown = spec;
  const refuse = (why: string): never => {
    if (field !== undefined) throw new InterfaceError("interface-malformed", field, `${where}: ${why}`);
    throw new TypeError(`${where}: ${why}`);
  };
  if (isWrapper(spec)) {
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
  if (typeof shape["description"] === "string") {
    desc = desc ?? (shape["description"] as string);
    delete shape["description"];
  }
  return { shape, desc: withChoices(desc, meanings) };
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
