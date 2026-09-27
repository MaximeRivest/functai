/**
 * Shapes: the JSON Schema of each input and output (contract/functions.md,
 * "A definition"). Write them with `t` (no dependency), with zod 4, or as
 * plain JSON Schema. The same data type has the same shape in every
 * language, so zod's JSON Schema is read the way Python writes the type.
 */

import * as lmcc from "lmcc";

type JsonObject = Record<string, unknown>;

/** A JSON Schema carrying the static type of its values (erased at run time). */
export type Shape<T = unknown> = lmcc.TypedShape<T>;

/** A zod 4 schema, as far as this package needs one (no dependency on zod). */
export interface ZodLike<T = unknown> {
  readonly _zod: { readonly output: T };
  toJSONSchema(params?: { io?: "input" | "output" }): JsonObject;
}

/** What a field is written as: a shape, a zod schema, or `{ shape, desc }`. */
export type FieldSpec<T = unknown> = Shape<T> | ZodLike<T> | { readonly shape: Shape<T> | ZodLike<T>; readonly desc?: string };

/** The value type of a field spec. */
export type ValueOf<S> =
  S extends ZodLike<infer T> ? T :
  S extends { readonly shape: infer X } ? ValueOf<X> :
  S extends lmcc.TypedShape<infer T> ? (unknown extends T ? unknown : T) : unknown;

/** Builders for shapes (lmcc's, plus a map and `optional`). Each takes extra JSON Schema keys. */
export const t = {
  ...lmcc.t,
  /** Text, integers, numbers or yes/no: `t.string()`, `t.integer()`, `t.number()`, `t.boolean()`. */
  /** A map from text keys to values: `t.record(t.integer())`. */
  record: <V>(values: Shape<V>): Shape<Record<string, V>> => ({ type: "object", additionalProperties: values }) as Shape<Record<string, V>>,
  /** The shape or null. */
  optional: <T>(shape: Shape<T>): Shape<T | null> => lmcc.t.nullable(shape),
};

/** A field with words about it: `describe(t.string(), "the customer's own words")`. */
export function describe<T>(shape: Shape<T> | ZodLike<T>, desc: string): { shape: Shape<T> | ZodLike<T>; desc: string } {
  return { shape, desc };
}

function isZod(x: unknown): x is ZodLike {
  return typeof x === "object" && x !== null && "_zod" in x && typeof (x as ZodLike).toJSONSchema === "function";
}

const SAFE = Number.MAX_SAFE_INTEGER;

/** zod's JSON Schema as Python writes the same type: no `$schema`, no safe-integer bounds, nullable as `anyOf`. */
function fromZod(schema: JsonObject): JsonObject {
  const walk = (s: unknown): unknown => {
    if (Array.isArray(s)) return s.map(walk);
    if (typeof s !== "object" || s === null) return s;
    const o: JsonObject = {};
    for (const [k, v] of Object.entries(s)) {
      if (k === "$schema") continue;
      o[k] = walk(v);
    }
    if (o["type"] === "integer" && o["minimum"] === -SAFE && o["maximum"] === SAFE) {
      delete o["minimum"];
      delete o["maximum"];
    }
    if (o["type"] === "object" && "additionalProperties" in o && typeof o["additionalProperties"] === "object"
      && JSON.stringify(o["propertyNames"]) === '{"type":"string"}') {
      delete o["propertyNames"];
    }
    if ("const" in o && !("enum" in o)) {
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

/** A field spec as `{shape, desc}`: a top-level `description` in the shape becomes the desc. */
export function readField(spec: FieldSpec, where: string): { shape: JsonObject; desc: string | null } {
  let desc: string | null = null;
  let raw: unknown = spec;
  if (typeof spec === "object" && spec !== null && "shape" in spec && !("type" in spec) && !isZod(spec)) {
    desc = (spec as { desc?: string }).desc ?? null;
    raw = (spec as { shape: unknown }).shape;
  }
  let shape: JsonObject;
  if (isZod(raw)) shape = fromZod(raw.toJSONSchema({ io: "input" }));
  else if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) shape = structuredClone(raw) as JsonObject;
  else throw new TypeError(`${where}: expected a shape (t.string(), a zod schema or JSON Schema), not ${JSON.stringify(raw)}`);
  if (typeof shape["description"] === "string") {
    desc = desc ?? (shape["description"] as string);
    delete shape["description"];
  }
  return { shape, desc };
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
      if (!(name in obj)) return `${where}: missing ${name}`;
    }
    for (const [k, v] of Object.entries(obj)) {
      if (k in props) {
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
