/**
 * The contract's JSON Schemas (contract/schema/), checked here without a
 * JSON Schema library: the keywords those schemas use, read as draft
 * 2020-12 reads them. Patterns are ECMA-262 regular expressions, as the
 * standard says (the contract's generator reads them so too).
 *
 * A store refuses an event that does not pass `event` (`event-malformed`),
 * and a loader a manifest that does not pass `saved` (`saved-malformed`).
 */

import * as lmcc from "lmcc";
import { SCHEMAS } from "./generated/contract.ts";

type Rec = Record<string, unknown>;
export type SchemaName = keyof typeof SCHEMAS;

const FILES: Record<string, SchemaName> = {
  "call.schema.json": "call", "event.schema.json": "event", "interface.schema.json": "interface",
  "rating.schema.json": "rating", "saved.schema.json": "saved",
};

const patterns = new Map<string, RegExp>();
const regex = (p: string): RegExp => {
  let r = patterns.get(p);
  if (!r) {
    r = new RegExp(p, "u");
    patterns.set(p, r);
  }
  return r;
};

const isObject = (v: unknown): v is Rec => typeof v === "object" && v !== null && !Array.isArray(v);

function typeFits(v: unknown, t: string): boolean {
  switch (t) {
    case "null": return v === null;
    case "boolean": return typeof v === "boolean";
    case "integer": return typeof v === "number" && Number.isInteger(v);
    case "number": return typeof v === "number" && Number.isFinite(v);
    case "string": return typeof v === "string";
    case "array": return Array.isArray(v);
    case "object": return isObject(v);
    default: return false;
  }
}

/** The schema a `$ref` names: `#/$defs/x` in the same file, or `file.schema.json` and its fragment. */
function resolve(ref: string, root: SchemaName): [Rec, SchemaName] {
  const [file, fragment = ""] = ref.split("#");
  const name = file ? FILES[file] : root;
  if (!name) throw new Error(`schema: unknown reference ${ref}`);
  let node: unknown = SCHEMAS[name];
  for (const part of fragment.split("/").filter(Boolean)) node = (node as Rec)[part];
  if (!isObject(node)) throw new Error(`schema: unknown reference ${ref}`);
  return [node, name];
}

function valid(v: unknown, s: unknown, root: SchemaName): boolean {
  if (s === true || s === undefined) return true;
  if (s === false) return false;
  const schema = s as Rec;
  if (typeof schema["$ref"] === "string") {
    const [target, file] = resolve(schema["$ref"], root);
    if (!valid(v, target, file)) return false;
  }
  if (schema["type"] !== undefined) {
    const types = Array.isArray(schema["type"]) ? schema["type"] as string[] : [schema["type"] as string];
    if (!types.some((t) => typeFits(v, t))) return false;
  }
  if (Object.hasOwn(schema, "const") && !lmcc.jsonEqual(schema["const"] as lmcc.Json, v as lmcc.Json)) return false;
  if (Array.isArray(schema["enum"]) && !schema["enum"].some((x) => lmcc.jsonEqual(x as lmcc.Json, v as lmcc.Json))) return false;
  for (const key of ["allOf", "anyOf", "oneOf"] as const) {
    const list = schema[key] as unknown[] | undefined;
    if (!Array.isArray(list)) continue;
    const n = list.filter((x) => valid(v, x, root)).length;
    if (key === "allOf" ? n !== list.length : key === "anyOf" ? n === 0 : n !== 1) return false;
  }
  if (schema["not"] !== undefined && valid(v, schema["not"], root)) return false;
  if (schema["if"] !== undefined) {
    const branch = valid(v, schema["if"], root) ? schema["then"] : schema["else"];
    if (branch !== undefined && !valid(v, branch, root)) return false;
  }
  if (typeof v === "string") {
    if (typeof schema["minLength"] === "number" && [...v].length < schema["minLength"]) return false;
    if (typeof schema["maxLength"] === "number" && [...v].length > schema["maxLength"]) return false;
    if (typeof schema["pattern"] === "string" && !regex(schema["pattern"]).test(v)) return false;
  }
  if (typeof v === "number") {
    if (typeof schema["minimum"] === "number" && v < schema["minimum"]) return false;
    if (typeof schema["maximum"] === "number" && v > schema["maximum"]) return false;
  }
  if (Array.isArray(v)) {
    if (typeof schema["minItems"] === "number" && v.length < schema["minItems"]) return false;
    if (schema["items"] !== undefined && !v.every((x) => valid(x, schema["items"], root))) return false;
    if (schema["uniqueItems"] === true && new Set(v.map((x) => lmcc.canonicalJson(x as lmcc.Json))).size !== v.length) return false;
  }
  if (isObject(v)) {
    if (typeof schema["maxProperties"] === "number" && Object.keys(v).length > schema["maxProperties"]) return false;
    if (Array.isArray(schema["required"]) && !(schema["required"] as string[]).every((k) => Object.hasOwn(v, k))) return false;
    const props = (schema["properties"] ?? {}) as Rec;
    for (const [k, x] of Object.entries(v)) {
      if (schema["propertyNames"] !== undefined && !valid(k, schema["propertyNames"], root)) return false;
      if (Object.hasOwn(props, k)) {
        if (!valid(x, props[k], root)) return false;
      } else if (schema["additionalProperties"] !== undefined && !valid(x, schema["additionalProperties"], root)) {
        return false;
      }
    }
  }
  return true;
}

/** Whether `value` passes the contract's schema `name` (a call record, a rating, an event, an interface, a manifest). */
export function passes(name: SchemaName, value: unknown): boolean {
  return valid(value, SCHEMAS[name], name);
}
