/**
 * Values as the log holds them (contract/calls.md, "Values"): the JSON a
 * value's type describes, or, for a value with no JSON form, a description
 * `{ $type, $repr }`.
 */

import * as lmcc from "lmcc";

type Rec = Record<string, unknown>;

/** A value's JSON form, or `undefined` when it has none (a class instance, a function, a Map, NaN, …). */
export function jsonForm(value: unknown): lmcc.Json | undefined {
  try {
    return plain(value);
  } catch {
    return undefined;
  }
}

function plain(v: unknown): lmcc.Json {
  if (v === null || typeof v === "string" || typeof v === "boolean") return v;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) throw new TypeError("not finite");
    return v;
  }
  if (typeof v === "bigint") {
    if (v <= BigInt(Number.MAX_SAFE_INTEGER) && v >= BigInt(-Number.MAX_SAFE_INTEGER)) return Number(v);
    throw new TypeError("bigint");
  }
  if (v instanceof Date) return v.toISOString();
  if (Array.isArray(v)) return v.map(plain);
  if (typeof v === "object") {
    const proto = Object.getPrototypeOf(v);
    if (proto !== Object.prototype && proto !== null) throw new TypeError("not plain data");
    const out: Record<string, lmcc.Json> = {};
    for (const [k, x] of Object.entries(v as Rec)) if (x !== undefined) setOwn(out, k, plain(x));
    return out;
  }
  throw new TypeError(typeof v);
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (typeof v === "object") return (v as object).constructor?.name ?? "Object";
  return typeof v;
}

/** What the log writes for a value: its JSON form, or its description; its size; whether it is a description. */
export function toJson(value: unknown): [lmcc.Json, number, boolean] {
  const data = jsonForm(value);
  if (data !== undefined) return [data, [...lmcc.canonicalJson(data)].length, false];
  let text: string;
  try {
    text = String(value);
  } catch {
    text = Object.prototype.toString.call(value);
  }
  const described = { $type: typeName(value), $repr: [...text].slice(0, 2000).join("") };
  return [described, [...lmcc.canonicalJson(described)].length, true];
}

/**
 * Set an own property, whatever its name: `obj["__proto__"] = v` would set
 * the object's prototype instead (a JSON member named `__proto__` is data).
 */
export function setOwn(obj: object, key: string, value: unknown): void {
  if (key === "__proto__") Object.defineProperty(obj, key, { value, enumerable: true, writable: true, configurable: true });
  else (obj as Record<string, unknown>)[key] = value;
}

/** An own property's value (never an inherited one: `toString`, `constructor`), or undefined. */
export function getOwn<T>(obj: Readonly<Record<string, T>> | null | undefined, key: string): T | undefined {
  return obj !== null && obj !== undefined && Object.hasOwn(obj, key) ? obj[key] : undefined;
}

/** Code-point order of two strings (JavaScript's `<` compares UTF-16 code units). */
export function byCodePoint(a: string, b: string): number {
  const x = [...a], y = [...b];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i]!.codePointAt(0)! - y[i]!.codePointAt(0)!;
    if (d) return d;
  }
  return x.length - y.length;
}
