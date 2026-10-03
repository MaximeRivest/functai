/**
 * Values as the log holds them (contract/calls.md, "Values"): the JSON a
 * value's type describes, or, for a value with no JSON form, a description
 * `{ $type, $repr }`.
 */

import * as lmcc from "lmcc";
import { RawNumber } from "@lm15/lm15";

type Rec = Record<string, unknown>;

const isPlain = (v: object): boolean => {
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
};

/** A value's JSON form, or `undefined` when it has none (a class instance, a function, a Map, NaN, …). */
export function jsonForm(value: unknown): lmcc.Json | undefined {
  try {
    return plain(value);
  } catch {
    return undefined;
  }
}

function plain(v: unknown): lmcc.Json {
  v = unboxed(v);
  if (v === null || typeof v === "string" || typeof v === "boolean") return v;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) throw new TypeError("not finite");
    return v;
  }
  if (typeof v === "bigint") {
    // an integer past 2^53 is data, as Python's int is (lmcc reads one as a bigint, and writes it as its digits)
    return v <= BigInt(Number.MAX_SAFE_INTEGER) && v >= BigInt(-Number.MAX_SAFE_INTEGER) ? Number(v) : v;
  }
  if (v instanceof Date) return v.toISOString();
  if (Array.isArray(v)) {
    const out: lmcc.Json[] = [];
    for (let i = 0; i < v.length; i++) out.push(v[i] === undefined ? null : plain(v[i]));   // a hole is null, as JSON writes it
    return out;
  }
  if (typeof v === "object") {
    const proto = Object.getPrototypeOf(v);
    if (proto !== Object.prototype && proto !== null) throw new TypeError("not plain data");
    const out: Record<string, lmcc.Json> = {};
    for (const k of lmcc.memberNames(v)) {
      const x = (v as Rec)[k];
      if (x !== undefined) lmcc.setMember(out, k, plain(x));
    }
    return out;
  }
  throw new TypeError(typeof v);
}

/** Whether `method` (a primitive wrapper's `valueOf`) accepts `v`: whether `v` holds that primitive (a slot, not a prototype). */
function holds(method: () => unknown, v: object): boolean {
  try {
    method.call(v);
    return true;
  } catch {
    return false;
  }
}

/**
 * A boxed primitive (`new Number(42)`, `new String("x")`, `new Boolean(false)`, `Object(1n)`) as the value it holds,
 * as JSON serialization reads one; anything else as it is. Only these four: no other object's `valueOf` is read.
 */
export function unboxed(v: unknown): unknown {
  if (v === null || typeof v !== "object" || Array.isArray(v) || isPlain(v)) return v;
  if (holds(Number.prototype.valueOf, v)) return Number(v);
  if (holds(String.prototype.valueOf, v)) return String(v);
  if (holds(Boolean.prototype.valueOf, v)) return Boolean.prototype.valueOf.call(v);
  if (holds(BigInt.prototype.valueOf, v)) return BigInt.prototype.valueOf.call(v);
  return v;
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
 * Records keyed by names (field names, JSON members) are data: any name is an
 * own member or absent, never one `Object.prototype` has (`toString`,
 * `__proto__`), and members keep the order the value holds, integer-like
 * names (`"10"`) included, which JavaScript would list first. lmcc carries
 * that order on the objects it builds (its `memberNames`, `setMember`); these
 * are the ways FunctAI reads, writes, copies, parses and writes out such
 * records, so every record, request and saved file holds a value's members
 * in its order, as Python's do (tests/order.test.ts).
 */

// An lmcc without these (0.8.4 and earlier) reads a missing output named `toString` as `""`, drops a `__proto__`
// member, and reorders members: functai would send and keep other data than it was given. It refuses to run instead.
const HELPERS = ["setMember", "ownValue", "memberNames", "orderedObject", "copyObject", "parseJson", "jsonText"] as const;
const missing = HELPERS.filter((h) => typeof (lmcc as unknown as Record<string, unknown>)[h] !== "function");
if (missing.length) {
  throw new Error(`functai needs lmcc with decision D-58 (names are data, members keep their order; lmcc 0.8.5 or later): this lmcc has no ${missing.join(", ")}`);
}

/** Set a member as data: `__proto__` is an own member, and a new name comes after the others (lmcc's `setMember`). */
export function setOwn(obj: object, key: string, value: unknown): void {
  lmcc.setMember(obj as Record<string, unknown>, key, value);
}

/** An own member's value (never an inherited one: `toString`, `constructor`), or undefined. */
export function getOwn<T>(obj: Readonly<Record<string, T>> | null | undefined, key: string): T | undefined {
  return obj !== null && obj !== undefined ? lmcc.ownValue(obj, key) as T | undefined : undefined;
}

/** A record's names in the value's order (lmcc's `memberNames`). */
export const namesOf = (obj: object): string[] => lmcc.memberNames(obj);

/** A record's (name, value) pairs in the value's order. */
export function entriesOf<T = unknown>(obj: Readonly<Record<string, T>>): [string, T][] {
  return lmcc.memberNames(obj).map((k) => [k, obj[k] as T]);
}

/** A record of these (name, value) pairs, in their order, any name a member. */
export function recordOf<T = unknown>(entries: Iterable<readonly [string, T]>): Record<string, T> {
  return lmcc.orderedObject(entries);
}

/**
 * A deep copy, as `structuredClone` makes one, that keeps each object's
 * members in the value's order (a structured clone loses lmcc's record of
 * it): plain objects and arrays are copied member by member, anything else
 * by `structuredClone` (which refuses a function, as it does).
 */
export function copyData<T>(value: T): T {
  const seen = new Map<object, unknown>();
  const walk = (v: unknown): unknown => {
    if (typeof v === "function" || typeof v === "symbol") return structuredClone(v);
    if (v === null || typeof v !== "object") return v;
    if (seen.has(v)) return seen.get(v);
    if (v instanceof RawNumber) return new RawNumber(v.raw);   // a number lm15 keeps as written (a structured clone would make it an object)
    if (Array.isArray(v)) {
      const out: unknown[] = new Array(v.length);
      seen.set(v, out);
      for (let i = 0; i < v.length; i++) if (i in v) out[i] = walk(v[i]);
      return out;
    }
    if (!isPlain(v)) {
      const out = structuredClone(v);
      seen.set(v, out);
      return out;
    }
    const out: Rec = {};
    seen.set(v, out);
    for (const k of lmcc.memberNames(v)) lmcc.setMember(out, k, walk((v as Rec)[k]));
    return out;
  };
  return walk(value) as T;
}

/**
 * A value as the contract's canonical JSON reads it (calls.md, "Canonical
 * JSON"): an lm15 `RawNumber` (a number kept as written, `0.0`) becomes the
 * number it is (`0`, as ECMAScript writes it; an integer past 2^53, a
 * `bigint`, its digits kept). What lmcc hashes, so a key is every language's.
 */
export function canonicalData<T>(value: T): T {
  const walk = (v: unknown): unknown => {
    if (v instanceof RawNumber) {
      const n = Number(v.raw);
      return /^-?\d+$/.test(v.raw) && !Number.isSafeInteger(n) ? BigInt(v.raw) : n;
    }
    if (Array.isArray(v)) return v.map(walk);
    if (v !== null && typeof v === "object" && isPlain(v)) {
      const out: Rec = {};
      for (const k of lmcc.memberNames(v)) lmcc.setMember(out, k, walk((v as Rec)[k]));
      return out;
    }
    return v;
  };
  return walk(value) as T;
}

/** JSON text as a record's reader expects it: lmcc's parser, which keeps members in the order written (JSON.parse does not). */
export function parseData(text: string): unknown {
  return lmcc.parseJson(text);
}

/**
 * JSON text of a value, members in the value's order (`JSON.stringify` lists
 * integer-like names first): what `JSON.stringify(value, null, indent)`
 * writes otherwise (a `toJSON` is used; a boxed primitive is its value; an
 * `undefined`, a function or a symbol member is left out, and is `null` in
 * an array, as a hole is; a number that is not finite is `null`; a value
 * that holds itself is refused with a `TypeError`), except that a `bigint`
 * is written as its digits and an lm15 `RawNumber` as it came (a
 * temperature of `0.0` stays `0.0`).
 */
export function writeData(value: unknown, indent = 0): string {
  const out: string[] = [];
  const open = new Set<object>();          // the objects and arrays being written, outermost first
  const skip = (v: unknown) => v === undefined || typeof v === "function" || typeof v === "symbol";
  const lowered = (v: unknown, key: string): unknown =>
    unboxed(v !== null && typeof v === "object" && !(v instanceof RawNumber) && typeof (v as { toJSON?: unknown }).toJSON === "function"
      ? (v as { toJSON(k: string): unknown }).toJSON(key) : v);
  const write = (v: unknown, depth: number): void => {
    if (v === null || skip(v)) {
      out.push("null");
      return;
    }
    if (typeof v === "bigint") {
      out.push(v.toString());
      return;
    }
    if (v instanceof RawNumber) {
      out.push(v.raw);
      return;
    }
    if (typeof v !== "object") {
      out.push(JSON.stringify(v) ?? "null");
      return;
    }
    if (open.has(v)) throw new TypeError("Converting circular structure to JSON");
    open.add(v);
    const nl = indent > 0 ? "\n" + " ".repeat(indent * (depth + 1)) : "";
    const close = indent > 0 ? "\n" + " ".repeat(indent * depth) : "";
    if (Array.isArray(v)) {
      if (!v.length) out.push("[]");
      else {
        out.push("[");
        for (let i = 0; i < v.length; i++) {             // every index: a hole is null, as JSON writes it
          out.push(i ? "," + nl : nl);
          write(lowered(v[i], String(i)), depth + 1);
        }
        out.push(close, "]");
      }
      open.delete(v);
      return;
    }
    let first = true;
    out.push("{");
    for (const k of lmcc.memberNames(v)) {
      const x = lowered((v as Rec)[k], k);
      if (skip(x)) continue;
      out.push(first ? nl : "," + nl, JSON.stringify(k), indent > 0 ? ": " : ":");
      first = false;
      write(x, depth + 1);
    }
    out.push(first ? "}" : close + "}");
    open.delete(v);
  };
  write(lowered(value, ""), 0);
  return out.join("");
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
