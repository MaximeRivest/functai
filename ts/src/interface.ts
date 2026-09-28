/**
 * A program's interface (contract/programs.md): the inputs it takes and the
 * outputs it gives, named and typed as JSON. Every program has one: an AI
 * function's is its definition's, a module declares its own. This file is
 * the contract's rules, as data: which interfaces are refused, their
 * signature, and whether values fit them.
 */

import * as lmcc from "lmcc";
import { byCodePoint, jsonForm } from "./values.ts";

type Rec = Record<string, unknown>;

/** A field of an interface (programs.md, "The interface"). */
export interface InterfaceField {
  readonly name: string;
  /** JSON Schema, read by the vocabulary programs.md lists. `{}` is any JSON value. */
  readonly shape: Rec;
  readonly desc?: string;
  /** The host language's name for the type, for people. */
  readonly type?: string;
  /** Values may have no JSON form (a class instance, a buffer); never checked. Its shape is `{}`. */
  readonly opaque?: true;
  /** Inputs only: a caller may leave it out (it then takes its shape's `default`, or stays out). */
  readonly optional?: true;
}

/** What a program takes and gives, as data (the same JSON in every language). */
export interface Interface {
  readonly description: string;
  readonly inputs: readonly InterfaceField[];
  /** At least one; the last is the answer. */
  readonly outputs: readonly InterfaceField[];
}

export type InterfaceCode = "interface-input" | "interface-output" | "interface-malformed";

/**
 * A program was given, or returned, what its interface does not take or give
 * (`interface-input`, `interface-output`), or its interface breaks the rules
 * (`interface-malformed`, when it is defined or read). `field` names the
 * first field at fault (null when the fault is not a field's).
 */
export class InterfaceError extends TypeError {
  readonly code: InterfaceCode;
  readonly field: string | null;
  constructor(code: InterfaceCode, field: string | null, message: string) {
    super(message);
    this.code = code;
    this.field = field;
    this.name = "InterfaceError";
  }
}

const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const REF = /^#\/\$defs\/([A-Za-z0-9_.-]+)$/;
const TYPES = new Set(["null", "boolean", "integer", "number", "string", "array", "object"]);
const ANNOTATIONS: Record<string, (v: unknown) => boolean> = {
  title: (v) => typeof v === "string", description: (v) => typeof v === "string", format: (v) => typeof v === "string",
  $comment: (v) => typeof v === "string", deprecated: (v) => typeof v === "boolean", readOnly: (v) => typeof v === "boolean",
  writeOnly: (v) => typeof v === "boolean", examples: Array.isArray, default: () => true,
};
const ASSERTIONS = new Set(["type", "enum", "const", "anyOf", "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
  "properties", "required", "additionalProperties", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
  "exclusiveMaximum", "$ref", "$defs"]);
const INPUT_KEYS = new Set(["name", "shape", "desc", "type", "opaque", "optional"]);
const OUTPUT_KEYS = new Set(["name", "shape", "desc", "type", "opaque"]);

const isObject = (v: unknown): v is Rec => typeof v === "object" && v !== null && !Array.isArray(v);
const isNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** The JSON type of a JSON value: an integer is a number with no fraction (`5.0` is one). */
function jsonType(v: unknown): string {
  if (v === null) return "null";
  if (typeof v === "boolean") return "boolean";
  if (typeof v === "number") return Number.isInteger(v) ? "integer" : "number";
  if (typeof v === "string") return "string";
  if (Array.isArray(v)) return "array";
  return "object";
}

/** A field's shape without its own `default`: what its data looks like. */
export function dataShape(shape: Rec): Rec {
  const { default: _default, ...rest } = shape;
  return rest;
}

/** The interface's signature: lmcc's fingerprint of its fields, plain and untyped, each shape without its own default. */
export function interfaceSignature(iface: Interface): string {
  const field = (direction: string) => (f: InterfaceField) =>
    ({ direction, name: f.name, purpose: "plain", shape: dataShape(f.shape), type: "" });
  return lmcc.sha256([...iface.inputs.map(field("input")), ...iface.outputs.map(field("output"))] as lmcc.Json);
}

/**
 * Whether a shape uses the vocabulary, each keyword with a value of its kind.
 * `carry`: an AI function's shape, whose other keywords are lmcc's (carried,
 * never read); a module's may have no other keyword.
 */
function wellFormed(shape: unknown, root: Rec, carry: boolean): boolean {
  if (!isObject(shape)) return false;
  for (const [k, v] of Object.entries(shape)) {
    if (k in ANNOTATIONS) {
      if (!ANNOTATIONS[k]!(v)) return false;
      continue;
    }
    if (!ASSERTIONS.has(k)) {
      if (carry) continue;
      return false;
    }
    switch (k) {
      case "type": {
        const names = Array.isArray(v) ? v : [v];
        if (!names.length || names.some((n) => typeof n !== "string" || !TYPES.has(n)) || new Set(names).size !== names.length) return false;
        break;
      }
      case "enum":
        if (!Array.isArray(v) || !v.length) return false;
        break;
      case "anyOf": case "prefixItems":
        if (!Array.isArray(v) || !v.length || !v.every((x) => wellFormed(x, root, carry))) return false;
        break;
      case "items":
        if (!wellFormed(v, root, carry)) return false;
        break;
      case "properties": case "$defs":
        if (!isObject(v) || !Object.values(v).every((x) => wellFormed(x, root, carry))) return false;
        break;
      case "additionalProperties":
        if (typeof v !== "boolean" && !wellFormed(v, root, carry)) return false;
        break;
      case "required":
        if (!Array.isArray(v) || !v.every((x) => typeof x === "string") || new Set(v).size !== v.length) return false;
        break;
      case "minItems": case "maxItems": case "minLength": case "maxLength":
        if (!(jsonType(v) === "integer" && (v as number) >= 0)) return false;
        break;
      case "minimum": case "maximum": case "exclusiveMinimum": case "exclusiveMaximum":
        if (!isNumber(v)) return false;
        break;
      case "uniqueItems":
        if (typeof v !== "boolean") return false;
        break;
      case "$ref": {
        const m = typeof v === "string" ? REF.exec(v) : null;
        const defs = isObject(root["$defs"]) ? root["$defs"] : {};
        if (!m || !Object.hasOwn(defs, m[1]!)) return false;
        break;
      }
    }
  }
  return true;
}

/** The `$defs` a shape checks the same value against: its `$ref`, and those of its `anyOf`'s shapes. */
function sameValueRefs(shape: Rec): Set<string> {
  const out = new Set<string>();
  if (typeof shape["$ref"] === "string") out.add(REF.exec(shape["$ref"])![1]!);
  for (const x of (shape["anyOf"] ?? []) as Rec[]) for (const n of sameValueRefs(x)) out.add(n);
  return out;
}

/** Whether a `$defs` entry reaches itself by `$ref` and `anyOf` alone (checking a value against it would never end). */
function loops(root: Rec): boolean {
  const defs = (isObject(root["$defs"]) ? root["$defs"] : {}) as Record<string, Rec>;
  const graph = new Map(Object.entries(defs).map(([n, d]) => [n, sameValueRefs(d)]));
  const reaches = (start: string, seen: string[]): boolean => {
    for (const n of graph.get(start) ?? []) {
      if (n === seen[0] || (!seen.includes(n) && reaches(n, [...seen, n]))) return true;
    }
    return false;
  };
  return [...graph.keys()].some((n) => reaches(n, [n]));
}

/** Whether a JSON value fits a shape of the vocabulary (programs.md, "Checking values"); other keywords are never read. */
export function fitsShape(v: unknown, shape: Rec, root: Rec): boolean {
  const t = jsonType(v);
  if (typeof shape["$ref"] === "string") {
    const target = (root["$defs"] as Record<string, Rec> | undefined)?.[REF.exec(shape["$ref"])?.[1] ?? ""];
    if (!target || !fitsShape(v, target, root)) return false;
  }
  if (shape["type"] !== undefined) {
    const names = (Array.isArray(shape["type"]) ? shape["type"] : [shape["type"]]) as string[];
    if (!(names.includes(t) || (t === "integer" && names.includes("number")))) return false;
  }
  const same = (a: unknown, b: unknown) => lmcc.canonicalJson(a as lmcc.Json) === lmcc.canonicalJson(b as lmcc.Json);
  if (Array.isArray(shape["enum"]) && !shape["enum"].some((x) => same(x, v))) return false;
  if ("const" in shape && !same(shape["const"], v)) return false;
  if (Array.isArray(shape["anyOf"]) && !(shape["anyOf"] as Rec[]).some((s) => fitsShape(v, s, root))) return false;
  const n = (k: string) => shape[k] as number;
  if (t === "string") {
    const length = [...(v as string)].length;
    if (length < (n("minLength") ?? 0) || ("maxLength" in shape && length > n("maxLength"))) return false;
  }
  if (t === "integer" || t === "number") {
    const x = v as number;
    if (("minimum" in shape && x < n("minimum")) || ("maximum" in shape && x > n("maximum"))) return false;
    if ("exclusiveMinimum" in shape && x <= n("exclusiveMinimum")) return false;
    if ("exclusiveMaximum" in shape && x >= n("exclusiveMaximum")) return false;
  }
  if (t === "array") {
    const a = v as unknown[];
    const prefix = (shape["prefixItems"] ?? []) as Rec[];
    if (prefix.some((s, i) => i < a.length && !fitsShape(a[i], s, root))) return false;
    if (isObject(shape["items"]) && a.slice(prefix.length).some((x) => !fitsShape(x, shape["items"] as Rec, root))) return false;
    if (a.length < (n("minItems") ?? 0) || ("maxItems" in shape && a.length > n("maxItems"))) return false;
    if (shape["uniqueItems"] === true && new Set(a.map((x) => lmcc.canonicalJson(x as lmcc.Json))).size !== a.length) return false;
  }
  if (t === "object") {
    const o = v as Rec;
    const props = (shape["properties"] ?? {}) as Record<string, Rec>;
    if (((shape["required"] ?? []) as string[]).some((k) => !(k in o))) return false;
    for (const [k, x] of Object.entries(o)) {
      if (Object.hasOwn(props, k)) {
        if (!fitsShape(x, props[k]!, root)) return false;
      } else if ("additionalProperties" in shape) {
        const extra = shape["additionalProperties"];
        if (extra === false || (isObject(extra) && !fitsShape(x, extra, root))) return false;
      }
    }
  }
  return true;
}

/** Whether a value (as a program holds it) fits a field: opaque fields take anything; others take a value whose JSON form fits. */
export function fits(value: unknown, field: InterfaceField): boolean {
  if (field.opaque) return true;
  const data = jsonForm(value);
  if (data === undefined) return false;
  const shape = dataShape(field.shape);
  return fitsShape(data, shape, shape);
}

/**
 * The first fault of an interface, or null when it is accepted (programs.md,
 * "Interfaces that are refused"): `field` names the first field at fault,
 * inputs then outputs, in order; null when the fault is not a field's. `ai`:
 * an AI function's interface, whose shapes may carry lmcc's other keywords.
 */
export function malformed(iface: unknown, opts: { ai?: boolean } = {}): { field: string | null; why: string } | null {
  const ai = opts.ai ?? false;
  if (!isObject(iface) || Object.keys(iface).some((k) => !["description", "inputs", "outputs"].includes(k))
    || typeof iface["description"] !== "string" || !Array.isArray(iface["inputs"]) || !Array.isArray(iface["outputs"])
    || !iface["outputs"].length) {
    return { field: null, why: "an interface is {description, inputs, outputs}, with at least one output, and nothing else" };
  }
  const seen = new Set<string>();
  for (const direction of ["inputs", "outputs"] as const) {
    for (const f of iface[direction] as unknown[]) {
      const name = isObject(f) ? f["name"] : undefined;
      const fault = (why: string) => ({ field: typeof name === "string" ? name : null, why });
      if (!isObject(f)) return fault("a field is an object");
      const extra = Object.keys(f).filter((k) => !(direction === "inputs" ? INPUT_KEYS : OUTPUT_KEYS).has(k));
      if (extra.length) return fault(`a field has no key ${extra.sort(byCodePoint).join(", ")}`);
      if (typeof name !== "string" || !NAME.test(name)) return fault("a name is an ASCII identifier");
      if (seen.has(name)) return fault(`the name ${name} is used twice`);
      seen.add(name);
      if (["desc", "type"].some((k) => k in f && typeof f[k] !== "string")) return fault("desc and type are text");
      if (["opaque", "optional"].some((k) => k in f && f[k] !== true)) return fault("opaque and optional are true when present");
      const shape = f["shape"];
      if (!isObject(shape)) return fault("its shape is an object");
      if (!wellFormed(shape, shape, ai)) {
        return fault(ai ? "its shape uses a keyword the vocabulary lists with a value of another kind"
          : "its shape uses a keyword the vocabulary does not list, or one with a value of another kind");
      }
      if (loops(shape)) return fault("its shape has a $defs entry that comes back to itself by $ref and anyOf alone");
      if (ai && f["optional"] && !("default" in shape)) return fault("an AI function's optional input has a default (a model is sent every input)");
      if (f["opaque"] && Object.keys(shape).length) return fault("an opaque field's shape is {}");
      if ("default" in shape) {
        const ds = dataShape(shape);
        if (!fitsShape(shape["default"], ds, ds)) return fault(`its default ${JSON.stringify(shape["default"])} does not fit its shape`);
      }
    }
  }
  return null;
}

/** Refuse an interface that breaks the rules: `InterfaceError` (`interface-malformed`). */
export function checkInterface(iface: unknown, where: string, opts: { ai?: boolean } = {}): Interface {
  const fault = malformed(iface, opts);
  if (fault) {
    throw new InterfaceError("interface-malformed", fault.field,
      `${where}: ${fault.field === null ? "its interface" : `field ${fault.field}`}: ${fault.why}`);
  }
  return iface as Interface;
}

const show = (v: unknown) => {
  const data = jsonForm(v);
  const text = data === undefined ? Object.prototype.toString.call(v) : JSON.stringify(data);
  return text.length > 120 ? text.slice(0, 117) + "..." : text;
};

/**
 * A call's inputs checked against the interface (programs.md, "Checking
 * values", 1): every given name is an input, each given value fits, every
 * required one is given; an optional one left out takes its shape's default,
 * or stays left out. `undefined` is left out. Returns the inputs the code
 * gets, or throws `InterfaceError` (`interface-input`).
 */
export function checkInputs(iface: Interface, given: Readonly<Rec>, where: string): Rec {
  const names = iface.inputs.map((f) => f.name);
  const present = Object.keys(given).filter((k) => given[k] !== undefined);
  const unknown = present.filter((k) => !names.includes(k)).sort(byCodePoint);
  if (unknown.length) throw new InterfaceError("interface-input", unknown[0]!, `${where} has no input ${unknown[0]} (its inputs: ${names.join(", ") || "none"})`);
  const out: Rec = {};
  for (const f of iface.inputs) {
    if (present.includes(f.name)) {
      if (!fits(given[f.name], f)) {
        throw new InterfaceError("interface-input", f.name, `${where}: input ${f.name}: ${show(given[f.name])} does not fit ${JSON.stringify(dataShape(f.shape))}`);
      }
      out[f.name] = given[f.name];
    } else if (!f.optional) {
      throw new InterfaceError("interface-input", f.name, `${where} needs ${f.name}`);
    } else if ("default" in f.shape) {
      out[f.name] = structuredClone(f.shape["default"]);
    }
  }
  return out;
}

/**
 * What a program's code returned, as its outputs by name (programs.md,
 * "Checking values", 2): one output is the value; several are a record of
 * each by name and nothing else. Throws `InterfaceError` (`interface-output`).
 */
export function checkReturned(iface: Interface, returned: unknown, where: string): Rec {
  const outputs = iface.outputs;
  let values: Rec;
  if (outputs.length === 1) values = { [outputs[0]!.name]: returned };
  else {
    const first = outputs[0]!.name;
    const proto = typeof returned === "object" && returned !== null ? Object.getPrototypeOf(returned) : undefined;
    if (proto !== Object.prototype && proto !== null) {
      throw new InterfaceError("interface-output", first, `${where} returns its outputs as one record by name ({ ${outputs.map((f) => f.name).join(", ")} }), not ${show(returned)}`);
    }
    values = returned as Rec;
    const names = outputs.map((f) => f.name);
    const unknown = Object.keys(values).filter((k) => !names.includes(k)).sort(byCodePoint);
    if (unknown.length) throw new InterfaceError("interface-output", unknown[0]!, `${where} has no output ${unknown[0]} (its outputs: ${names.join(", ")})`);
  }
  for (const f of outputs) {
    if (!(f.name in values) || (outputs.length > 1 && values[f.name] === undefined)) throw new InterfaceError("interface-output", f.name, `${where} did not return ${f.name}`);
    if (!fits(values[f.name], f)) {
      throw new InterfaceError("interface-output", f.name, `${where}: output ${f.name}: ${show(values[f.name])} does not fit ${JSON.stringify(dataShape(f.shape))}`);
    }
  }
  return Object.fromEntries(outputs.map((f) => [f.name, values[f.name]]));
}
