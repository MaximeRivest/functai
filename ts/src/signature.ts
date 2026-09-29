/**
 * A definition becomes an lmcc signature (contract/functions.md, "The
 * signature"): the fields in their order, the instructions from the name,
 * the description and the guidance. And the sample input a version is
 * rendered with (contract/calls.md, "Versions").
 */

import * as lmcc from "lmcc";
import { trimWhite } from "./text.ts";
import { getOwn, setOwn } from "./values.ts";

type JsonObject = Record<string, unknown>;

export interface FieldDef {
  readonly name: string;
  /** Its shape; an optional input's holds its `default`, the value it is sent with when left out. */
  readonly shape: JsonObject;
  readonly desc?: string | null;
  /** Inputs only: a caller may leave it out. */
  readonly optional?: boolean;
}

/** A shape without its own `default` (functions.md: a default is how a call is bound, not what the model is told). */
const withoutDefault = (shape: JsonObject): JsonObject => {
  const { default: _default, ...rest } = shape;
  return rest;
};

/** A definition as data: what every language's AI function comes down to. */
export interface Definition {
  readonly name: string;
  readonly description: string;
  readonly inputs: readonly FieldDef[];
  /** In order; the last is the answer. */
  readonly outputs: readonly FieldDef[];
  readonly cot: boolean;
  readonly tools: boolean;
  readonly includeName: boolean;
  /** The instructions as another language wrote them (a loaded function): used as they are. */
  readonly written?: string;
}

export const TOOL_LIST: JsonObject = {
  type: "array", items: {
    type: "object", properties: {
      name: { type: "string" }, description: { anyOf: [{ type: "string" }, { type: "null" }] },
      parameters: { anyOf: [{ type: "object" }, { type: "null" }] },
    }, required: ["name", "description", "parameters"],
  },
};
export const CALL_LIST: JsonObject = {
  type: "array", items: {
    type: "object", properties: { id: { type: "string" }, name: { type: "string" }, input: { type: "object" } },
    required: ["id", "name", "input"],
  },
};

/** The instructions: an improved one as it is (trimmed), else name, description and guidance. */
export function instructions(d: Definition, improved: string | null | undefined): string {
  if (improved !== null && improved !== undefined) return trimWhite(improved);
  if (d.written !== undefined) return d.written;
  const head: string[] = [];
  if (d.includeName && d.name) head.push(`Function: ${d.name}`);
  const description = trimWhite(d.description);
  if (description) head.push(description);
  const top = trimWhite(head.join("\n\n"));
  const lines: string[] = [];
  const inputs = d.inputs.filter((f) => f.desc);
  if (inputs.length) lines.push("Parameter guidance:", ...inputs.map((f) => `- ${f.name}: ${f.desc}`), "");
  const outputs = d.outputs.filter((f) => f.desc);
  if (outputs.length) lines.push("Output guidance:", ...outputs.map((f) => `- ${f.name}: ${f.desc}`), "");
  const guidance = trimWhite(lines.join("\n"));
  if (!guidance) return top;
  return top + (top ? "\n\n" : "") + guidance;
}

/** The fields, in the contract's order. */
export function fields(d: Definition): lmcc.FieldInput[] {
  const out: lmcc.FieldInput[] = d.inputs.map((f) => ({
    name: f.name, direction: "input", shape: withoutDefault(f.shape) as lmcc.JsonObject, purpose: "plain", ...(f.desc ? { desc: f.desc } : {}),
  }));
  if (d.tools) out.push({ name: "tools", direction: "input", shape: TOOL_LIST as lmcc.JsonObject, purpose: "tools", type: "list[Tool]" });
  const names = new Set([...d.inputs.map((f) => f.name), ...d.outputs.map((f) => f.name)]);
  if (d.cot && !names.has("reasoning")) out.push({ name: "reasoning", direction: "output", shape: { type: "string" }, purpose: "reasoning" });
  if (d.tools) out.push({ name: "calls", direction: "output", shape: CALL_LIST as lmcc.JsonObject, purpose: "tools.calls", type: "list[ToolCall]" });
  for (const f of d.outputs) out.push({ name: f.name, direction: "output", shape: f.shape as lmcc.JsonObject, purpose: "plain" });
  return out;
}

export function signature(d: Definition, improved: string | null | undefined): lmcc.Signature {
  return new lmcc.Signature(instructions(d, improved), fields(d));
}

const BY_TYPE: Readonly<Record<string, unknown>> = { string: "example text", integer: 3, number: 2.5, boolean: true, array: [], object: {}, null: null };

/** A value for a shape, for the sample input. */
export function sample(shape: JsonObject): unknown {
  if (Array.isArray(shape["enum"])) return shape["enum"][0];
  if (Array.isArray(shape["anyOf"])) {
    const options = (shape["anyOf"] as JsonObject[]).filter((s) => s["type"] !== "null");
    return options.length ? sample(options[0]!) : null;
  }
  const t = shape["type"];
  return typeof t === "string" && Object.hasOwn(BY_TYPE, t) ? structuredClone(BY_TYPE[t]) : "example text";
}

/** The sample input: a value for each plain input (by own properties: `__proto__` is a field name like any other). */
export function sampleInputs(sig: lmcc.Signature): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of sig.fields) if (f.direction === "input" && f.purpose === "plain") setOwn(out, f.name, sample(f.shape as JsonObject));
  return out;
}

/**
 * A call's `program.signature` (contract/calls.md): lmcc's fingerprint with
 * every type name empty, so the shapes decide, not how a language spells types.
 */
export function signatureId(sig: lmcc.Signature): string {
  return lmcc.sha256(sig.fields.map((f) => ({ direction: f.direction, name: f.name, purpose: f.purpose || "plain", shape: f.shape, type: "" })));
}

/**
 * Values as their fields expect them: a non-text value given to a text input
 * is written as text. Only own properties are values: an inherited member
 * (`toString`) is never sent as an input. The record has no prototype (see
 * {@link ownRecord}).
 */
export function prepareInputs(sig: lmcc.Signature, values: Record<string, unknown>): Record<string, unknown> {
  const out = Object.create(null) as Record<string, unknown>;
  for (const f of sig.fields) {
    if (f.direction !== "input" || !Object.hasOwn(values, f.name)) continue;
    let v = getOwn(values, f.name);
    const shape = f.shape as JsonObject;
    if (shape["type"] === "string" && !Object.hasOwn(shape, "enum") && v !== null && v !== undefined && typeof v !== "string") {
      v = typeof v === "object" ? JSON.stringify(v, null, 2) : String(v);
    }
    setOwn(out, f.name, v);
  }
  return out;
}

/**
 * Field values as lmcc is given them: a record with no prototype, holding
 * the own properties of `values` that are not `undefined`. lmcc's
 * TypeScript renderer reads a turn's values with `name in values` and
 * `values[name]`; on an ordinary object both reach Object's members (a
 * worked example without its `toString` input would be given
 * `Object.prototype.toString`), and `__proto__` is the prototype, not a
 * value. On this record both read own properties only, and every name,
 * `__proto__` included, is a key like any other.
 */
export function ownRecord(values: Readonly<Record<string, unknown>>): Record<string, unknown> {
  const out = Object.create(null) as Record<string, unknown>;
  for (const k of Object.keys(values)) {
    const v = values[k];                              // an own key: read as own, `__proto__` included
    if (v !== undefined) out[k] = v;                  // no prototype: `out["__proto__"] = v` makes an own property
  }
  return out;
}

/**
 * A turn of this plan with these inputs (and, for a worked example, these
 * outputs), its values held as {@link ownRecord}s. Built with lmcc's `Turn`
 * rather than `plan.turn`/`plan.example`, which copy the values into
 * ordinary objects with `out[name] = value`; `render` checks the names as
 * those do.
 */
export function turnOf(plan: lmcc.Plan, inputs: Readonly<Record<string, unknown>>, outputs: Readonly<Record<string, unknown>> | null = null): lmcc.Turn {
  return new lmcc.Turn(plan.fingerprint, ownRecord(inputs), [], outputs === null ? null : ownRecord(outputs));
}

/** The turn a call sends, from its prepared values (and tools): refused first when lmcc would not write a value as given ({@link checkCarried}). */
export function currentTurn(plan: lmcc.Plan, values: Readonly<Record<string, unknown>>): lmcc.Turn {
  checkCarried(plan.signature, values, "input");
  return turnOf(plan, values);
}

/** A recorded turn (a worked example used as it is), its inputs, outputs and model steps' outputs held as {@link ownRecord}s. */
export function ownTurn(t: lmcc.Turn): lmcc.Turn {
  const steps = t.steps.map((s) => (s instanceof lmcc.ModelStep ? new lmcc.ModelStep(ownRecord(s.outputs), s.message, s.request, s.callsField) : s));
  return new lmcc.Turn(t.signature, ownRecord(t.inputs), steps, t.outputs === null ? null : ownRecord(t.outputs), t.score, t.meta);
}

/** Whether a value holds a member named `__proto__` in any object inside it (through `toJSON`, as lmcc writes it). */
function holdsProto(v: unknown, seen: Set<object>): boolean {
  if (v === null || typeof v !== "object" || seen.has(v)) return false;
  seen.add(v);
  if (Array.isArray(v)) return v.some((x) => holdsProto(x, seen));
  if (typeof (v as { toJSON?: unknown }).toJSON === "function") return holdsProto((v as { toJSON(): unknown }).toJSON(), seen);
  if (Object.hasOwn(v, "__proto__")) return true;
  return Object.values(v).some((x) => holdsProto(x, seen));
}

/**
 * Refuse (`format-write-error`, before anything is sent) a value lmcc would
 * write as other data than it is: a plain input's (or a worked example's
 * output's) value holding a member named `__proto__` inside it. lmcc's
 * TypeScript json format copies an object's members with `out[key] = value`
 * before writing it (`lower`, lmcc `ts/src/std/formats.ts`): that member
 * would be dropped, or become the copy's prototype, and the model would be
 * sent other data than the caller gave (Python sends it). Until lmcc carries
 * it, the call is refused rather than sent wrong. A text input's value is
 * text by then, and is not concerned; nor are the tools FunctAI adds, which
 * lmcc writes whole.
 */
export function checkCarried(sig: lmcc.Signature, values: Readonly<Record<string, unknown>>, direction: "input" | "output" = "input"): void {
  for (const f of sig.fields) {
    if (f.direction !== direction || f.purpose === "tools" || f.purpose === "tools.calls" || !Object.hasOwn(values, f.name)) continue;
    if (holdsProto(values[f.name], new Set())) {
      throw new lmcc.Refusal("format-write-error", `field ${JSON.stringify(f.name)}: its value holds a member named "__proto__", which lmcc's TypeScript json format cannot write yet (it would be dropped): nothing was sent`);
    }
  }
}
