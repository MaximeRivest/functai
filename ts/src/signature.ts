/**
 * A definition becomes an lmcc signature (contract/functions.md, "The
 * signature"): the fields in their order, the instructions from the name,
 * the description and the guidance. And the sample input a version is
 * rendered with (contract/calls.md, "Versions").
 */

import * as lmcc from "lmcc";
import { trimWhite } from "./text.ts";

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

/** A value for a shape, for the sample input. */
export function sample(shape: JsonObject): unknown {
  if (Array.isArray(shape["enum"])) return shape["enum"][0];
  if (Array.isArray(shape["anyOf"])) {
    const options = (shape["anyOf"] as JsonObject[]).filter((s) => s["type"] !== "null");
    return options.length ? sample(options[0]!) : null;
  }
  const byType: Record<string, unknown> = { string: "example text", integer: 3, number: 2.5, boolean: true, array: [], object: {}, null: null };
  const t = shape["type"];
  return typeof t === "string" && t in byType ? structuredClone(byType[t]) : "example text";
}

/** The sample input: a value for each plain input. */
export function sampleInputs(sig: lmcc.Signature): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of sig.fields) if (f.direction === "input" && f.purpose === "plain") out[f.name] = sample(f.shape as JsonObject);
  return out;
}

/**
 * A call's `program.signature` (contract/calls.md): lmcc's fingerprint with
 * every type name empty, so the shapes decide, not how a language spells types.
 */
export function signatureId(sig: lmcc.Signature): string {
  return lmcc.sha256(sig.fields.map((f) => ({ direction: f.direction, name: f.name, purpose: f.purpose || "plain", shape: f.shape, type: "" })));
}

/** Values as their fields expect them: a non-text value given to a text input is written as text. */
export function prepareInputs(sig: lmcc.Signature, values: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of sig.fields) {
    if (f.direction !== "input" || !(f.name in values)) continue;
    let v = values[f.name];
    const shape = f.shape as JsonObject;
    if (shape["type"] === "string" && !("enum" in shape) && v !== null && v !== undefined && typeof v !== "string") {
      v = typeof v === "object" ? JSON.stringify(v, null, 2) : String(v);
    }
    out[f.name] = v;
  }
  return out;
}
