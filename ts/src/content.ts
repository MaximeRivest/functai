/**
 * What the log keeps of a call's values (contract/calls.md, "Content").
 * `logContent` is `true`, `false`, or a map from field names to booleans
 * (`"*"` for the fields it does not name). A value is written only when no
 * layer drops it: the program's own setting, each enclosing block,
 * `configure`, and `FUNCTAI_LOG_CONTENT` (`0` drops everything). A program
 * can ask for less, never for more.
 */

import { copyData, entriesOf, getOwn, recordOf, setOwn } from "./values.ts";

type Rec = Record<string, unknown>;

/** Which values the log may keep: all (`true`), none (`false`), or by field (`{ transcript: false }`, `{ "*": false, question: true }`). */
export type LogContent = boolean | Readonly<Record<string, boolean>>;

/** Where a setting was made: the program's own, a block (`withSettings`, a call's options), or `configure`. */
export type Where = "own" | "block" | "configure";

/** The fields of a call: its interface's inputs and outputs, and the outputs FunctAI added (`reasoning`, `calls`, also among `outputs`). */
export interface CallFields {
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly added: readonly string[];
}

/** A setting that cannot be taken: `code` says why, `field` names the key at fault. */
export class SettingError extends TypeError {
  readonly code: "log-content-field";
  readonly field: string;
  constructor(code: "log-content-field", field: string, message: string) {
    super(message);
    this.code = code;
    this.field = field;
    this.name = "SettingError";
  }
}

const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;
const OFF = new Set(["0", "false", "no", "off"]);

/**
 * Refuse a `logContent` setting (`log-content-field`): a key that is neither
 * a field name nor `"*"`, anywhere; and, in a program's own map (`fields`
 * given), a name that is not one of its fields.
 */
export function checkLogContent(value: unknown, where: string, fields?: CallFields): void {
  if (value === undefined || value === null || typeof value === "boolean") return;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${where}: logContent is true, false, or a map from field names to true or false`);
  }
  const names = fields ? [...fields.inputs, ...fields.outputs] : null;
  for (const [key, v] of Object.entries(value)) {
    if (key !== "*" && !NAME.test(key)) {
      throw new SettingError("log-content-field", key, `${where}: logContent key ${JSON.stringify(key)} is neither a field name nor "*"`);
    }
    if (names && key !== "*" && !names.includes(key)) {
      throw new SettingError("log-content-field", key,
        `${where}: logContent names ${key}, which is not one of its fields (${names.join(", ")}): it would be written`);
    }
    if (typeof v !== "boolean") throw new TypeError(`${where}: logContent.${key} is true or false`);
  }
}

/** Whether one layer's setting drops a field. */
function drops(v: LogContent, name: string): boolean {
  if (typeof v === "boolean") return !v;
  if (Object.hasOwn(v, name)) return !v[name];
  return v["*"] === false;
}

/** Whether `FUNCTAI_LOG_CONTENT` drops every field (`0`, `false`, `no`, `off`, in any case, white space around). */
export function environmentDrops(environment: string | null | undefined): boolean {
  return environment !== null && environment !== undefined && OFF.has(environment.trim().toLowerCase());
}

/**
 * For each field, whether its value is written: only when no layer drops it,
 * and, for a field FunctAI added, only when no field of the call is dropped.
 */
export function keptFields(fields: CallFields, layers: readonly LogContent[], environment: string | null | undefined): Record<string, boolean> {
  const off = environmentDrops(environment);
  const out: Record<string, boolean> = {};
  for (const name of [...fields.inputs, ...fields.outputs]) setOwn(out, name, !off && !layers.some((v) => drops(v, name)));
  if (!Object.values(out).every(Boolean)) for (const n of fields.added) setOwn(out, n, false);
  return out;
}

const ALWAYS_KEPT = new Set(["functai_call", "id", "parent", "root", "program", "started", "seconds", "sizes", "model", "usage",
  "confidence", "caller", "process", "saw", "escalated", "truncated", "journal"]);
const ERROR_KEPT = ["type", "code"];
const errorKept = (e: Rec) => Object.fromEntries(Object.entries(e).filter(([k]) => ERROR_KEPT.includes(k)));

/**
 * The record a call writes, from its whole record, when `keep` says which
 * fields are written (calls.md, "What the record keeps"). A record that does
 * not keep every value has `content: false` and `omitted`, keeps only the
 * written values, no exchange request, reply or request hash, and of each
 * error only its type and code.
 */
export function writtenRecord(record: Rec, fields: CallFields, keep: Record<string, boolean>): Rec {
  if (Object.values(keep).every(Boolean)) return copyData(record);
  const out: Rec = {};
  for (const [k, v] of Object.entries(record)) {
    if (ALWAYS_KEPT.has(k)) out[k] = copyData(v);
    if (k === "content") {
      out["content"] = false;
      out["omitted"] = { inputs: fields.inputs.filter((n) => getOwn(keep, n) !== true), outputs: fields.outputs.filter((n) => getOwn(keep, n) !== true) };
    }
  }
  const only = (values: Rec) => recordOf(entriesOf(values).filter(([k]) => getOwn(keep, k) === true));
  const answer = (record["program"] as Rec)["answer"] as string;
  const inputs = only((record["inputs"] ?? {}) as Rec);
  if (Object.keys(inputs).length) out["inputs"] = inputs;
  if (record["outputs"] === null || record["outputs"] === undefined) out["outputs"] = null;
  else {
    const outputs = only(record["outputs"] as Rec);
    if (Object.keys(outputs).length) out["outputs"] = outputs;
  }
  if (record["described"]) {
    const d = record["described"] as Record<string, string[]>;
    const described = Object.fromEntries(Object.entries(d).map(([k, v]) => [k, v.filter((n) => getOwn(keep, n) === true)]));
    if (Object.values(described).some((v) => v.length)) out["described"] = described;
  }
  if ("returned" in record && getOwn(keep, answer) === true) out["returned"] = record["returned"];
  if (record["probabilities"]) {
    const p = only(record["probabilities"] as Rec);
    if (Object.keys(p).length) out["probabilities"] = p;
  }
  out["error"] = record["error"] ? errorKept(record["error"] as Rec) : null;
  out["exchanges"] = ((record["exchanges"] ?? []) as Rec[]).map((ex) => {
    const kept: Rec = {};
    for (const [k, v] of Object.entries(ex)) if (!["request", "response", "request_hash"].includes(k)) kept[k] = copyData(v);
    if (kept["error"]) kept["error"] = errorKept(kept["error"] as Rec);
    return kept;
  });
  const order = Object.keys(record);
  order.splice(order.indexOf("content") + 1, 0, "omitted");
  const rank = (k: string) => (order.includes(k) ? order.indexOf(k) : order.length);
  return Object.fromEntries(Object.keys(out).sort((a, b) => rank(a) - rank(b)).map((k) => [k, out[k]]));
}
