/**
 * What a call saw (contract/calls.md, "Saw"): the earlier calls it was
 * given as context, read from the call log, and whether the log keeps what
 * showing them again needs. Knowing is not replaying: `keepsSaw` proves
 * retention, never shows anything.
 *
 * ```ts
 * sawOf(records, callId);     // [{ call: "…" }, { call: "…", without: ["photo"] }]
 * keepsSaw(records, callId);  // { ok: true } or { refuses: "not-kept", call: "…" }
 * ```
 */

import type { SawEntry } from "./events.ts";
import { copyData, entriesOf, recordOf } from "./values.ts";

type Rec = Record<string, unknown>;

/** Why what a call saw cannot be known, or shown again. */
export type SawCode = "not-recorded" | "missing-call" | "unknown-key" | "saw-cycle" | "not-kept" | "turn-invalid";

/** What a call saw cannot be known: `code` says why, `call` names the record that says so. */
export class SawUnknown extends Error {
  readonly code: SawCode;
  readonly call: string;
  constructor(code: SawCode, call: string) {
    super(`${code}: what call ${call} saw cannot be known`);
    this.name = "SawUnknown";
    this.code = code;
    this.call = call;
  }
}

const ENTRY_KEYS = new Set(["call", "steps", "without", "slot"]);
const CALLS = "calls";

const byId = (records: Iterable<Rec> | ReadonlyMap<string, Rec>): ReadonlyMap<string, Rec> =>
  records instanceof Map ? records : new Map([...(records as Iterable<Rec>)].filter((r) => "functai_call" in r).map((r) => [r["id"] as string, r]));

function expand(records: ReadonlyMap<string, Rec>, call: string, following: readonly string[]): SawEntry[] {
  const rec = records.get(call);
  if (rec === undefined || !("saw" in rec)) throw new SawUnknown(following.length ? "missing-call" : "not-recorded", call);
  const out: SawEntry[] = [];
  (rec["saw"] as Rec[]).forEach((entry, i) => {
    if (typeof entry === "object" && entry !== null && "saw_of" in entry) {
      if (i !== 0 || Object.keys(entry).length !== 1) throw new SawUnknown("unknown-key", call);
      const target = entry["saw_of"] as string;
      if (following.includes(target) || target === call) throw new SawUnknown("saw-cycle", target);
      out.push(...expand(records, target, [...following, call]));
    } else {
      if (typeof entry !== "object" || entry === null || !("call" in entry) || Object.keys(entry).some((k) => !ENTRY_KEYS.has(k))) {
        throw new SawUnknown("unknown-key", call);
      }
      out.push(copyData(entry) as SawEntry);
    }
  });
  return out;
}

/**
 * The calls a call saw, in order, each `saw_of` replaced (recursively) by the
 * entries of the call it names. Throws `SawUnknown` when they cannot be
 * known: `not-recorded` (its record has no `saw`), `missing-call`,
 * `unknown-key` (an entry no reader knows), `saw-cycle`.
 */
export function sawOf(records: Iterable<Rec> | ReadonlyMap<string, Rec>, call: string): SawEntry[] {
  return expand(byId(records), call, []);
}

/** Whether a call's record keeps what the entry says it was shown, as data. */
function hasValues(rec: Rec, entry: Rec): boolean {
  if (rec["truncated"]) return false;
  const leftOut = new Set((entry["without"] ?? []) as string[]);
  const sizes = (rec["sizes"] ?? {}) as { inputs?: Rec; outputs?: Rec };
  const shown = [...Object.keys(sizes.inputs ?? {}), ...Object.keys(sizes.outputs ?? {})].filter((n) => !leftOut.has(n));
  const described = (rec["described"] ?? { inputs: [], outputs: [] }) as { inputs: string[]; outputs: string[] };
  if (shown.some((n) => described.inputs.includes(n) || described.outputs.includes(n))) return false;
  if (entry["steps"]) {
    return rec["content"] === true && ((rec["exchanges"] ?? []) as Rec[]).every((ex) =>
      "request_hash" in ex && ("response" in ex || ex["finish"] === null || ex["finish"] === undefined));
  }
  if (rec["content"] === true) return true;
  const omitted = rec["omitted"] as { inputs: string[]; outputs: string[] } | undefined;
  if (!omitted) return false;                                   // format 1, or no value kept
  return [...omitted.inputs, ...omitted.outputs].every((n) => leftOut.has(n));
}

/**
 * Whether the log keeps what showing a call its context again needs (calls.md,
 * "Knowing is not replaying"): every call it saw is in the log, not
 * truncated, with the values it was shown as data (`missing-call`,
 * `not-kept`); with steps, each exchange's request hash and reply; an entry
 * no call can have been shown refuses `turn-invalid`.
 */
export function keepsSaw(records: Iterable<Rec> | ReadonlyMap<string, Rec>, call: string): { ok: true } | { refuses: SawCode; call: string } {
  const by = byId(records);
  let entries: SawEntry[];
  try {
    entries = expand(by, call, []);
  } catch (err) {
    if (err instanceof SawUnknown) return { refuses: err.code, call: err.call };
    throw err;
  }
  for (const e of entries as Rec[]) {
    const id = e["call"] as string;
    if (e["steps"] && ((e["without"] ?? []) as string[]).includes(CALLS)) return { refuses: "turn-invalid", call: id };
    const rec = by.get(id);
    if (rec === undefined) return { refuses: "missing-call", call: id };
    if (!hasValues(rec, e)) return { refuses: "not-kept", call: id };
  }
  return { ok: true };
}

// ------------------------------------------------------------------ rows that keep their context (calls.md, stage 5)

/** The turn a `saw` entry stands for, as data: the record's values without the entry's `without`; with `steps`, its steps and signature. */
function turnOf(rec: Rec, entry: Rec): Rec {
  const leftOut = new Set((entry["without"] ?? []) as string[]);
  const pick = (o: unknown) => recordOf(entriesOf((o ?? {}) as Rec).filter(([k]) => !leftOut.has(k)));
  const out: Rec = { inputs: pick(rec["inputs"]), outputs: pick(rec["outputs"]) };
  if (entry["steps"]) {
    if (!Array.isArray(rec["steps"])) throw new SawUnknown("not-kept", rec["id"] as string);
    out["steps"] = copyData(rec["steps"]);
    out["signature"] = ((rec["program"] ?? {}) as Rec)["signature"] ?? null;
  }
  return out;
}

const startedOrder = (r: Rec) => `${r["started"] ?? ""}\u0000${r["id"] ?? ""}`;

function under(rec: Rec, ancestor: string, by: ReadonlyMap<string, Rec>): boolean {
  const seen = new Set<string>();
  let parent = rec["parent"] as string | null | undefined;
  while (parent && !seen.has(parent)) {
    if (parent === ancestor) return true;
    seen.add(parent);
    parent = (by.get(parent)?.["parent"] ?? null) as string | null;
  }
  return false;
}

function checkKept(by: ReadonlyMap<string, Rec>, call: string): void {
  const kept = keepsSaw(by, call);
  if ("refuses" in kept) throw new SawUnknown(kept.refuses, kept.call);
}

/**
 * What a rated call was shown before its own inputs, as data a row carries
 * (calls.md, "Rows that keep their context"): `earlier` (the turns it was
 * shown, in order), `conversation` (its conversation's id, or null),
 * `sections` (what its conversation's context hooks gave it), and for a
 * module's call `helpers` (each call inside it that was shown earlier turns,
 * in the order they started). Throws `SawUnknown` when the log cannot show
 * them again: never a part.
 */
export function earlierOf(records: Iterable<Rec> | ReadonlyMap<string, Rec>, call: string): Rec {
  const by = byId(records);
  const rec = by.get(call);
  if (!rec) throw new SawUnknown("missing-call", call);
  let earlier: Rec[] = [];
  if (Array.isArray(rec["saw"]) && rec["saw"].length) {
    checkKept(by, call);
    earlier = (expand(by, call, []) as unknown as Rec[]).map((e) => turnOf(by.get(e["call"] as string)!, e));
  }
  const out: Rec = { earlier, conversation: ((rec["conversation"] ?? {}) as Rec)["id"] ?? null };
  if (Array.isArray(rec["sections"]) && rec["sections"].length) out["sections"] = [...rec["sections"] as string[]];
  if (((rec["program"] ?? {}) as Rec)["kind"] === "module") {
    const inside = [...by.values()].filter((c) => c["root"] === rec["root"] && c["id"] !== call && under(c, call, by)
      && ((Array.isArray(c["saw"]) && c["saw"].length) || (Array.isArray(c["sections"]) && c["sections"].length)));
    inside.sort((a, b) => (startedOrder(a) < startedOrder(b) ? -1 : startedOrder(a) > startedOrder(b) ? 1 : 0));
    out["helpers"] = inside.map((c) => {
      checkKept(by, c["id"] as string);
      const helper: Rec = {
        program: ((c["program"] ?? {}) as Rec)["name"] ?? null, call: c["id"],
        earlier: (expand(by, c["id"] as string, []) as unknown as Rec[]).map((e) => turnOf(by.get(e["call"] as string)!, e)),
      };
      if (Array.isArray(c["sections"]) && c["sections"].length) helper["sections"] = [...c["sections"] as string[]];
      return helper;
    });
  }
  return out;
}

/** Whether a rated call was shown anything before its inputs: itself, or, for a module, a call inside it. */
export function needsContext(rec: Rec | undefined, by: ReadonlyMap<string, Rec>): boolean {
  if (!rec) return false;
  const has = (r: Rec) => (Array.isArray(r["saw"]) && r["saw"].length > 0) || (Array.isArray(r["sections"]) && r["sections"].length > 0);
  if (has(rec) || rec["conversation"]) return true;
  if (((rec["program"] ?? {}) as Rec)["kind"] !== "module") return false;
  return [...by.values()].some((c) => c["root"] === rec["root"] && has(c) && under(c, rec["id"] as string, by));
}

export { byId as recordsById };

/**
 * The turn a `saw` entry stands for, shown again (calls.md, "The turn an
 * entry stands for"): every field in `without` taken out of its inputs,
 * outputs and each model step's outputs; a model step whose outputs held one
 * loses its recorded message (it would show the field); without `steps`, its
 * inputs and outputs only. With steps, leaving out the field that holds a
 * step's tool calls is refused (`turn-invalid`): tool steps would answer no call.
 */
export function shownTurn(turn: Rec, entry: Rec): { slot: string; turn: Rec } | { refuses: "turn-invalid" } {
  const left = new Set((entry["without"] ?? []) as string[]);
  const drop = (x: unknown) => recordOf(entriesOf((x ?? {}) as Rec).filter(([k]) => !left.has(k)));
  const out: Rec = { signature: turn["signature"], inputs: drop(turn["inputs"]) };
  if (entry["steps"]) {
    const steps = (turn["steps"] ?? []) as Rec[];
    if (steps.some((st) => typeof st["calls_field"] === "string" && left.has(st["calls_field"]))) return { refuses: "turn-invalid" };
    out["steps"] = steps.map((st) => {
      if (st["kind"] !== "model") return copyData(st);
      const held = Object.keys((st["outputs"] ?? {}) as Rec).some((k) => left.has(k));
      const copy: Rec = { ...copyData(st), outputs: drop(st["outputs"]) };
      if (held) delete copy["message"];
      return copy;
    });
  } else out["steps"] = [];
  out["outputs"] = turn["outputs"] === null || turn["outputs"] === undefined ? null : drop(turn["outputs"]);
  return { slot: (entry["slot"] ?? "turns") as string, turn: out };
}
