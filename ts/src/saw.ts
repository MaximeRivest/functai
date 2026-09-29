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
import { copyData } from "./values.ts";

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
