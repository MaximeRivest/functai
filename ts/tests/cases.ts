/** Reading the contract's cases (../../contract/cases, README.md there), and defining what they describe the way a TypeScript user would. */

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { ai, type Tool } from "../src/index.ts";
import { CONTRACT } from "../tools/generate.ts";

export type Rec = Record<string, any>;

/** Every case of a folder, by file name (without `.json`), in order; `prefix` keeps one kind of `events/`. */
export const cases = (folder: string, prefix = ""): [string, Rec][] =>
  readdirSync(join(CONTRACT, "cases", folder)).filter((f) => f.endsWith(".json") && f.startsWith(prefix)).sort()
    .map((f) => [f.replace(/\.json$/, ""), JSON.parse(readFileSync(join(CONTRACT, "cases", folder, f), "utf8"))]);

/** A contract definition (functions.md, "A definition") written the way a TypeScript user writes it. */
export function tsFunction(d: Rec) {
  const field = (f: Rec, input: boolean) => {
    const words = f.desc ? { desc: f.desc } : {};
    return input ? { shape: f.shape, ...words, optional: Boolean(f.optional) } : f.desc ? { shape: f.shape, ...words } : f.shape;
  };
  const tools: Tool[] = (d.tools ?? []).map((t: Rec) => ({ ...t, run: () => "" }));
  const settings: Rec = {};
  if (d.settings.adapter) settings.adapter = d.settings.adapter;
  if (d.settings.module) settings.module = d.settings.module;
  if (d.settings.include_fn_name_in_instructions === false) settings.includeFnName = false;
  const fn = ai(d.name, {
    description: d.description,
    input: Object.fromEntries(d.inputs.map((f: Rec) => [f.name, field(f, true)])),
    outputs: Object.fromEntries(d.outputs.map((f: Rec) => [f.name, field(f, false)])),
    ...(tools.length ? { tools } : {}), ...settings,
  } as never);
  fn.loadState(d.state);
  return fn;
}

/** A position, as the cases write one. */
export const pos = (e: Rec) => ({ writer: e.writer, seq: e.seq });

/** The contract's stand-in for a value with no JSON form, `{ $type, $repr }`, and a native value for it. */
export class Native {
  readonly $type: string;
  readonly $repr: string;
  constructor(d: Rec) {
    this.$type = d.$type;
    this.$repr = d.$repr;
  }
  toString() {
    return this.$repr;
  }
}
export const isStandIn = (v: unknown): v is Rec => typeof v === "object" && v !== null && !Array.isArray(v)
  && Object.keys(v).length === 2 && "$type" in v && "$repr" in v;
/** A case's value as the program gets it: stand-ins become native values. */
export const native = (v: unknown): unknown => (isStandIn(v) ? new Native(v) : v);
/** A program's value as the case writes it: native values become their stand-ins. */
export const described = (v: unknown): unknown => (v instanceof Native ? { $type: v.$type, $repr: v.$repr } : v);
