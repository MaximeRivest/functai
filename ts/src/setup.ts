/**
 * What a program runs on (contract/calls.md, *Setups*; design/18).
 *
 * A version says what a program **is**: its instruction, examples, layout,
 * code. Its **setup** says what it runs on: which model answers each step,
 * with which settings and plugins, with which tool code, able to reach
 * what. Two runs of one version on two models have two setups, so a score,
 * a comparison or a release belongs to exactly one thing that ran.
 *
 * `fn.setup` is `"sha256:"` of the canonical JSON of `fn.setupDocument()`,
 * computed where the program runs, from the settings in force there, before
 * its first call and without the network. Every call records it
 * (`program.setup`), beside where it ran (`placement`).
 */

import * as lmcc from "lmcc";
import { ReachError } from "./errors.ts";
import type { Plugin } from "./plugins.ts";
import { mergedNeeds, wallOf, type Needs, type Settings } from "./settings.ts";
import type { Tool } from "./tools.ts";
import { builtin } from "./host.ts";
import { byCodePoint, copyData, parseData } from "./values.ts";

type Rec = Record<string, unknown>;

/** A setup document (contract/calls.md, *Setups*). */
export type SetupDocument = Rec;

/** A program whose setup can be worked out (an AI function, a module). */
export interface HasSetup {
  readonly name: string;
  readonly setup: string;
  setupDocument(): SetupDocument;
}

/** FunctAI's settings that change what the model is asked or what comes back, with their defaults. */
export const SETTINGS: readonly (readonly [string, keyof Settings, unknown])[] = [
  ["max_steps", "maxSteps", 8], ["retries", "retries", 1], ["tool_errors", "toolErrors", "report"], ["on_unreadable", "onUnreadable", "raise"],
];

/** `"sha256:"` of the canonical JSON of a document. */
export const setupHash = (doc: SetupDocument): string => lmcc.sha256(doc as lmcc.Json);

/** A plugin as a setup counts it: its name, version, and settings when it says them. */
export function pluginEntry(p: Plugin): Rec {
  return { name: p.name, version: p.version, ...(p.config !== undefined ? { config: copyData(p.config) } : {}) };
}

/** The program a tool stands for (`delegate(program)`), when it is one. */
export const programOfTool = (t: Tool): HasSetup | undefined => (t as { _program?: HasSetup })._program;

/** A tool as a setup counts it: the setup of the program it is, or its code (TypeScript: the function's text). */
export function toolEntry(t: Tool): Rec {
  const program = programOfTool(t);
  if (program) return { setup: program.setup };
  const id = (t as { codeId?: string }).codeId;
  return { code: { language: "typescript", code: "sha256:" + lmcc.sha256Hex(String(t.run)), ...(id !== undefined ? { id } : {}) } };
}

/** What a tool's code reaches (`needs`), or the program it stands for. */
export function toolNeeds(t: Tool): Needs | undefined {
  const program = programOfTool(t) as { _needs?: () => Needs | undefined } | undefined;
  if (program && typeof program._needs === "function") return program._needs();
  return (t as { needs?: Needs }).needs;
}

/** `"open"` with no wall; else what the wall lets through (or the needs), and its limits (calls.md, *Reach*). */
export function reachOf(needs: Needs | undefined, settings: Settings): unknown {
  if (settings.wall === undefined || settings.wall === null) return "open";
  const wall = wallOf(settings.wall);
  const pick = (k: "domains" | "secrets") => [...new Set(wall[k] ?? needs?.[k] ?? [])].sort(byCodePoint);
  return { domains: pick("domains"), secrets: pick("secrets"), ...(wall.limits !== undefined ? { limits: copyData(wall.limits) } : {}) };
}

const letsThrough = (granted: readonly string[], domain: string) =>
  granted.some((g) => g === domain || (g.startsWith("*.") && domain.endsWith(g.slice(1))));

/** Refuse a program that needs what its wall does not let through (`ReachError`, `needs-unmet`). */
export function checkReach(name: string, needs: Needs | undefined, settings: Settings): void {
  if (settings.wall === undefined || settings.wall === null) return;
  const wall = wallOf(settings.wall);
  if (wall.domains === undefined && wall.secrets === undefined) return;
  const missing = {
    domains: wall.domains === undefined ? [] : (needs?.domains ?? []).filter((d) => !letsThrough(wall.domains!, d)),
    secrets: wall.secrets === undefined ? [] : (needs?.secrets ?? []).filter((s) => !wall.secrets!.includes(s)),
  };
  if (missing.domains.length || missing.secrets.length) {
    const what = [...missing.domains, ...missing.secrets.map((s) => `the secret ${s}`)].join(", ");
    throw new ReachError(`${name} needs ${what}, which the wall ${JSON.stringify(wall.name)} does not let through`, missing);
  }
}

/** Where a call runs (its record's `placement`): the wall's name, and an AI function's server address. */
export function placementOf(settings: Settings, ai: boolean): Rec | null {
  const out: Rec = {};
  if (settings.wall !== undefined && settings.wall !== null) out["wall"] = wallOf(settings.wall).name;
  if (ai && typeof settings.baseUrl === "string") out["base_url"] = settings.baseUrl;
  return Object.keys(out).length ? out : null;
}

export { mergedNeeds };

// ------------------------------------------------------------------ what differs

/** One difference between two setups: the program, a dotted path into its document, and each side's value. */
export interface SetupDifference {
  readonly program: string;
  readonly field: string;
  readonly before: unknown;
  readonly after: unknown;
}

const isRec = (x: unknown): x is Rec => typeof x === "object" && x !== null && !Array.isArray(x);
const same = (a: unknown, b: unknown) => lmcc.canonicalJson((a ?? null) as lmcc.Json) === lmcc.canonicalJson((b ?? null) as lmcc.Json);

function differences(a: unknown, b: unknown, path: string, out: [string, unknown, unknown][]): void {
  if (isRec(a) && isRec(b)) {
    for (const k of [...new Set([...Object.keys(a), ...Object.keys(b)])].sort(byCodePoint)) {
      differences(Object.hasOwn(a, k) ? a[k] : undefined, Object.hasOwn(b, k) ? b[k] : undefined, path ? `${path}.${k}` : k, out);
    }
  } else if (!same(a, b)) out.push([path, a === undefined ? null : a, b === undefined ? null : b]);
}

/** Each program's setup document by key: a program here (and the AI functions it reaches), or a saved manifest's nodes. */
function documents(source: unknown): Record<string, SetupDocument> {
  if (typeof source === "function" && typeof (source as { setupDocument?: unknown }).setupDocument === "function") {
    const p = source as unknown as HasSetup & { _aiReached?: () => Record<string, HasSetup> };
    const out: Record<string, SetupDocument> = { [p.name]: p.setupDocument() };
    for (const [k, fn] of Object.entries(p._aiReached?.() ?? {}).sort(([x], [y]) => byCodePoint(x, y))) out[k] = fn.setupDocument();
    return out;
  }
  let manifest = source;
  if (typeof source === "string") {
    const fs = builtin("node:fs");
    if (!fs) throw new Error("reading a saved folder needs a file system; pass its manifest (functai.json, parsed)");
    const file = fs.statSync(source).isDirectory() ? `${source.replace(/\/$/, "")}/functai.json` : source;
    manifest = parseData(fs.readFileSync(file, "utf8"));
  }
  if (!isRec(manifest) || !isRec(manifest["nodes"])) throw new TypeError("setupDiff compares programs, saved folders, or their functai.json");
  const out: Record<string, SetupDocument> = {};
  for (const [key, node] of Object.entries(manifest["nodes"] as Rec)) {
    if (isRec(node) && isRec(node["setup"])) out[key === manifest["entry"] ? node["name"] as string : key] = node["setup"] as Rec;
  }
  return out;
}

/**
 * What differs between two setups, step by step: each `{ program, field,
 * before, after }`, `field` a dotted path into the setup document
 * (`model.model`, `config.temperature`, `plugins`). `a` and `b`: programs
 * here, saved folders, or their manifests.
 */
export function setupDiff(a: unknown, b: unknown): SetupDifference[] {
  const da = documents(a);
  const db = documents(b);
  const out: SetupDifference[] = [];
  for (const program of [...new Set([...Object.keys(da), ...Object.keys(db)])].sort(byCodePoint)) {
    const strip = (d: SetupDocument | undefined) => Object.fromEntries(Object.entries(d ?? {}).filter(([k]) => k !== "ai"));
    const found: [string, unknown, unknown][] = [];
    differences(strip(da[program]), strip(db[program]), "", found);
    for (const [field, before, after] of found) out.push({ program, field, before, after });
  }
  return out;
}
