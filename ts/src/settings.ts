/**
 * Settings: where a call goes and how it behaves. A function's own settings
 * beat a call's options, which beat `withSettings(...)` blocks, which beat
 * `configure(...)`, which beats the defaults (the order Python uses). Three
 * settings are policy, and combine over every layer instead: `logContent`
 * (a value is written only when no layer drops it), `observers` (they add
 * up) and `journal` (a program cannot replace or remove a host's, and no
 * closer layer a required one).
 */

import type { Config, Request, Response, StreamEvent } from "@lm15/lm15";
import { checkLogContent, type LogContent, type Where } from "./content.ts";
import { Context } from "./host.ts";
import type { Journal, Observer } from "./log.ts";
import type { Capabilities } from "./models.ts";
import type { ReplyCache } from "./cache.ts";

/** What calls go through: an lm15 `LMRouter`, or anything with its `resolve` and `complete` (a fake, in tests). */
export interface Router {
  resolve(model: string): { provider: string; model: string };
  complete(request: Request, opts?: { signal?: AbortSignal }): Promise<Response>;
  stream?(request: Request, opts?: { signal?: AbortSignal }): AsyncIterable<StreamEvent>;
}

export interface Settings {
  /** The model: `"gpt-4.1-mini"`, `"claude-haiku-4-5"`, `"groq:openai/gpt-oss-120b"`, … */
  lm?: string | null;
  /** The lm15 router calls go through (keys, base URLs, a fake for tests). */
  router?: Router | null;
  temperature?: number | null;
  maxTokens?: number | null;
  topP?: number | null;
  stop?: readonly string[] | null;
  seed?: number | null;
  /** Any other lm15 Config fields. */
  config?: Partial<Config> | null;
  /** The layout: `"xml"` (default), `"chat"`, `"json"`, an lmcc adapter, or an artifact. */
  adapter?: unknown;
  /** A chat template of lmcc messages, instead of a named layout. */
  template?: readonly Record<string, unknown>[] | null;
  /** `"predict"` (default) or `"cot"`: reasoning before the answer. */
  module?: "predict" | "cot" | null;
  includeFnName?: boolean;
  capabilities?: Capabilities | null;
  /** Re-asks after an unreadable reply (default 1). */
  retries?: number;
  /** Re-sends after a transient provider error (default 3). */
  apiRetries?: number;
  /** Model calls per tool loop (default 8). */
  maxSteps?: number;
  /** `"report"` (the model sees a tool's error) or `"raise"`. */
  toolErrors?: "report" | "raise";
  /** The call log: a folder, `true` (the default folder), or `false`. Unset: `FUNCTAI_LOG_CALLS` decides. */
  logCalls?: string | boolean | null;
  /**
   * Which values the call log and the kept form of events keep: `true`,
   * `false` (sizes, times and tokens only), or by field (`{ transcript: false }`;
   * `{ "*": false, question: true }` keeps only the question). It only ever
   * removes: a value is written only when no layer drops it, and
   * `FUNCTAI_LOG_CONTENT=0` drops every value.
   */
  logContent?: LogContent | null;
  /**
   * Functions given the kept form of every event of the calls in scope, as it
   * happens (a page, telemetry). They add up over the layers, and never slow
   * a call: one that throws is warned about once and given no more events.
   */
  observers?: readonly Observer[] | null;
  /**
   * Where each call tree's kept log is written while it runs: a store
   * (best effort: the call never waits), or `{ store, mode: "required" }`
   * (the call waits until its events are kept, before its code runs, before
   * each tool and before it returns). `null`: none. A program's own setting
   * cannot replace or remove a host's journal (`JournalError`
   * `journal-policy`).
   */
  journal?: Journal | null;
  /** Who is calling, added to `FUNCTAI_CALLER`: `{ kind: "agent", conversation: "…" }`. */
  caller?: Record<string, unknown> | null;
  /**
   * Answer an identical request with the reply it got before (`true`: in this
   * process's memory; or a store: a `Map`, Redis, …). Off by default. It
   * returns identical samples too: leave it off where you want different ones.
   */
  cacheReplies?: boolean | ReplyCache | null;
}

export const DEFAULTS: Required<Pick<Settings, "retries" | "apiRetries" | "maxSteps" | "toolErrors" | "includeFnName">> = {
  retries: 1, apiRetries: 3, maxSteps: 8, toolErrors: "report", includeFnName: true,
};

let global: Settings = {};
/** The blocks in force, outermost first. */
const scoped = new Context<readonly Settings[]>();

/**
 * Settings for every program (their own settings still win). Returns the
 * settings now in force. A `logContent` key that is not a field name refuses
 * here (`SettingError`, `log-content-field`).
 */
export function configure(settings: Settings = {}): Settings {
  checkLogContent(settings.logContent, "configure");
  global = { ...global, ...settings };
  return { ...global };
}

/** Run `fn` with these settings over `configure`'s (a program's own settings and a call's options still win). */
export function withSettings<R>(settings: Settings, fn: () => R): R {
  checkLogContent(settings.logContent, "withSettings");
  return scoped.run([...(scoped.get() ?? []), settings], fn);
}

/** The settings a program with `own` settings runs with (a call's options are part of `own`). */
export function effective(own: Settings): Settings & typeof DEFAULTS {
  const out: Record<string, unknown> = { ...DEFAULTS };
  const blocks = scoped.get() ?? [];
  for (const layer of [global, ...blocks, own]) {
    for (const [k, v] of Object.entries(layer)) if (v !== undefined) out[k] = v;
  }
  // a caller adds to the enclosing ones' (an evaluation inside an optimization is both)
  out["caller"] = Object.assign({}, global.caller ?? {}, ...blocks.map((b) => b.caller ?? {}), own.caller ?? {});
  return out as unknown as Settings & typeof DEFAULTS;
}

/** One layer of settings around a call, and where it was set. */
export interface Layer {
  readonly where: Where;
  readonly settings: Settings;
}

/**
 * The layers around a call, closest first: the program's own settings, the
 * call's options (a block around this one call), the blocks in force
 * (closest first), and `configure`.
 */
export function layersOf(own: Settings, call: Settings = {}): Layer[] {
  const blocks = [...(scoped.get() ?? [])].reverse();
  return [
    { where: "own", settings: own },
    ...(Object.keys(call).length ? [{ where: "block" as const, settings: call }] : []),
    ...blocks.map((settings) => ({ where: "block" as const, settings })),
    { where: "configure", settings: global },
  ];
}

/** The lm15 Config fields of the settings. */
export function configOf(s: Settings, overrides: Partial<Config> = {}): Config | undefined {
  const c: Record<string, unknown> = { ...(s.config ?? {}) };
  if (s.temperature !== undefined && s.temperature !== null) c["temperature"] = s.temperature;
  if (s.maxTokens !== undefined && s.maxTokens !== null) c["maxTokens"] = s.maxTokens;
  if (s.topP !== undefined && s.topP !== null) c["topP"] = s.topP;
  if (s.stop !== undefined && s.stop !== null && s.stop.length) c["stop"] = [...s.stop];
  if (s.seed !== undefined && s.seed !== null) c["seed"] = s.seed;
  Object.assign(c, overrides);
  return Object.keys(c).length ? (c as Config) : undefined;
}
