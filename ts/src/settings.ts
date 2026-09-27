/**
 * Settings: where a call goes and how it behaves. A function's own settings
 * beat `withSettings(...)` blocks, which beat `configure(...)`, which beats
 * the defaults (the order Python uses).
 */

import type { Config, Request, Response, StreamEvent } from "@lm15/lm15";
import { Context } from "./host.ts";
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
  /** `false`: log sizes, times and tokens, never values or messages. */
  logContent?: boolean | null;
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
const scoped = new Context<Settings>();

/** Settings for every AI function (their own settings still win). Returns the settings now in force. */
export function configure(settings: Settings = {}): Settings {
  global = { ...global, ...settings };
  return { ...global };
}

/** Run `fn` with these settings over `configure`'s (their own settings still win). */
export function withSettings<R>(settings: Settings, fn: () => R): R {
  const outer = scoped.get() ?? {};
  // a caller adds to the enclosing block's (an evaluation inside an optimization is both)
  const caller = settings.caller ? { caller: { ...(outer.caller ?? {}), ...settings.caller } } : {};
  return scoped.run({ ...outer, ...settings, ...caller }, fn);
}

/** The settings a function with `own` settings runs with. */
export function effective(own: Settings): Settings & typeof DEFAULTS {
  const out: Record<string, unknown> = { ...DEFAULTS };
  for (const layer of [global, scoped.get() ?? {}, own]) {
    for (const [k, v] of Object.entries(layer)) if (v !== undefined) out[k] = v;
  }
  const caller = { ...(global.caller ?? {}), ...(scoped.get()?.caller ?? {}), ...(own.caller ?? {}) };
  out["caller"] = caller;
  return out as unknown as Settings & typeof DEFAULTS;
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
