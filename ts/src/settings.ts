/**
 * Settings: where a call goes and how it behaves. A call's options beat the
 * program's own settings (`ai(...)`, `.using(...)`), which beat
 * `withSettings(...)` blocks (the closest first), which beat
 * `configure(...)`, which beats the defaults.
 *
 * Three settings are policy, and combine over every layer instead
 * (contract/calls.md, "Content"; streaming.md, "Keeping a log while it is
 * written"): `logContent` (a value is written only when no layer drops it),
 * `observers` (they add up) and `journal` (a program's own setting cannot
 * replace or remove a host's journal, and no closer layer a required one).
 * For those, the layers are, closest first: the program's own settings, a
 * call's options (a block around that one call), the blocks, `configure`.
 *
 * A setting that could only fail later refuses where it is set: a
 * `logContent` key that is not a field name (`SettingError`), an observer
 * that is neither a function nor has `postMessage`, a journal that is not a
 * store (`TypeError`).
 */

import type { Config, Request, Response, StreamEvent } from "@lm15/lm15";
import { checkLogContent, type CallFields, type LogContent, type Where } from "./content.ts";
import { Context } from "./host.ts";
import { checkObservers, journalOf, type Journal, type Observer } from "./log.ts";
import type { Capabilities } from "./models.ts";
import { checkCacheSetting, type CacheSetting } from "./replies.ts";
import { checkApprove, type ApproveSetting } from "./tools.ts";
import { checkPlugins, type Plugin } from "./plugins.ts";
import { byCodePoint, copyData } from "./values.ts";

/** What calls go through: an lm15 `LMRouter`, or anything with its `resolve` and `complete` (a fake, in tests). */
export interface Router {
  resolve(model: string): { provider: string; model: string };
  complete(request: Request, opts?: { signal?: AbortSignal }): Promise<Response>;
  stream?(request: Request, opts?: { signal?: AbortSignal }): AsyncIterable<StreamEvent>;
}

/** A wall programs run behind, as the host says it (calls.md, *Reach*). */
export type Wall = string | { readonly name: string; readonly domains?: readonly string[]; readonly secrets?: readonly string[]; readonly limits?: Readonly<Record<string, unknown>> };

export interface Settings {
  /** The model: `"gpt-4.1-mini"`, `"claude-haiku-4-5"`, `"groq:openai/gpt-oss-120b"`, … */
  lm?: string | null;
  /** The lm15 router calls go through (keys, base URLs, a fake for tests). It wins over `apiKey` and `baseUrl`. */
  router?: Router | null;
  /** A key for the provider `lm` routes to: it beats saved logins and the environment. Never saved with a function. */
  apiKey?: string | null;
  /**
   * Where that provider is reached: a server of your own that speaks its API
   * (`lm: "openai:qwen3"` with `baseUrl: "http://localhost:8000/v1"` for vLLM or
   * Ollama), a gateway. Saved with a function, as its model is.
   */
  baseUrl?: string | null;
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
  /**
   * What a reply that still cannot be read once the re-asks are spent does:
   * `"raise"` (the default) ends the call with its refusal; `"record"` ends it
   * well with no answer (`undefined`; `prediction.refusal` says why), recorded
   * with `outputs: {}` and the refusal as its error, and a conversation goes on
   * past it. For runs that must not stop at one bad reply. A provider error
   * still raises.
   */
  onUnreadable?: "raise" | "record";
  /**
   * The wall the host runs programs behind (calls.md, *Reach*): a name
   * (`"srt"`: made from each program's needs), or `{ name, domains?, secrets?,
   * limits? }` (what it lets through, and its limits). Unset: none it says,
   * so a program's reach is open. A program that needs what the wall does
   * not let through is refused before its code runs (`ReachError`).
   */
  wall?: Wall | null;
  /**
   * What identifies code a program runs that its version cannot read
   * (calls.md, *Code the language cannot read*): a TypeScript version holds a
   * function's own text, not what it imports. A host that knows the files it
   * bundled a program from says so here (a bundle's hash, a revision): every
   * setup worked out under it holds it, so changed code there is another setup.
   * Text, at most 200 characters.
   */
  codeId?: string | null;
  /** Re-sends after a transient provider error (default 3). */
  apiRetries?: number;
  /** The longest a provider may ask to wait before a re-send, in seconds (default 60). A longer
   * `retryAfter` (a spent subscription window) raises at once, saying until when. */
  apiMaxWait?: number;
  /** Model calls per tool loop (default 8). */
  maxSteps?: number;
  /** `"report"` (the model sees a tool's error) or `"raise"`. */
  toolErrors?: "report" | "raise";
  /** The call log: a folder, `true` (the default folder), or `false`. Unset: `FUNCTAI_LOG_CALLS` decides. */
  /** A folder, `true` (the default folder), `false` (never), or a store of one's own (`{ append, read }`: calls.md). */
  logCalls?: string | boolean | import("./calllog.ts").CallStore | null;
  /**
   * Which values the call log and the kept form of events keep: `true`,
   * `false` (sizes, times and tokens only), or by field (`{ transcript: false }`;
   * `{ "*": false, question: true }` keeps only the question). It only ever
   * removes: a value is written only when no layer drops it, and
   * `FUNCTAI_LOG_CONTENT=0` drops every value.
   */
  logContent?: LogContent | null;
  /**
   * Receivers of the kept form of every event of the calls in scope (a
   * page, telemetry). They add up over the layers (`configure`'s replaces
   * `configure`'s: keep your own list to add one). A function is called
   * soon after each event, off the call's own turn, from a bounded queue
   * (an observer 10,000 events behind loses events); it still runs on this
   * thread, so heavy work belongs in a `Worker`: an object with
   * `postMessage` is posted each event. One that throws is warned about
   * once and given no more events.
   */
  observers?: readonly Observer[] | null;
  /**
   * Where each call tree's kept log is written while it runs: a store
   * (best effort: the call never waits), or `{ store, mode: "required" }`
   * (the call waits until its events are kept, before its code runs, before
   * each tool and before it returns; at most `timeout` ms, 30 s by default).
   * `null`: none. A program's own setting cannot replace or remove a host's
   * journal (`JournalError` `journal-policy`).
   */
  journal?: Journal | null;
  /**
   * `false` (a host's block or `configure`): the observers a program sets for
   * itself are given no event; the host's still are. It only removes: a
   * program's own `true` does not undo it.
   */
  programObservers?: boolean | null;
  /** Who is calling, added to `FUNCTAI_CALLER`: `{ kind: "agent", conversation: "…" }`. */
  caller?: Record<string, unknown> | null;
  /**
   * Answer an identical request with the reply it got before (`true`: in this
   * process's memory; or a store: a `Map`, Redis, …). Off by default. It
   * returns identical samples too: leave it off where you want different ones.
   */
  cacheReplies?: CacheSetting;
  /** The n-th independent answer to the same request (default 0): part of the reply cache's key, and nothing else. */
  replicate?: number | null;
  /**
   * Who answers a tool call before it runs (contract/tools.md): a function
   * asked at once, or a rule a person answers later (`"changes"`, `"all"`, a
   * list of tool names or approval paths). No approval by default.
   */
  approve?: ApproveSetting | null;
  /** Plugins around the calls in scope (contract/plugins.md): they add up over the layers, the program's own first, the host's last. */
  plugins?: readonly Plugin[] | null;
  /** `false` (a host's layer): the program's own plugins do not run. */
  programPlugins?: boolean | null;
  /**
   * When the first model is less sure of its answer than `escalateBelow`
   * (default 0.9), another answers instead: a model, or an AI function. The
   * first needs to measure its confidence (a baked model, TypeSafe's Jev,
   * `config: { probabilities: "required" }`).
   */
  escalateTo?: string | object | null;
  escalateBelow?: number | null;
}

export const DEFAULTS: Required<Pick<Settings, "retries" | "apiRetries" | "apiMaxWait" | "maxSteps" | "toolErrors" | "includeFnName">> = {
  retries: 1, apiRetries: 3, apiMaxWait: 60, maxSteps: 8, toolErrors: "report", includeFnName: true,
};

let global: Settings = {};
/** The blocks in force, outermost first. */
const scoped = new Context<readonly Settings[]>();

/**
 * Refuse, where they are set, the policy settings that could only fail later:
 * `logContent` (`SettingError` `log-content-field`: a key that is not a
 * field name or `"*"`; with `fields`, a program's own map naming a field it
 * does not have), `observers` and `journal` (`TypeError`).
 */
export function checkSettings(settings: Settings | null | undefined, where: string, fields?: CallFields): void {
  if (!settings) return;
  checkLogContent(settings.logContent, where, fields);
  checkObservers(settings.observers, where);
  if (settings.programObservers !== undefined && settings.programObservers !== null && typeof settings.programObservers !== "boolean") {
    throw new TypeError(`${where}: programObservers is true or false`);
  }
  if (settings.journal !== undefined) journalOf(settings.journal, where);
  checkCacheSetting(settings.cacheReplies, where);
  checkApprove(settings.approve, where);
  checkPlugins(settings.plugins, where);
  if (settings.replicate !== undefined && settings.replicate !== null && !(Number.isInteger(settings.replicate) && settings.replicate >= 0)) {
    throw new TypeError(`${where}: replicate is a whole number, 0 or more`);
  }
  if (settings.onUnreadable !== undefined && settings.onUnreadable !== "raise" && settings.onUnreadable !== "record") {
    throw new TypeError(`${where}: onUnreadable is "raise" or "record", not ${JSON.stringify(settings.onUnreadable)}`);
  }
  for (const k of ["apiKey", "baseUrl"] as const) {
    const v = settings[k];
    if (v !== undefined && v !== null && (typeof v !== "string" || !v)) throw new TypeError(`${where}: ${k} is text`);
  }
  const wait = settings.apiMaxWait;
  if (wait !== undefined && wait !== null && !(typeof wait === "number" && wait >= 0)) {
    throw new TypeError(`${where}: apiMaxWait is a number of seconds, at least 0`);
  }
  if (settings.wall !== undefined && settings.wall !== null) wallOf(settings.wall, where);
  const code = settings.codeId;
  if (code !== undefined && code !== null && (typeof code !== "string" || !code.trim() || [...code].length > 200)) {
    throw new TypeError(`${where}: codeId is text naming code a version cannot read (a bundle's hash), at most 200 characters, not ${JSON.stringify(code)}`);
  }
}

/** A domain a program needs (calls.md, *Reach*): a host name, or `*.` and one (every name under it). */
const DOMAIN = /^(\*\.)?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*$/;

/** What code needs to reach, as data (tools.md, *What a tool needs*): domains in lower case, each list sorted and once each. */
export interface Needs {
  readonly domains: readonly string[];
  readonly secrets: readonly string[];
}

/** `{ domains?, secrets? }` checked and written as data; `undefined` when nothing was said. */
export function needsOf(value: unknown, where: string): Needs | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "object" || Array.isArray(value) || Object.keys(value).some((k) => k !== "domains" && k !== "secrets")) {
    throw new TypeError(`${where}: needs is { domains?: [...], secrets?: [...] }: the host names its code reaches and the names of the secrets it uses`);
  }
  const out: Record<string, string[]> = {};
  for (const key of ["domains", "secrets"] as const) {
    const items = (value as Record<string, unknown>)[key] ?? [];
    if (!Array.isArray(items) || !items.every((x) => typeof x === "string" && x.trim())) throw new TypeError(`${where}: needs.${key} is a list of names`);
    let list = items as string[];
    if (key === "domains") {
      list = list.map((x) => x.trim().toLowerCase());
      const bad = list.find((x) => !DOMAIN.test(x));
      if (bad !== undefined) throw new TypeError(`${where}: ${JSON.stringify(bad)} is not a host name (api.example.com) or *. and one (*.example.com)`);
    }
    out[key] = [...new Set(list)].sort(byCodePoint);
  }
  return { domains: out["domains"]!, secrets: out["secrets"]! };
}

/** Several programs' needs as one; `undefined` when none says any. */
export function mergedNeeds(...needs: (Needs | undefined)[]): Needs | undefined {
  const said = needs.filter((n): n is Needs => n !== undefined);
  if (!said.length) return undefined;
  const all = (k: "domains" | "secrets") => [...new Set(said.flatMap((n) => n[k]))].sort(byCodePoint);
  return { domains: all("domains"), secrets: all("secrets") };
}

/** The `wall` setting as data: `{ name, domains?, secrets?, limits? }`. */
export function wallOf(wall: Wall, where = "wall"): { name: string; domains?: string[]; secrets?: string[]; limits?: Record<string, unknown> } {
  if (typeof wall === "string" && wall.trim()) return { name: wall };
  if (typeof wall !== "object" || wall === null || Array.isArray(wall) || typeof wall.name !== "string" || !wall.name.trim()
    || Object.keys(wall).some((k) => !["name", "domains", "secrets", "limits"].includes(k))) {
    throw new TypeError(`${where}: wall is the name of the wall programs run behind ("srt"), or { name, domains?, secrets?, limits? }`);
  }
  const lets = needsOf({ ...(wall.domains !== undefined ? { domains: wall.domains } : {}), ...(wall.secrets !== undefined ? { secrets: wall.secrets } : {}) }, `${where}: wall`)!;
  const out: { name: string; domains?: string[]; secrets?: string[]; limits?: Record<string, unknown> } = { name: wall.name };
  if (wall.domains !== undefined) out.domains = [...lets.domains];
  if (wall.secrets !== undefined) out.secrets = [...lets.secrets];
  if (wall.limits !== undefined) {
    if (typeof wall.limits !== "object" || wall.limits === null || Array.isArray(wall.limits)) throw new TypeError(`${where}: wall.limits is an object`);
    out.limits = copyData(wall.limits) as Record<string, unknown>;
  }
  return out;
}

/**
 * Settings for every program (their own settings still win). Returns the
 * settings now in force. A `logContent` key that is not a field name refuses
 * here (`SettingError`, `log-content-field`), and so does an observer or a
 * journal that could only fail later (`TypeError`).
 */
export function configure(settings: Settings = {}): Settings {
  checkSettings(settings, "configure");
  global = { ...global, ...settings };
  return { ...global };
}

/** Run `fn` with these settings over `configure`'s (a program's own settings and a call's options still win). */
export function withSettings<R>(settings: Settings, fn: () => R): R {
  checkSettings(settings, "withSettings");
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
