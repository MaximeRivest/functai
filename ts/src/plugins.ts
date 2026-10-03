/**
 * Plugins: code that changes what programs do, through a small set of hooks,
 * with every change returned as data and recorded (contract/plugins.md).
 *
 * ```ts
 * const modes = new Plugin("modes", { version: "1.0.0" })
 *   .beforeCall(() => ({ sections: ["Answer carefully. Cite the file you read."], tools: ["read_note"] }));
 * configure({ plugins: [modes] });                 // every call; or per conversation, or per program
 * ```
 *
 * A hook either changes something (it returns a change, or nothing for no
 * opinion) or only hears of something (`turnEnd`). Changes are data: "add
 * this section to the instruction", "show these earlier turns", "offer only
 * these tools", "this tool call may not run". The call's record keeps every
 * change (`changes`), so a rated call is asked again as it was.
 *
 * Built-in features use these hooks and nothing private: `approve` is the
 * `approval` plugin (`toolCall`), `compaction()` keeps a long conversation
 * short (`turnEnd` and `context`), and `delegate(program)` is a tool that
 * runs another program's conversation.
 */

import { Config, type Request } from "@lm15/lm15";
import type { Call } from "./calllog.ts";
import { warnOnce } from "./calllog.ts";
import { Cancelled, FunctAIError, PluginError, type Approval } from "./errors.ts";
import { layersOf, type Layer, type Settings } from "./settings.ts";
import { askPerson, asks, denial, type ApproveFunction } from "./tools.ts";
import { copyData, toJson } from "./values.ts";

type Rec = Record<string, unknown>;

/** The plugin API this FunctAI implements. */
export const API = 1;

/** The hooks, as TypeScript names them (records use the contract's: `before_call`, …). */
export type Hook = "turnStart" | "context" | "beforeCall" | "request" | "toolCall" | "toolResult" | "turnEnd";

/** Each hook's name in records (the contract's), and the change fields it may return. */
const HOOKS: Record<Hook, { readonly name: string; readonly fields: readonly (keyof Change)[] }> = {
  turnStart: { name: "turn_start", fields: ["inputs"] },
  context: { name: "context", fields: ["keep", "without", "sections"] },
  beforeCall: { name: "before_call", fields: ["instruction", "sections", "lm", "settings", "tools"] },
  request: { name: "request", fields: [] },
  toolCall: { name: "tool_call", fields: ["inputs", "block"] },
  toolResult: { name: "tool_result", fields: ["output"] },
  turnEnd: { name: "turn_end", fields: [] },
};

/**
 * What a hook changes, as data. Each hook takes some fields; another refuses
 * (`plugin-change`).
 *
 * - `instruction`: the instruction itself, for this call (`beforeCall`);
 *   `sections`: text added after it, in order (`beforeCall`, `context`);
 * - `lm`: the model for this call; `settings`: lm15 settings for it
 *   (`temperature`, `maxTokens`, `reasoning`, …) (`beforeCall`);
 * - `tools`: the names of the tools offered, from the function's own (`beforeCall`);
 * - `keep`: the ids of the earlier turns shown; `without`: fields left out of
 *   earlier turns, a list for every turn or `{ [turnId]: names }` (`context`);
 * - `inputs`: inputs replaced, by name (`turnStart`: the turn's; `toolCall`: the tool's);
 * - `block`: the tool may not run, and why (`toolCall`);
 * - `output`: the tool's result as the model is shown it (`toolResult`).
 */
export interface Change {
  readonly instruction?: string;
  readonly sections?: readonly string[];
  readonly lm?: string;
  readonly settings?: Readonly<Rec>;
  readonly tools?: readonly string[];
  readonly keep?: readonly string[];
  readonly without?: readonly string[] | Readonly<Record<string, readonly string[]>>;
  readonly inputs?: Readonly<Rec>;
  readonly block?: string;
  readonly output?: string;
}

const CHANGE_FIELDS = new Set(["instruction", "sections", "lm", "settings", "tools", "keep", "without", "inputs", "block", "output"]);
const NAME = /^[a-z][a-z0-9_-]{0,63}$/;

type Handler = (event: any) => unknown;

/**
 * A named, versioned set of hooks. `name` is lower-case letters, digits, `_`
 * and `-` (its changes and entries are named by it); `version` its own,
 * recorded with every change; `api` the plugin API it was written for (1).
 * Register handlers with `on(hook, fn)` or the hook's own method
 * (`.toolCall(fn)`, chainable); each is given an event and returns a change,
 * or nothing.
 *
 * ```ts
 * const guard = new Plugin("no-deletes", { version: "1.0.0" })
 *   .toolCall((tool) => tool.name === "delete_file" ? { block: "deleting files is not allowed here" } : undefined);
 * configure({ plugins: [guard] });
 * ```
 */
export class Plugin {
  readonly name: string;
  readonly version: string;
  readonly api: number;
  readonly description: string;
  readonly handlers: Partial<Record<Hook, Handler[]>> = {};

  constructor(name: string, opts: { version?: string; api?: number; description?: string } = {}) {
    if (typeof name !== "string" || !NAME.test(name)) {
      throw new PluginError("plugin-name", `a plugin's name is lower-case letters, digits, '_' or '-' (at most 64), starting with a letter; not ${JSON.stringify(name)}`);
    }
    const api = opts.api ?? API;
    if (api !== API) throw new PluginError("plugin-api", `plugin ${name} is written for plugin API ${api}; this FunctAI implements ${API}`, { plugin: name });
    this.name = name;
    this.version = String(opts.version ?? "0.0.0");
    this.api = api;
    this.description = opts.description ?? "";
  }

  /** Register `fn` for `hook`. */
  on(hook: Hook, fn: Handler): this {
    if (!Object.hasOwn(HOOKS, hook)) {
      throw new PluginError("plugin-hook", `${this.name}: there is no hook ${JSON.stringify(hook)}; hooks: ${Object.keys(HOOKS).join(", ")}`, { plugin: this.name, hook });
    }
    if (typeof fn !== "function") throw new TypeError(`${this.name}.${hook}: a handler is a function of one event`);
    (this.handlers[hook] ??= []).push(fn);
    return this;
  }

  /** A conversation's turn is about to be recorded: may replace its inputs. */
  turnStart(fn: (event: TurnStartHook) => Change | void | undefined | null): this { return this.on("turnStart", fn); }
  /** Which earlier turns a turn is shown, and sections about them. */
  context(fn: (event: ContextHook) => Change | void | undefined | null): this { return this.on("context", fn); }
  /** An AI function is about to be asked: its instruction, sections, model, settings, tools. */
  beforeCall(fn: (event: BeforeCallHook) => Change | void | undefined | null): this { return this.on("beforeCall", fn); }
  /** The escape hatch: replace the provider request (the call is then not replayable). */
  request(fn: (event: RequestHook) => Request | void | undefined | null): this { return this.on("request", fn); }
  /** A tool is about to run: change its input, block it, or ask a person (`event.ask()`). */
  toolCall(fn: (event: ToolCallHook) => Change | void | undefined | null | Promise<Change | void | undefined | null>): this { return this.on("toolCall", fn); }
  /** A tool ran: change what the model is shown. */
  toolResult(fn: (event: ToolResultHook) => Change | void | undefined | null): this { return this.on("toolResult", fn); }
  /** A conversation's turn ended (hears only; may keep entries in its conversation). */
  turnEnd(fn: (event: TurnEndHook) => void | Promise<void>): this { return this.on("turnEnd", fn); }

  /** Its manifest, as data: name, version, api, the hooks it uses (by the contract's names). */
  describe(): Rec {
    return { name: this.name, version: this.version, api: this.api, description: this.description,
      hooks: (Object.keys(this.handlers) as Hook[]).map((h) => HOOKS[h].name).sort() };
  }

  toString(): string {
    return `Plugin(${this.name} ${this.version}: ${Object.keys(this.handlers).join(", ") || "no hooks"})`;
  }
}

/** Refuse, where it is set, a `plugins` value that is not a list of plugins. */
export function checkPlugins(value: unknown, where: string): void {
  if (value === undefined || value === null) return;
  if (!Array.isArray(value)) throw new TypeError(`${where}: plugins is a list: plugins: [myPlugin]`);
  for (const p of value) if (!(p instanceof Plugin)) throw new TypeError(`${where}: a plugin is a Plugin (new Plugin("name", { version }))`);
}

/**
 * The plugins around a call, in the order their handlers run: the program's
 * own first, then each layer around it from the innermost out, then
 * `configure`'s, so the host's handlers see (and have the last word on) what
 * the program's did. Within a layer, as listed. A plugin set in several
 * layers runs once, in its outermost layer. A host layer's
 * `programPlugins: false` drops the program's own.
 */
export function inOrder(layers: readonly { where: string; plugins?: readonly Plugin[] | null; programPlugins?: boolean | null }[]): Plugin[] {
  const vetoed = layers.some((l) => l.where !== "own" && l.programPlugins === false);
  const seen = new Set<Plugin>();
  const kept: Plugin[][] = [];
  for (const layer of [...layers].reverse()) {                     // outermost first, for "runs in its outermost"
    if (vetoed && layer.where === "own") {
      kept.push([]);
      continue;
    }
    const mine: Plugin[] = [];
    for (const p of layer.plugins ?? []) {
      if (!seen.has(p)) {
        seen.add(p);
        mine.push(p);
      }
    }
    kept.push(mine);
  }
  return kept.reverse().flat();
}

/** The plugins of these settings layers (closest first), in order. */
export function aroundLayers(layers: readonly Layer[]): Plugin[] {
  return inOrder(layers.map((l) => ({ where: l.where, plugins: l.settings.plugins ?? null, programPlugins: l.settings.programPlugins ?? null })));
}

/** The plugins around a call made now with these own settings (and a call's options). */
export function around(own: Settings, call: Settings = {}): Plugin[] {
  return aroundLayers(layersOf(own, call));
}

// ------------------------------------------------------------------ running hooks

/** The changes made in one place, for the record: `{ plugin, version, hook, change }`. */
export class Applied {
  readonly items: Rec[] = [];
  add(plugin: Plugin, hook: Hook, change: Rec): void {
    this.items.push({ plugin: plugin.name, version: plugin.version, hook: HOOKS[hook].name, change });
  }
}

/** Settings a plugin gives, as lm15 writes them in a request (snake_case): what its record keeps. */
function settingsJson(s: Readonly<Rec>): Rec {
  try {
    return Config.toJSON(Config.create(s)) as Rec;
  } catch {
    return toJson(s)[0] as Rec;
  }
}

/** A change as the record keeps it. */
function recorded(change: Change): Rec {
  const out: Rec = {};
  for (const [k, v] of Object.entries(change)) {
    if (v === undefined || v === null) continue;
    out[k] = k === "settings" ? settingsJson(v as Rec) : toJson(v)[0];
  }
  return out;
}

/** Text sections: a list of texts, the blank ones left out. */
export function texts(sections: unknown): string[] {
  if (typeof sections === "string" || !Array.isArray(sections) || !sections.every((s) => typeof s === "string")) {
    throw new TypeError("sections is a list of texts");
  }
  return (sections as string[]).filter((s) => s.trim());
}

/** Base of every hook's event: a plugin's entries in the conversation the call runs in. */
export class HookEvent {
  /** @internal The plugin whose handler runs now. */
  _plugin: Plugin | null = null;
  /** @internal The conversation turn the call runs in, when there is one. */
  _turn: { conversation: EntriesHost; turn: string } | null = null;

  /** This plugin's entries of a kind on the branch of the turn the call runs in (`[]` outside a conversation). */
  entries(kind: string): Rec[] {
    if (!this._turn) return [];
    return this._turn.conversation.entries(this._plugin?.name ?? "", kind, { branch: this._turn.turn });
  }

  /** Keep an entry of this plugin at the turn the call runs in (outside a conversation, a `TypeError`). */
  async remember(kind: string, data: unknown): Promise<void> {
    if (!this._turn) throw new TypeError("remember keeps an entry in a conversation; this call runs in none");
    await this._turn.conversation.remember(this._plugin?.name ?? "", kind, data, { turn: this._turn.turn });
  }
}

/** What an event's `entries`/`remember` reach: a conversation. */
export interface EntriesHost {
  entries(plugin: string, kind: string, opts?: { branch?: string | null }): Rec[];
  remember(plugin: string, kind: string, data: unknown, opts?: { turn?: string | null }): Promise<void>;
}

/**
 * Every handler of `hook`, in order: each is given the event as the changes
 * before it left it, and its change is checked, recorded and applied. A
 * handler that throws stops the call (`plugin-failed`).
 */
export async function runHook(hook: Hook, plugins: readonly Plugin[], event: HookEvent, applied: Applied,
  apply: (change: Change) => void, stop?: () => boolean): Promise<void> {
  const allowed = HOOKS[hook].fields;
  for (const plugin of plugins) {
    for (const fn of plugin.handlers[hook] ?? []) {
      if (stop?.()) return;
      event._plugin = plugin;
      let got: unknown;
      try {
        got = await fn(event);
      } catch (err) {
        if (err instanceof FunctAIError || err instanceof Cancelled || (err as { code?: unknown })?.code === "turn-waiting") throw err;
        throw new PluginError("plugin-failed", `plugin ${plugin.name} failed in ${HOOKS[hook].name}: ${(err as Error)?.name ?? "Error"}: ${(err as Error)?.message ?? String(err)}`,
          { plugin: plugin.name, hook: HOOKS[hook].name, cause: err });
      } finally {
        event._plugin = null;
      }
      if (got === undefined || got === null) continue;
      if (typeof got !== "object" || Array.isArray(got)) {
        throw new PluginError("plugin-change", `plugin ${plugin.name}: ${HOOKS[hook].name} returns a change (an object) or nothing`, { plugin: plugin.name, hook: HOOKS[hook].name });
      }
      const fields = Object.keys(got).filter((k) => (got as Rec)[k] !== undefined && (got as Rec)[k] !== null);
      const unknown = fields.filter((k) => !CHANGE_FIELDS.has(k));
      const wrong = fields.filter((k) => !allowed.includes(k as keyof Change));
      if (unknown.length || wrong.length) {
        throw new PluginError("plugin-change", `plugin ${plugin.name}: ${HOOKS[hook].name} cannot change ${[...new Set([...unknown, ...wrong])].join(", ")} (it may change ${allowed.join(", ") || "nothing"})`,
          { plugin: plugin.name, hook: HOOKS[hook].name });
      }
      if (!fields.length) continue;
      try {
        apply(got as Change);
      } catch (err) {
        if (err instanceof PluginError) throw err;
        throw new PluginError("plugin-change", `plugin ${plugin.name}: ${HOOKS[hook].name}: ${(err as Error).message}`, { plugin: plugin.name, hook: HOOKS[hook].name });
      }
      applied.add(plugin, hook, recorded(got as Change));
    }
  }
}

// ------------------------------------------------------------------ events

/** An earlier turn a turn would be shown. */
export interface ShownTurn {
  readonly id: string;
  readonly inputs: Rec;
  readonly outputs: Rec;
  /** The fields already left out of it. */
  readonly without: readonly string[];
}

/** A conversation's turn, before it is recorded: its inputs (as given), its conversation's id, the turn it continues. */
export class TurnStartHook extends HookEvent {
  inputs: Rec;
  readonly conversation: string;
  readonly parent: string | null;
  readonly program: unknown;
  constructor(inputs: Rec, conversation: string, parent: string | null, program: unknown) {
    super();
    this.inputs = inputs;
    this.conversation = conversation;
    this.parent = parent;
    this.program = program;
  }
}

/** What a conversation `ContextHook` and `TurnEndHook` reach. */
export interface ConversationLike extends EntriesHost {
  readonly id: string;
  /** The ids of the turns from the first to `turn`. */
  branchOf(turn: string | null): string[];
  /** The done turns of the branch through `parent`. */
  turnsOf(parent: string | null): ShownTurn[];
}

/** What a turn is shown: the earlier turns the conversation's rule picked, in order, and the sections so far. */
export class ContextHook extends HookEvent {
  turns: ShownTurn[];
  readonly sections: string[];
  readonly conversation: ConversationLike;
  readonly parent: string | null;
  readonly program: unknown;
  constructor(turns: ShownTurn[], sections: string[], conversation: ConversationLike, parent: string | null, program: unknown) {
    super();
    this.turns = turns;
    this.sections = sections;
    this.conversation = conversation;
    this.parent = parent;
    this.program = program;
  }

  override entries(kind: string): Rec[] {
    return this.conversation.entries(this._plugin?.name ?? "", kind, { branch: this.parent });
  }
}

/** An AI function about to be asked. */
export class BeforeCallHook extends HookEvent {
  /** The instruction as it stands: the program's, or a replacement. */
  instruction: string;
  /** The function's name. */
  readonly function: string;
  readonly program: unknown;
  /** Its inputs, bound. */
  readonly inputs: Rec;
  lm: string | null;
  /** lm15 settings as they stand (`temperature`, `maxTokens`, …). */
  readonly settings: Rec;
  /** The names of the tools offered. */
  tools: string[];
  /** The function's own tools. */
  readonly allTools: readonly string[];
  /** The sections so far. */
  readonly sections: string[];
  /** Its place in the call tree, by names: `support/answer`. */
  readonly path: string;
  readonly conversation: string | null;
  readonly turn: string | null;
  constructor(f: {
    instruction: string; function: string; program: unknown; inputs: Rec; lm: string | null; settings: Rec; tools: string[];
    allTools: readonly string[]; sections: string[]; path: string; conversation: string | null; turn: string | null;
  }) {
    super();
    this.instruction = f.instruction;
    this.function = f.function;
    this.program = f.program;
    this.inputs = f.inputs;
    this.lm = f.lm;
    this.settings = f.settings;
    this.tools = f.tools;
    this.allTools = f.allTools;
    this.sections = f.sections;
    this.path = f.path;
    this.conversation = f.conversation;
    this.turn = f.turn;
  }
}

/** The provider request about to be sent: a handler may return another (the call is then not replayable). */
export class RequestHook extends HookEvent {
  request: Request;
  readonly function: string;
  readonly path: string;
  constructor(request: Request, fn: string, path: string) {
    super();
    this.request = request;
    this.function = fn;
    this.path = path;
  }
}

/** A tool about to run. `ask(reason)` asks a person whether it may run. */
export class ToolCallHook extends HookEvent {
  /** @internal */
  _refused: string | null = null;
  readonly name: string;
  /** Its input, as the changes before this handler left it. */
  input: unknown;
  readonly effects: "reads" | "changes" | null;
  /** `support/answer/refund`. */
  readonly path: string;
  readonly invocation: number;
  readonly id: string;
  /** The AI function that asked for it. */
  readonly function: string;
  /** What a person would be shown. */
  readonly approval: Approval;
  /** The asking call's settings (its `approve` among them). */
  readonly settings: Rec;
  /** @internal */
  readonly _call: Call;
  constructor(approval: Approval, fn: string, settings: Rec, call: Call) {
    super();
    this.name = approval.name;
    this.input = copyData(approval.input);
    this.effects = approval.effects;
    this.path = approval.path;
    this.invocation = approval.invocation;
    this.id = approval.id;
    this.function = fn;
    this.approval = approval;
    this.settings = settings;
    this._call = call;
  }

  /**
   * Ask a person whether the tool may run: in a conversation the turn waits,
   * saved, and goes on when someone answers; on a stream the call waits for
   * `s.approve()`; a plain call refuses (`approval-required`). `decide`
   * answers in place of a person. Resolves true, or false (the person's
   * refusal is then the result the model sees).
   */
  async ask(reason?: string | null, opts: { decide?: ApproveFunction | null } = {}): Promise<boolean> {
    const approval: Approval = { ...this.approval, input: this.input, plugin: this._plugin?.name ?? "approval", question: reason ?? null };
    const [allowed, why] = await askPerson(this._call, approval, opts.decide ?? null);
    if (!allowed) this._refused = denial(why);
    return allowed;
  }
}

/** A tool ran: its result as it stands, which a handler may change. */
export class ToolResultHook extends HookEvent {
  readonly name: string;
  readonly input: unknown;
  output: string;
  readonly path: string;
  readonly invocation: number;
  readonly function: string;
  constructor(approval: Approval, input: unknown, output: string, fn: string) {
    super();
    this.name = approval.name;
    this.input = input;
    this.output = output;
    this.path = approval.path;
    this.invocation = approval.invocation;
    this.function = fn;
  }
}

/**
 * A conversation's turn ended, before its end is recorded (so the next turn,
 * which waits for that end, sees what this hook keeps): `state` is `done`,
 * `failed` or `stopped`. `turns()`: the done turns of its branch, this one
 * last when it is done.
 */
export class TurnEndHook extends HookEvent {
  readonly turn: string;
  readonly state: string;
  readonly inputs: Rec;
  readonly outputs: Rec;
  readonly conversation: ConversationLike;
  readonly parent: string | null;
  constructor(turn: string, state: string, inputs: Rec, outputs: Rec, conversation: ConversationLike, parent: string | null) {
    super();
    this.turn = turn;
    this.state = state;
    this.inputs = inputs;
    this.outputs = outputs;
    this.conversation = conversation;
    this.parent = parent;
    this._turn = { conversation, turn };
  }

  turns(): ShownTurn[] {
    const out = this.conversation.turnsOf(this.parent);
    if (this.state === "done") out.push({ id: this.turn, inputs: copyData(this.inputs), outputs: copyData(this.outputs), without: [] });
    return out;
  }
}

// ------------------------------------------------------------------ what the engine asks

/** A call as its `beforeCall` hooks left it. */
export interface Shaped {
  readonly settings: Settings;
  /** The instruction (null: the program's own). */
  readonly instruction: string | null;
  /** Its context's sections, then the hooks'. */
  readonly sections: readonly string[];
  /** The sections its context gave (what it was shown of its conversation: a rated row carries them). */
  readonly contextSections: readonly string[];
  /** The tools offered (null: all its own). */
  readonly tools: readonly string[] | null;
  readonly applied: Applied;
}

const SETTING_NAMES = new Set(["temperature", "maxTokens", "topP", "stop", "seed"]);
const CONFIG_NAMES = new Set(["maxTokens", "temperature", "topP", "topK", "stop", "seed", "frequencyPenalty", "presencePenalty",
  "responseFormat", "toolChoice", "reasoning", "cache", "serviceTier", "userId", "store", "logprobs", "probabilities", "extensions"]);

/**
 * `beforeCall` for an AI function's call: the sections its context gives
 * first (a conversation's turn, a row asked again), then each handler's.
 */
export async function beforeCall(plugins: readonly Plugin[], spec: {
  instruction: string; name: string; program: unknown; inputs: Rec; settings: Settings; tools: readonly string[];
  given: readonly string[]; call: Call | null;
}): Promise<Shaped> {
  const applied = new Applied();
  const sections = [...spec.given];
  if (!plugins.some((p) => p.handlers.beforeCall?.length)) {
    return { settings: { ...spec.settings }, instruction: null, sections, contextSections: [...spec.given], tools: null, applied };
  }
  const call = spec.call;
  const conv = (call?.conversation ?? null) as Rec | null;
  const s = spec.settings;
  const lm15: Rec = { ...((s.config ?? {}) as Rec) };
  for (const k of SETTING_NAMES) if ((s as Rec)[k] !== undefined && (s as Rec)[k] !== null) lm15[k] = (s as Rec)[k];
  const event = new BeforeCallHook({
    instruction: spec.instruction, function: spec.name, program: spec.program, inputs: copyData(spec.inputs),
    lm: typeof s.lm === "string" ? s.lm : null, settings: lm15, tools: [...spec.tools], allTools: [...spec.tools], sections,
    path: call ? namesOf(call) : "", conversation: (conv?.["id"] as string) ?? null, turn: (conv?.["turn"] as string) ?? null,
  });
  const run = call?.turnRun as unknown as { conversation?: EntriesHost; turn: string } | null;
  if (run?.conversation) event._turn = { conversation: run.conversation, turn: run.turn };
  const out: Rec = { ...spec.settings };
  let offered: string[] | null = null;
  let instruction: string | null = null;
  await runHook("beforeCall", plugins, event, applied, (c) => {
    if (c.instruction !== undefined) {
      if (typeof c.instruction !== "string" || !c.instruction.trim()) throw new TypeError("instruction is the text of an instruction");
      instruction = event.instruction = c.instruction;
    }
    if (c.sections !== undefined) event.sections.push(...texts(c.sections));
    if (c.lm !== undefined) {
      if (typeof c.lm !== "string" || !c.lm.trim()) throw new TypeError("lm is a model's name");
      out["lm"] = event.lm = c.lm;
    }
    if (c.settings !== undefined) {
      const bad = Object.keys(c.settings).filter((k) => !CONFIG_NAMES.has(k));
      if (bad.length) throw new TypeError(`settings are lm15 settings (${[...CONFIG_NAMES].join(", ")}), not ${bad.join(", ")}`);
      for (const [k, v] of Object.entries(c.settings)) {
        if (SETTING_NAMES.has(k)) out[k] = v;
        else out["config"] = { ...((out["config"] ?? {}) as Rec), [k]: v };
        event.settings[k] = v;
      }
    }
    if (c.tools !== undefined) {
      const names = [...c.tools];
      const unknown = names.filter((n) => !spec.tools.includes(n));
      if (unknown.length) throw new TypeError(`${spec.name} has no tool ${JSON.stringify(unknown[0])} (its tools: ${spec.tools.join(", ") || "none"})`);
      offered = event.tools = spec.tools.filter((n) => names.includes(n));
    }
  });
  return { settings: out as Settings, instruction, sections: event.sections, contextSections: [...spec.given], tools: offered, applied };
}

/** A call's place in its tree, by names: `support/answer`. */
export function namesOf(call: Call): string {
  return call.path.split("/").filter(Boolean).map((p) => p.split("#")[0]).join("/");
}

/**
 * The `request` hook (the escape hatch): the request to send, and whether a
 * handler replaced it (the call is then not replayable: its exchange has no
 * `request_hash`).
 */
export async function requestHook(plugins: readonly Plugin[], request: Request, call: Call | null, fn: string): Promise<[Request, boolean]> {
  if (!call || !plugins.some((p) => p.handlers.request?.length)) return [request, false];
  const event = new RequestHook(request, fn, namesOf(call));
  let changed = false;
  for (const plugin of plugins) {
    for (const h of plugin.handlers.request ?? []) {
      event._plugin = plugin;
      let got: unknown;
      try {
        got = await h(event);
      } catch (err) {
        if (err instanceof Cancelled) throw err;
        throw new PluginError("plugin-failed", `plugin ${plugin.name} failed in request: ${(err as Error)?.name}: ${(err as Error)?.message}`, { plugin: plugin.name, hook: "request", cause: err });
      } finally {
        event._plugin = null;
      }
      if (got === undefined || got === null || got === event.request) continue;
      if (typeof got !== "object" || !Array.isArray((got as Request).messages) || typeof (got as Request).model !== "string") {
        throw new PluginError("plugin-change", `plugin ${plugin.name}: request returns an lm15 Request or nothing`, { plugin: plugin.name, hook: "request" });
      }
      event.request = got as Request;
      changed = true;
      call.changes.push({ plugin: plugin.name, version: plugin.version, hook: "request", change: { request: "replaced" } });
    }
  }
  if (changed) call.replayable = false;
  return [event.request, changed];
}

/**
 * `toolCall` for one tool call: `[the input it runs with, null]`, or `[null,
 * what the model is shown instead]`. Handlers run in order, then the
 * `approval` plugin (`approve`) last, on the input they left: a host's rule
 * sees what will run. A handler that throws blocks the tool.
 */
export async function toolCall(plugins: readonly Plugin[], call: Call, approval: Approval, settings: Rec): Promise<[unknown, string | null]> {
  const all = [...plugins, APPROVAL];
  const event = new ToolCallHook(approval, call.node?.name ?? "", settings, call);
  const run = call.turnRun as unknown as { conversation?: EntriesHost; turn: string } | null;
  if (run?.conversation) event._turn = { conversation: run.conversation, turn: run.turn };
  const applied = new Applied();
  let blocked: string | null = null;
  try {
    for (const p of all) {
      if (blocked !== null || event._refused !== null) break;
      await runHook("toolCall", [p], event, applied, (c) => {
        if (c.inputs !== undefined) {
          if (typeof c.inputs !== "object" || c.inputs === null || Array.isArray(c.inputs)) throw new TypeError("inputs is an object of the tool's inputs");
          event.input = { ...(typeof event.input === "object" && event.input !== null ? event.input as Rec : {}), ...c.inputs };
        }
        if (c.block !== undefined) {
          if (typeof c.block !== "string") throw new TypeError("block is the reason, a text");
          blocked = c.block;
        }
      }, () => blocked !== null || event._refused !== null);
    }
  } catch (err) {
    if (!(err instanceof PluginError) || err.code !== "plugin-failed") throw err;
    warnOnce(`tool-call:${err.plugin}`, `${err.message}: the tool does not run`);
    blocked = `a check on this tool call failed (${err.plugin})`;
    applied.items.push({ plugin: err.plugin, version: "", hook: "tool_call", change: { block: blocked } });
  }
  call.changes.push(...applied.items);
  if (blocked !== null) {
    const who = applied.items.length ? applied.items[applied.items.length - 1]!["plugin"] : "?";
    return [null, `This call was blocked (${who}): ${blocked}`];
  }
  if (event._refused !== null) return [null, event._refused];
  return [event.input, null];
}

/** `toolResult`: what the model is shown of a tool's result. */
export async function toolResult(plugins: readonly Plugin[], call: Call, approval: Approval, input: unknown, output: string): Promise<string> {
  if (!plugins.some((p) => p.handlers.toolResult?.length)) return output;
  const event = new ToolResultHook(approval, input, output, call.node?.name ?? "");
  const run = call.turnRun as unknown as { conversation?: EntriesHost; turn: string } | null;
  if (run?.conversation) event._turn = { conversation: run.conversation, turn: run.turn };
  const applied = new Applied();
  await runHook("toolResult", plugins, event, applied, (c) => {
    if (typeof c.output !== "string") throw new TypeError("output is the text the model is shown");
    event.output = c.output;
  });
  call.changes.push(...applied.items);
  return event.output;
}

// ------------------------------------------------------------------ the approval plugin (approve)

/** `approve` as a plugin: a rule names which tool calls a person is asked about; a function answers in place of a person. */
export const APPROVAL = new Plugin("approval", { version: "1.0.0", description: "approve: ask before tools run, as a rule says" })
  .toolCall(async (tool) => {
    const rule = tool.settings["approve"];
    if (!asks(rule, tool.approval)) return undefined;
    await tool.ask(null, { decide: typeof rule === "function" ? rule as ApproveFunction : null });
    return undefined;
  });

export { HOOKS };
