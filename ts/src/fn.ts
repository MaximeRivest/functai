/**
 * AI functions: a typed signature whose body a model writes.
 *
 * ```ts
 * import { ai, t } from "functai";
 *
 * const mood = ai("mood", {
 *   description: "How does the customer feel about what they bought?",
 *   input: { review: t.string() },
 *   output: t.enum("happy", "unhappy", "mixed"),
 * });
 * await mood({ review: "Broke after a day." });   // "unhappy"
 * await mood("Broke after a day.");               // one input: its value alone
 * ```
 */

import * as lmcc from "lmcc";
import type { Request } from "@lm15/lm15";
import type * as calllog from "./calllog.ts";
import type { CallFields } from "./content.ts";
import { lm15Request, run as runEngine, Prediction, type Router } from "./engine.ts";
import { Cancelled, PluginError } from "./errors.ts";
import type { Tool } from "./tools.ts";
import * as plugins from "./plugins.ts";
import { chatOf, Conversation, contextFor, sectionsFor, type ConversationOptions, type Shown, type TurnOptions } from "./conversations.ts";
import { Context as Slot } from "./host.ts";
import { Progress, showProgress } from "./progress.ts";
import { Binder, declareInput, declareOutput, rulesOf, type InputRule, type InputRules } from "./inputs.ts";
import { checkInputs, checkInterface, interfaceSignature, recordedInputs, type Interface } from "./interface.ts";
import { bind } from "./layouts.ts";
import { adjustSettings, callCapabilities, defaultModel, modelString, PROBE } from "./models.ts";
import { routerFor } from "./accounts.ts";
import { droppedFields, recordInputs, runCall } from "./program.ts";
import type { FieldSpec, InputValueOf, IsOptional, ValueOf } from "./shapes.ts";
import { checkSettings, configOf, effective, type Settings } from "./settings.ts";
import { JournalError } from "./log.ts";
import { copyData, entriesOf, getOwn, recordOf, setOwn, toJson, writeData } from "./values.ts";
import { dataShape } from "./interface.ts";
import { bakedJob, isBaked } from "./bake.ts";
import { builtin, env } from "./host.ts";
import * as sig from "./signature.ts";
import { PredictionStream } from "./stream.ts";

type Rec = Record<string, unknown>;

/** A worked example: inputs and the outputs that are right for them. */
export interface Demo {
  inputs: Rec;
  outputs: Rec;
}

/** What improving changes: the instruction and the worked examples. */
export interface State {
  instructions: string | null;
  demos: Array<Demo | Rec>;
}

type Fields = Record<string, FieldSpec>;

/** What `ai(name, options)` takes: the description, the fields, and settings. */
export interface Definition<I extends Fields, O extends Fields | undefined, A extends FieldSpec | undefined> extends Settings {
  /** What it does, in words: the model's instruction. */
  description?: string;
  /** The inputs, by name: `{ message: t.string() }`, zod, or any Standard Schema. */
  input: I;
  /** The answer, named `result`. */
  output?: A;
  /** Several outputs, in order; the last is the answer. */
  outputs?: O;
  /** Code the model may call. */
  tools?: readonly Tool[];
  /** The code module the call log files it under (`program.module`; default: the file's name, without extension). */
  definedIn?: string;
  /** Worked examples: rows of inputs and outputs, or `{inputs, outputs}`. */
  demos?: Array<Demo | Rec>;
  /** An improved instruction that replaces the written one. */
  instructions?: string | null;
}

type Simplify<T> = { [K in keyof T]: T[K] } & {};
type RequiredFields<I> = { [K in keyof I]: IsOptional<I[K]> extends true ? never : K }[keyof I];
/** What a call takes, by name: the inputs a caller must give, and those it may leave out. */
type Inputs<I extends Fields> = Simplify<
  { [K in RequiredFields<I>]: InputValueOf<I[K]> } & { [K in Exclude<keyof I, RequiredFields<I>>]?: InputValueOf<I[K]> }>;
type Outputs<O extends Fields | undefined, A extends FieldSpec | undefined> =
  O extends Fields ? { [K in keyof O]: ValueOf<O[K]> } : A extends FieldSpec ? { result: ValueOf<A> } : { result: string };
type Last<T> = T extends Fields ? T[keyof T] : never;
type Answer<O extends Fields | undefined, A extends FieldSpec | undefined> =
  O extends Fields ? ValueOf<Last<O>> : A extends FieldSpec ? ValueOf<A> : string;

/** The inputs of an AI function, by name. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type InputOf<F> = F extends AIFunction<infer I, any, any> ? I : never;
/** The outputs of an AI function, by name. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type OutputOf<F> = F extends AIFunction<any, infer O, any> ? O : never;
/** A row an AI function can run on: its inputs, typed, and any other columns (the right answers, ids, …). */
export type Row<F> = InputOf<F> & Record<string, unknown>;
/** A column of a row type. */
export type Column<R> = Extract<keyof R, string>;
/** Where the right answers are: a column for the answer, or a column per output. */
export type Expected<F, R> = Column<R> | { [K in keyof OutputOf<F>]?: Column<R> };

/** Any AI function, whatever its types. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyAIFunction = AIFunction<any, any, any>;

/** The keys a caller must give. */
type RequiredKeys<T> = { [K in keyof T]-?: {} extends Pick<T, K> ? never : K }[keyof T];
/** The one required key, when there is exactly one (never otherwise). */
type OnlyRequired<T> = { [K in RequiredKeys<T>]: [Exclude<RequiredKeys<T>, K>] extends [never] ? K : never }[RequiredKeys<T>];

/**
 * What a call takes: the inputs by name, or, for a function with exactly one
 * required input, that input's value alone (`summarize(text)`, its optional
 * inputs left out).
 */
export type Input<I> = I | ([OnlyRequired<I>] extends [never] ? never : I[OnlyRequired<I>]);

/** A call's arguments: the input (which may be left out when every input is optional), then options. */
export type CallArgs<I, Options = CallOptions> = {} extends I
  ? [input?: Input<I>, options?: Options]
  : [input: Input<I>, options?: Options];

/** One call's options: settings for this call only (they beat the function's own), and a signal to cancel it. */
export interface CallOptions extends Settings {
  signal?: AbortSignal;
}

/** Options of `fn.map`: calls in flight at once (default 8), a progress line, and the call options. */
export interface MapOptions extends CallOptions {
  concurrency?: number;
  /** A line on stderr, updated as rows finish: rows done, errors, tokens, time left. Default: on in a terminal. */
  progress?: boolean;
}

/** A conversation with an AI function: called like it (one turn), with the conversation's methods. */
export type ChatOf<I, O, A> = Conversation & {
  (...args: CallArgs<I, TurnOptions>): Promise<A>;
  /** One turn: every output, the lmcc turn, the replies (`p.callId` is the turn's id). */
  predict(...args: CallArgs<I, TurnOptions>): Promise<Prediction<O, A>>;
};

/** An AI function: call it with its inputs, get the answer. */
export interface AIFunction<I extends Rec = Rec, O extends Rec = Rec, A = unknown> {
  (...args: CallArgs<I>): Promise<A>;
  readonly name: string;
  readonly module: string;
  /** The answer's output name (the last output). */
  readonly answerName: string;
  /** The lmcc signature the model gets: fields and instructions. */
  readonly signature: lmcc.Signature;
  /** The instruction the model gets; set it to replace the written one (null: the written one). */
  instructions: string;
  /** The worked examples, as `{inputs, outputs}` (or recorded turns). Set rows or `{inputs, outputs}`. */
  get demos(): Demo[];
  set demos(items: Array<Demo | Rec>);
  /** A fingerprint of what the function sends besides its inputs (contract/calls.md, "Versions"). */
  readonly version: string;
  /** The call log's `program.signature`: the fields' names and shapes. */
  readonly signatureId: string;
  /** What it takes and gives, as data (contract/programs.md): its inputs (optional ones with their default) and outputs. */
  readonly interface: Interface;
  /** The interface's signature: the call log's `program.interface`. Calls with the same one record the same data. */
  readonly interfaceId: string;
  /** The call, with every output, the turn and the replies. */
  predict(...args: CallArgs<I>): Promise<Prediction<O, A>>;
  /** Call it and watch it being made: its answer's text as it is written, and every event of its log. */
  stream(...args: CallArgs<I>): PredictionStream<A, Prediction<O, A>>;
  /** Each item's answer, in order, `concurrency` calls at once. Rejects with the first failure, and stops starting new calls. */
  map(inputs: Iterable<Input<I>>, options?: MapOptions): Promise<A[]>;
  /**
   * Each item's outcome, in order, `concurrency` calls at once, failures
   * included (as `Promise.allSettled` gives them): a long run goes on past a
   * row that fails. With a reply cache on disk (`cacheReplies: "disk"`),
   * running it again sends only what has no kept reply.
   */
  mapSettled(inputs: Iterable<Input<I>>, options?: MapOptions): Promise<PromiseSettledResult<A>[]>;
  /** The exact request a call would send, without sending it (the plugins around it shape it, as they would the call). */
  render(...args: CallArgs<I, Settings>): Promise<Request>;
  /**
   * A conversation with this function: its calls remember each other, kept
   * in a store (`null`: this process's memory; a folder; your own). The same
   * id in the same store opens the same conversation. Called like the
   * function: `await chat(input)` is one turn.
   */
  conversation(id?: string | null, options?: ConversationOptions): ChatOf<I, O, A>;
  /** A copy with other settings (the model, the temperature, …). */
  using(settings: Settings): AIFunction<I, O, A>;
  state(): State;
  loadState(state: Partial<State>): AIFunction<I, O, A>;
  readonly settings: Settings;
  readonly definition: sig.Definition;
  readonly tools: readonly Tool[];
  /** The request this function sends for its sample input (its version's `R`). */
  probeRequest(inputs?: Rec): Rec;
  /** Set when the function was loaded from a saved folder: `sha256:` of its functai.json. */
  readonly saved?: string;
}

/**
 * Where the definition was written: the frame that called `entry` (`ai`).
 *
 * The frame is found by who called, never by file path: bundled, this
 * package's code shares a file with the program that uses it, and a path can
 * be `C:\...`, a `file:` URL or an `https:` URL. Where the host can cut a
 * stack at a function (V8: Node, Deno, Chrome; Bun), it cuts at `entry`;
 * elsewhere the two frames above the caller (this one and `entry`) are
 * dropped. The stack string is read rather than V8's call sites so that
 * source maps (`--enable-source-maps`) still name the original file. A caller
 * with no readable place (eval, native code) gives nothing rather than a guess.
 */
export function definedAt(entry: Function): { file?: string; line?: number; module?: string } {
  const E = Error as ErrorConstructor & { captureStackTrace?: (target: object, cut?: Function) => void };
  let frames: string[];
  if (typeof E.captureStackTrace === "function") {
    const holder: { stack?: string } = {};
    E.captureStackTrace(holder, entry);
    frames = framesOf(holder.stack);
  } else {
    frames = framesOf(new Error().stack).slice(2);
  }
  const where = frames[0] === undefined ? null : frameLocation(frames[0]);
  if (!where) return {};
  const base = where.file.split(/[\\/]/).pop() ?? "";
  return { ...where, module: base.replace(/[?#].*$/, "").replace(/\.[cm]?[jt]sx?$/, "") || undefined };
}

/** The frame lines of a stack string: V8's `    at …` lines, or every line (Firefox, Safari: `name@where`). */
function framesOf(stack: string | undefined): string[] {
  const lines = (stack ?? "").split("\n");
  const v8 = lines.filter((l) => /^\s+at /.test(l));
  return v8.length ? v8 : lines.filter((l) => /:\d+:\d+$/.test(l.trim()));
}

/** The file and line of one frame line, or null when it has none (eval, native). */
export function frameLocation(frame: string): { file: string; line: number } | null {
  let s = frame.trim();
  if (s.startsWith("at ")) {                     // V8: `at where`, `at name (where)`, `at async name (where)`
    s = s.slice(3);
    if (s.endsWith(")")) {                        // the parenthesis that opens the last group: a path may hold parentheses itself
      let depth = 0;
      for (let i = s.length - 1; i >= 0; i--) {
        if (s[i] === ")") depth++;
        else if (s[i] === "(" && --depth === 0) { s = s.slice(i + 1, -1); break; }
      }
    }
  } else {
    const at = s.indexOf("@");                     // Firefox, Safari: `name@where`; a name holds no `@`, a path may
    if (at >= 0) s = s.slice(at + 1);
  }
  if (s.startsWith("eval at ") || s.includes("<anonymous>")) return null;
  const m = /^(.+):(\d+):\d+$/.exec(s);
  if (!m) return null;
  return { file: filePath(m[1]!), line: Number(m[2]) };
}

/** A `file:` URL as the host's path (`C:\app\x.js` on Windows, spaces decoded); anything else as it is. */
function filePath(where: string): string {
  if (!where.startsWith("file:")) return where;
  const url = builtin("node:url");
  if (url) {
    try {
      return url.fileURLToPath(where);
    } catch {
      // a URL the host cannot turn into a path: read it below
    }
  }
  try {
    const u = new URL(where);
    const path = decodeURIComponent(u.pathname);
    return /^\/[A-Za-z]:\//.test(path) ? path.slice(1).replaceAll("/", "\\") : path;
  } catch {
    return where;
  }
}

const isRecordedTurn = (item: unknown): boolean =>
  typeof item === "object" && item !== null && Object.hasOwn(item, "signature") && Object.hasOwn(item, "steps");

function demoOf(item: Demo | Rec, inputNames: readonly string[], answer: string): Demo {
  if (isRecordedTurn(item)) return item as unknown as Demo;   // a recorded turn
  // copies member by member: `__proto__` stays a member, and the order stays the value's
  const copy = (r: unknown): Rec => recordOf(entriesOf(r as Rec));
  if (typeof item === "object" && item !== null && Object.hasOwn(item, "inputs") && Object.hasOwn(item, "outputs")
    && typeof item["inputs"] === "object" && !inputNames.includes("inputs")) {
    return { inputs: copy(item["inputs"]), outputs: copy(item["outputs"]) };
  }
  if (Array.isArray(item) && item.length === 2) {
    const [i, o] = item as unknown as [unknown, unknown];
    return {
      inputs: typeof i === "object" && i !== null && !Array.isArray(i) ? copy(i) : recordOf([[inputNames[0]!, i]]),
      outputs: typeof o === "object" && o !== null && !Array.isArray(o) ? copy(o) : recordOf([[answer, o]]),
    };
  }
  const inputs: Rec = {};
  const outputs: Rec = {};
  for (const [k, v] of entriesOf(item as Rec)) setOwn(inputNames.includes(k) ? inputs : outputs, k, v);
  return { inputs, outputs };
}

interface Core {
  definition: sig.Definition;
  /** What it takes and gives, as data; its inputs' rules follow it unless `rules` says more (a schema). */
  interface: Interface;
  /** By input name; a loaded function has none, and its rules come from its interface. */
  rules?: InputRules;
  own: Settings;
  tools: readonly Tool[];
  module: string;
  file?: string;
  line?: number;
  saved?: string;
  state: { instructions: string | null; demos: Demo[] };
  /** Inputs whose default is given as a function: its code (calls.md, "Versions", "Defaults"). */
  defaultCode?: Readonly<Record<string, string>>;
  /** Its module was taken from its file's name (no `definedIn`): its calls are known by its file too (calls.md, rated). */
  topLevel?: boolean;
}

/**
 * `D` of a version (calls.md, "Versions", "Defaults"): each input that has a
 * default, in the interface's order, `{ code }` when it is given as a
 * function, else `{ value }`; undefined when none has one.
 */
export function defaultsDocument(iface: Interface, code: Readonly<Record<string, string>> = {}): Rec | undefined {
  const out: Rec = {};
  for (const f of iface.inputs) {
    const c = getOwn(code, f.name);
    if (c !== undefined) setOwn(out, f.name, { code: c });
    else if (Object.hasOwn(f.shape, "default")) setOwn(out, f.name, { value: f.shape["default"] });
  }
  return Object.keys(out).length ? out : undefined;
}

const SETTING_KEYS = new Set(["lm", "router", "temperature", "maxTokens", "topP", "stop", "seed", "config", "adapter", "template",
  "module", "includeFnName", "capabilities", "retries", "apiRetries", "maxSteps", "toolErrors", "logCalls", "logContent", "caller",
  "cacheReplies", "observers", "journal", "programObservers", "replicate", "approve", "plugins", "programPlugins", "escalateTo",
  "escalateBelow"]);

/** The fields of a call of a function with this signature (calls.md, "Content"): its inputs, and every output, those FunctAI adds included. */
export function fieldsOf(iface: Interface, signature: lmcc.Signature): CallFields {
  const outputs = signature.fields.filter((f) => f.direction === "output");
  return {
    inputs: iface.inputs.map((f) => f.name),
    outputs: outputs.map((f) => f.name),
    added: outputs.filter((f) => (f.purpose ?? "plain") !== "plain").map((f) => f.name),
  };
}

/**
 * An AI function: its name, what it does, its input and output fields; a
 * language model writes the body. `ai("mood", { description, input: { review:
 * t.string() }, output: t.enum("happy", "unhappy") })`.
 */
export function ai<I extends Fields, O extends Fields | undefined = undefined, A extends FieldSpec | undefined = undefined>(
  name: string, def: Definition<I, O, A>,
): AIFunction<Inputs<I>, Outputs<O, A>, Answer<O, A>> {
  if (typeof name !== "string" || !name) throw new TypeError('ai(name, { input, output }): the name comes first: ai("mood", { ... })');
  if (!def || typeof def !== "object" || !def.input || typeof def.input !== "object") throw new TypeError(`ai("${name}", { input: { ... } }): the inputs are required`);
  if (def.output !== undefined && def.outputs !== undefined) throw new TypeError(`${name}: give output (one answer) or outputs (several), not both`);
  const rules = new Map<string, InputRule>();
  const defaultCode: Record<string, string> = {};
  const inputs = Object.entries(def.input).map(([field, spec]) => {
    const { field: declared, rule, code } = declareInput(field, spec, `${name}.input.${field}`, "ai");
    rules.set(field, rule);
    if (code !== undefined) setOwn(defaultCode, field, code);
    return { name: field, shape: declared.shape, desc: declared.desc ?? null, ...(rule.optional ? { optional: true } : {}) };
  });
  const outputSpecs: [string, FieldSpec][] = def.outputs
    ? Object.entries(def.outputs)
    : [["result", (def.output ?? { type: "string" }) as FieldSpec]];
  if (!outputSpecs.length) throw new TypeError(`${name}: outputs is empty`);
  const outputs = outputSpecs.map(([field, spec]) => {
    const declared = declareOutput(field, spec, `${name}.outputs.${field}`, "ai");
    return { name: field, shape: declared.shape, desc: declared.desc ?? null };
  });
  const where = definedAt(ai);
  const own: Settings = {};
  for (const [k, v] of Object.entries(def)) if (SETTING_KEYS.has(k)) (own as Rec)[k] = v;
  const moduleName = def.definedIn ?? where.module ?? "main";
  const definition: sig.Definition = {
    name, description: def.description ?? "", inputs, outputs,
    cot: own.module === "cot", tools: Boolean(def.tools?.length), includeName: own.includeFnName !== false,
  };
  // lmcc checks the signature first (signature-malformed), then the interface is checked by the contract's rules
  const signature = sig.signature(definition, def.instructions ?? null);
  const iface = checkInterface(interfaceOf(definition), { ai: true, where: `ai("${name}")` });
  checkSettings(own, `ai("${name}")`, fieldsOf(iface, signature));
  const core: Core = {
    definition: { ...definition, cot: false, includeName: true }, interface: iface, rules, own, tools: [...(def.tools ?? [])],
    module: moduleName, file: where.file, line: where.line, state: { instructions: def.instructions ?? null, demos: [] },
    defaultCode, topLevel: def.definedIn === undefined,
  };
  const fn = make(core) as unknown as AIFunction<Inputs<I>, Outputs<O, A>, Answer<O, A>>;
  if (def.demos) fn.demos = def.demos as Demo[];
  return fn;
}

/** An AI function's interface: its definition's description, inputs (optional ones with their default) and outputs. */
export function interfaceOf(d: sig.Definition): Interface {
  const field = (f: sig.FieldDef) => ({ name: f.name, shape: f.shape, ...(f.desc ? { desc: f.desc } : {}), ...(f.optional ? { optional: true as const } : {}) });
  return { description: d.description, inputs: d.inputs.map(field), outputs: d.outputs.map(field) };
}

const ids = new WeakMap<object, number>();
let lastId = 0;
function objectId(o: object): number {
  let id = ids.get(o);
  if (id === undefined) {
    id = ++lastId;
    ids.set(o, id);
  }
  return id;
}

/** An AI function over a core (ai(), using(), a saved folder). */
export function make(core: Core): AIFunction {
  const names = core.definition.inputs.map((f) => f.name);
  const answer = core.definition.outputs[core.definition.outputs.length - 1]!.name;
  const cache = new Map<string, unknown>();

  const settingsNow = (call: Settings = {}) => effective({ ...core.own, ...call });
  const definitionNow = (s: Settings): sig.Definition => ({
    ...core.definition, cot: s.module === "cot", includeName: s.includeFnName !== false,
  });
  const signatureNow = (s: Settings = settingsNow()) => {
    const d = definitionNow(s);
    const key = JSON.stringify(["sig", d.cot, d.includeName, core.state.instructions]);
    let got = cache.get(key) as lmcc.Signature | undefined;
    if (!got) {
      got = sig.signature(d, core.state.instructions);
      cache.set(key, got);
    }
    return got;
  };
  const layout = (s: Settings) => ({ adapter: s.adapter, template: s.template ?? null });
  /** A cache key for the layout: a named one by name, an adapter object by identity. */
  const layoutKey = (s: Settings): string => {
    const a = s.adapter;
    const adapter = a === undefined || a === null || typeof a === "string" ? (a ?? null) : `object:${objectId(a as object)}`;
    return JSON.stringify([adapter, s.template ?? null]);
  };

  const rules: InputRules = core.rules ?? rulesOf(core.interface.inputs);
  const binder = new Binder(core.definition.name, names, rules);
  /** A call's argument as inputs by name; inputs left out get their default. */
  const bindInputs = (arg: unknown): Rec => binder.bind(arg)[0];
  const parseInputsNow = (arg: unknown): Rec => {
    const [raw, filled] = binder.bind(arg, { check: false });
    const bound = checkInputs(core.interface, raw, core.definition.name, droppedFields(fieldsOf(core.interface, sig.signature(definitionNow({}), core.state.instructions)), core.own));
    binder.parse(bound, filled, true);
    return bound;
  };

  /** Worked examples as example turns for this plan (contract/functions.md, "Worked examples"). */
  const pastTurns = (plan: lmcc.Plan): lmcc.Turn[] => {
    const plain = new Set(plan.signature.fields.filter((f) => f.direction === "input" && f.purpose === "plain").map((f) => f.name));
    const kept = new Set(plan.signature.fields.filter((f) => f.direction === "output" && (f.purpose === "plain" || f.purpose === "reasoning")).map((f) => f.name));
    const out: lmcc.Turn[] = [];
    for (const d of core.state.demos) {
      if (isRecordedTurn(d)) {
        const recorded = d as unknown as { signature: string };
        if (recorded.signature === plan.fingerprint) {
          let t: lmcc.Turn | null = null;
          try {
            t = plan.loadTurn(d);
          } catch {
            // read as its inputs and outputs below
          }
          if (t) {
            out.push(t);
            continue;
          }
        }
      }
      const ins = sig.prepareInputs(plan.signature, recordOf(entriesOf(d.inputs ?? {}).filter(([k]) => plain.has(k))));
      const outs = recordOf(entriesOf(d.outputs ?? {}).filter(([k]) => kept.has(k)));
      if (!Object.keys(outs).length) continue;
      try {
        out.push(plan.example(ins, outs));
      } catch (err) {
        if (!lmcc.isRefusal(err)) throw err;              // a demo lmcc refuses to make an example of is skipped
      }
    }
    return out;
  };

  const probePlan = (s: Settings) => bind(layout(s), signatureNow(s), PROBE, "probe");

  const probeRequest = (inputs?: Rec): Rec => {
    const s = settingsNow();
    const plan = probePlan(s);
    const given = inputs ? binder.bind(inputs, { check: false })[0] : sig.sampleInputs(plan.signature);
    const values = sig.prepareInputs(plan.signature, given);
    if (core.tools.length) setOwn(values, "tools", core.tools.map((t) => ({ name: t.name, description: t.description ?? null, parameters: t.parameters })));
    return plan.render(plan.turn(values), { turns: pastTurns(plan) }).request("probe");
  };

  const version = (): string => {
    const s = settingsNow();
    const key = writeData(["version", layoutKey(s), s.module ?? null, s.includeFnName ?? null, core.state]);   // members in order: a demo in another order is sent otherwise
    let v = cache.get(key) as string | undefined;
    if (!v) {
      let r: string;
      try {
        r = lmcc.sha256(probeRequest());
      } catch (err) {
        if (!lmcc.isRefusal(err)) throw err;
        r = `refused:${(err as lmcc.Refusal).code}`;
      }
      const defaults = defaultsDocument(core.interface, core.defaultCode);
      v = lmcc.sha256((defaults ? { request: r, defaults } : { request: r }) as lmcc.Json);
      cache.set(key, v);
    }
    return v;
  };

  const route = (s: Settings) => {
    const lm = s.lm ?? defaultModel(env());
    if (!lm) throw new Error("no model configured, and no API key found to pick one: set OPENAI_API_KEY (or ANTHROPIC_API_KEY, GEMINI_API_KEY, …), or name a model: configure({ lm: \"gpt-4.1-mini\" })");
    const model = modelString(lm);
    const router = (s.router ?? routerFor(model)) as Router & { resolve(m: string): { provider: string; model: string } };
    const resolution = router.resolve(model);
    return { model, router, provider: resolution.provider, wire: resolution.model };
  };

  const planFor = <S extends Settings>(given: S, instructions?: string | null) => {
    const r = route(given);
    const s = adjustSettings(given, r.provider, r.wire);
    const caps = callCapabilities(r.provider, r.wire, { temperature: s.temperature, capabilities: s.capabilities });
    const text = instructions ?? core.state.instructions;
    const key = JSON.stringify(["plan", layoutKey(s), caps, r.provider, s.module ?? null, s.includeFnName ?? null, text, instructions !== undefined && instructions !== null]);
    let plan = cache.get(key) as lmcc.Plan | undefined;
    if (!plan) {
      const signature = instructions !== undefined && instructions !== null ? sig.signature(definitionNow(s), instructions) : signatureNow(s);
      plan = bind(layout(s), signature, caps, r.provider);
      if (cache.size > 64) cache.clear();
      cache.set(key, plan);
    }
    return { plan, ...r, settings: s };
  };

  const interfaceId = interfaceSignature(core.interface);
  const program = (s: Settings = settingsNow()): calllog.Program => ({
    name: core.definition.name, kind: "ai", module: core.module, version: version(),
    signature: sig.signatureId(signatureNow(s)), interface: interfaceId, answer,
    ...(core.saved ? { saved: core.saved } : {}),
    ...(core.file ? { file: core.file } : {}), ...(core.line ? { line: core.line } : {}),
  });

  /** Every output the model gave, the fields FunctAI added included (`reasoning`, `calls`: the last step's), in the signature's order: the record's. */
  const recordOutputs = (pred: Prediction, signature: lmcc.Signature): Rec => {
    const all = (pred.turn.outputs ?? {}) as Rec;
    const mine = pred.outputs as Rec;
    // outputs.calls: every tool call the model asked for in the call, across steps (contract/functions.md)
    const toolCalls = (pred as unknown as { toolCalls?: Rec[] }).toolCalls;
    const out: Rec = {};
    for (const f of signature.fields) {
      if (f.direction !== "output") continue;
      if (f.purpose === "tools.calls" && toolCalls !== undefined) setOwn(out, f.name, toolCalls);
      else if (Object.hasOwn(mine, f.name)) setOwn(out, f.name, mine[f.name]);
      else if (Object.hasOwn(all, f.name)) setOwn(out, f.name, all[f.name]);
    }
    return out;
  };

  /**
   * Earlier turns, as this plan shows them, and the `saw` entries that say so
   * (calls.md, *Saw*): a turn made for the plan whole, with its steps; one made
   * for another as its values alone (`without` the fields the plan no longer
   * has); one with no output left, not at all.
   */
  const shownAs = (plan: lmcc.Plan, shown: Shown): [Rec[], lmcc.Turn[]] => {
    const entries: Rec[] = [];
    const turns: lmcc.Turn[] = [];
    shown.turns.forEach((t, i) => {
      const id = shown.ids[i] ?? "";
      let fitted: lmcc.Turn | null = null;
      let whole = false;
      if (Object.hasOwn(t, "signature") && Object.hasOwn(t, "steps")) {
        try {
          fitted = plan.loadTurn({ ...t, signature: plan.fingerprint });
          whole = true;
        } catch {
          fitted = null;
        }
      }
      if (!fitted) fitted = exampleOf(plan, (t["inputs"] ?? {}) as Rec, (t["outputs"] ?? {}) as Rec);
      if (!fitted) return;
      if (whole) entries.push(((t["steps"] as unknown[]) ?? []).length ? { call: id, steps: true } : { call: id });
      else {
        const had = new Set([...Object.keys((t["inputs"] ?? {}) as Rec), ...Object.keys((t["outputs"] ?? {}) as Rec)]);
        const kept = new Set([...Object.keys(fitted.inputs ?? {}), ...Object.keys(fitted.outputs ?? {})]);
        const leftOut = [...had].filter((n) => !kept.has(n)).sort();
        entries.push(leftOut.length ? { call: id, without: leftOut } : { call: id });
      }
      turns.push(fitted);
    });
    return [entries, turns];
  };

  const exampleOf = (plan: lmcc.Plan, inputs: Rec, outputs: Rec): lmcc.Turn | null => {
    const plain = new Set(plan.signature.fields.filter((f) => f.direction === "input" && f.purpose === "plain").map((f) => f.name));
    const kept = new Set(plan.signature.fields.filter((f) => f.direction === "output" && (f.purpose === "plain" || f.purpose === "reasoning")).map((f) => f.name));
    const ins = sig.prepareInputs(plan.signature, recordOf(entriesOf(inputs).filter(([k]) => plain.has(k))));
    const outs = recordOf(entriesOf(outputs).filter(([k]) => kept.has(k)));
    if (!Object.keys(outs).length) return null;
    try {
      return plan.example(ins, outs);
    } catch (err) {
      if (!lmcc.isRefusal(err)) throw err;
      return null;
    }
  };

  /** What a call is shown as earlier turns, captured when it is prepared (its saw). */
  interface Captured { readonly plan: string | null; readonly shown: Shown; readonly entries: Rec[]; readonly turns: lmcc.Turn[] }
  const captured = new WeakMap<object, Captured>();

  /** Work out what the call is shown (its saw) before it starts: a conversation's turns, a helper's memory, a row asked again. */
  const prepare = async (call: calllog.Call, s: Settings): Promise<void> => {
    const found = await contextFor(fn, call);
    if (!found) return;
    let plan: lmcc.Plan | null = null;
    try {
      plan = planFor(s).plan;
    } catch {
      plan = null;                                     // no model to plan for: the call fails there, and says why
    }
    let entries: Rec[];
    let turns: lmcc.Turn[] = [];
    if (plan) [entries, turns] = shownAs(plan, found);
    else entries = found.ids.map((id) => ({ call: id }));
    if (found.finish) entries = found.finish(entries);
    call.saw = entries;
    captured.set(call, { plan: plan?.fingerprint ?? null, shown: found, entries, turns });
  };

  /** The turns placed before the call's own: the worked examples, then what the call is shown as earlier turns. */
  const pastFor = async (plan: lmcc.Plan, call: calllog.Call | null, baked: boolean): Promise<lmcc.Turn[]> => {
    const past = baked ? [] : pastTurns(plan);
    if (call) {
      const ctx = captured.get(call);
      if (!ctx) return past;
      if (ctx.plan === plan.fingerprint) return [...past, ...ctx.turns];
      // another plan than the one its saw was worked out for (an escalation to another model): shown again as this
      // plan shows it; when that differs, its record says the context changed
      const [entries, turns] = shownAs(plan, ctx.shown);
      if (writeData(entries) !== writeData(ctx.entries) && !call.saw.some((e) => (e as Rec)["context"] === "changed")) call.saw.push({ context: "changed" });
      return [...past, ...turns];
    }
    const found = await contextFor(fn, null);                  // render(): what the next turn's call would be shown
    return found ? [...past, ...shownAs(plan, found)[1]] : past;
  };

  /** The instruction a layout writes: a template without `{instruction}` would drop what plugins add to it. */
  const checkPlaced = (s: Settings): void => {
    if (isBaked(s.lm)) {
      throw new PluginError("plugin-change", `${core.definition.name}: plugins changed its instruction (sections, a summary of earlier turns, or a replacement), and it runs on a baked model, which reads only the message it was trained on: the change would not reach it. Run it without those plugins, or on a model that reads instructions.`);
    }
    if (s.template && !JSON.stringify(s.template).includes("{instruction}")) {
      throw new PluginError("plugin-change", `${core.definition.name}: plugins changed its instruction (sections, a summary of earlier turns, or a replacement), and its template never writes {instruction}, so the change would not be sent. Put {instruction} in its template, or use plugins that do not change the instruction with it.`);
    }
  };

  /** The call as its `beforeCall` hooks leave it: settings, the instruction (null: its own), the tools offered. */
  const shape = async (inputs: Rec, s: Settings, call: calllog.Call | null, extra: Settings) => {
    const around = plugins.around(core.own, extra);
    const base = signatureNow(s).instructions;
    const shaped = await plugins.beforeCall(around, {
      instruction: base, name: core.definition.name, program: fn, inputs, settings: s, tools: core.tools.map((t) => t.name),
      given: sectionsFor(fn, call), call,
    });
    if (call) {
      call.changes.push(...shaped.applied.items);
      call.sections = [...shaped.contextSections];             // what it was shown of its conversation (replayed)
    }
    const changed = shaped.instruction !== null || shaped.sections.length > 0;
    const instruction = changed ? [shaped.instruction ?? base, ...shaped.sections].join("\n\n") : null;
    if (changed) checkPlaced(shaped.settings);
    const offered = shaped.tools === null ? core.tools : core.tools.filter((t) => shaped.tools!.includes(t.name));
    return { settings: effective({ ...shaped.settings }) as typeof s & ReturnType<typeof effective>, instruction, offered, plugins: around };
  };

  /** One model call (and its tool loop): on the model the settings name, or a baked student. */
  const ask = async (call: calllog.Call, inputs: Rec, shaped: Awaited<ReturnType<typeof shape>>, lm?: unknown): Promise<Prediction> => {
    const s = lm === undefined ? shaped.settings : { ...shaped.settings, lm: lm as string };
    if (isBaked(s.lm)) {
      const job = bakedJob(fn, s.lm, s, inputs, call, answer);
      call.provider = job.provider;
      return runEngine({ ...job, past: await pastFor(job.plan, call, true), plugins: shaped.plugins });
    }
    const { plan, model, router, provider, settings } = planFor(s, shaped.instruction);
    call.provider = provider;
    return runEngine({
      function: core.definition.name, plan, past: await pastFor(plan, call, false), inputs, settings, router, model,
      tools: shaped.offered, call, answer, plugins: shaped.plugins,
    });
  };

  const predict = async (arg: unknown, options: CallOptions = {}, stream?: PredictionStream): Promise<Prediction> => {
    const { signal, ...extra } = options;
    if (signal?.aborted || stream?.signal.aborted) throw new Cancelled();
    const s = settingsNow(extra);
    const signature = signatureNow(s);
    const fields = fieldsOf(core.interface, signature);
    // Inputs are bound to the interface (programs.md, "Binding a call's inputs"), then parsed by their schemas; a refusal
    // is the call's outcome, recorded, and nothing is sent. The record holds the bound values, as JSON before a schema's
    // parsing runs (it may change them in place).
    const [raw, filled] = binder.bind(arg, { check: false });
    let bound: Rec;
    let refused: unknown;
    try {
      bound = checkInputs(core.interface, raw, core.definition.name, droppedFields(fields, core.own, extra as Settings));
    } catch (err) {
      refused = err;
      bound = recordedInputs(core.interface, raw);
    }
    const given = recordInputs(names, bound);
    if (refused === undefined) {
      try {
        await binder.parse(bound, filled);
      } catch (err) {
        refused = err;
      }
    }
    return runCall<Prediction>({
      program: () => program(s), fields, own: core.own, options: extra as Settings, self: fn,
      settings: s, stream, signal, inputs: given, ...(refused !== undefined ? { refused } : {}),
      prepare: (call) => prepare(call, s),
      body: async (call) => {
        const shaped = await shape(bound, s, call, extra as Settings);
        let pred = await ask(call, bound, shaped);
        // escalation: a first model less sure than escalateBelow has another model (or AI function) answer instead
        const target = ESCALATING.get() ? core.own.escalateTo : shaped.settings.escalateTo;
        if (target !== undefined && target !== null) {
          const conf = pred.confidence;
          if (conf === null) {
            throw new TypeError(`${core.definition.name}: escalateTo needs a first model that measures its confidence (a baked model, TypeSafe's Jev, or config: { probabilities: "required" }); ${pred.response?.model ?? "the model"} gave no probabilities`);
          }
          const threshold = shaped.settings.escalateBelow ?? 0.9;
          if (conf < threshold) {
            const who = typeof target === "function" ? (target as { name: string }).name : typeof target === "string" ? target : "another model";
            call.event("retry", { reason: `the first model was ${Math.round(conf * 100)}% sure (less than ${Math.round(threshold * 100)}%); ${who} answers instead`, wait: null });
            const first = pred;
            if (typeof target === "function" && typeof (target as { predict?: unknown }).predict === "function") {
              // the target follows its own escalateTo (a longer chain), never one around this call
              const other = target as unknown as AIFunction;
              const ins = recordOf(other.definition.inputs.filter((f) => Object.hasOwn(bound, f.name)).map((f) => [f.name, bound[f.name]] as const));
              pred = await ESCALATING.run(true, () => other.predict(ins as never)) as Prediction;
              pred = new Prediction(pred.outputs, answer, call.id, pred.turn, pred.response, pred.responses, pred.repairs);
            } else {
              pred = await ask(call, bound, { ...shaped, settings: { ...shaped.settings, escalateTo: null } }, target);
            }
            pred.first = first;
            call.escalated = true;
          }
        }
        call.confidence = pred.confidence;
        if (Object.keys(pred.probabilities).length) call.probabilities = pred.probabilities;
        call.steps = stepsOf(call, pred);
        return { value: pred, outputs: recordOutputs(pred, signature), shown: pred.answer, returned: pred.answer };
      },
    });
  };
  /** What `fn(x)` gives: the answer; a journal that did not keep the end holds the answer too. */
  const answerOf = (p: Promise<Prediction>): Promise<unknown> => p.then((pred) => pred.answer, (err: unknown) => {
    throw err instanceof JournalError ? err.withDone((v) => (v as Prediction).answer) : err;
  });

  /** Every item, `concurrency` at once: each one's outcome (stops starting new ones once `stop()` says so). */
  const each = async (items: Iterable<unknown>, options: MapOptions, stopOnFailure: boolean): Promise<PromiseSettledResult<unknown>[]> => {
    const { concurrency = 8, progress, ...call } = options;
    const list = [...items];
    const out: PromiseSettledResult<unknown>[] = new Array(list.length);
    const stop = new AbortController();
    const signal = call.signal ? AbortSignal.any([call.signal, stop.signal]) : stop.signal;
    const meter = showProgress(progress) && list.length ? new Progress(list.length, core.definition.name) : null;
    let next = 0;
    let failure: unknown = null;
    const worker = async () => {
      while (next < list.length && !signal.aborted) {
        const i = next++;
        try {
          const pred = await predict(list[i], { ...call, signal });
          out[i] = { status: "fulfilled", value: pred.answer };
          meter?.step(false, pred.usage["totalTokens"] ?? 0);
        } catch (err) {
          out[i] = { status: "rejected", reason: err instanceof JournalError ? err.withDone((v) => (v as Prediction).answer) : err };
          meter?.step(true, 0);
          if (stopOnFailure) {
            failure ??= { err: out[i] };
            stop.abort();                                // the first failure: start no more
          }
        }
      }
    };
    await Promise.all(Array.from({ length: Math.max(1, Math.min(concurrency, list.length || 1)) }, worker));
    if (failure) throw ((failure as { err: PromiseRejectedResult }).err).reason;
    if (call.signal?.aborted) throw new Cancelled();
    return out;
  };

  const map = async (items: Iterable<unknown>, options: MapOptions = {}): Promise<unknown[]> =>
    (await each(items, options, true)).map((r) => (r as PromiseFulfilledResult<unknown>).value);

  const fn = ((input: unknown, options?: CallOptions) => answerOf(predict(input, options))) as unknown as AIFunction;
  const streamOf = (input: unknown, options: CallOptions, passive: boolean) => {
    bindInputs(input);                                 // wrong arguments fail here, not in the stream
    return new PredictionStream<unknown, Prediction>((st) => predict(input, options, st as PredictionStream), (p) => p.answer, options.signal, passive);
  };
  const props: PropertyDescriptorMap = {
    name: { value: core.definition.name },
    module: { get: () => core.module },
    answerName: { value: answer },
    signature: { get: () => signatureNow() },
    instructions: {
      get: () => signatureNow().instructions,
      set: (text: string | null) => { core.state = { ...core.state, instructions: text }; },
    },
    demos: {
      get: () => core.state.demos.map((d) => copyData(d)),
      set: (items: Array<Demo | Rec>) => { core.state = { ...core.state, demos: (items ?? []).map((d) => demoOf(d, names, answer)) }; },
    },
    version: { get: version },
    signatureId: { get: () => sig.signatureId(signatureNow()) },
    interface: { get: () => copyData(core.interface) },
    defaultCode: { get: () => ({ ...(core.defaultCode ?? {}) }) },
    file: { get: () => (core.topLevel ? core.file : undefined) },
    interfaceId: { value: interfaceId },
    settings: { get: () => ({ ...core.own }) },
    definition: { get: () => core.definition },
    tools: { get: () => core.tools },
    saved: { get: () => core.saved },
  };
  Object.defineProperties(fn, props);
  Object.assign(fn, {
    predict: (input: unknown, options?: CallOptions) => predict(input, options),
    stream: (input: unknown, options: CallOptions = {}) => streamOf(input, options, false),
    conversation: (id?: string | null, options: ConversationOptions = {}) => chatOf(new Conversation(fn as never, id ?? null, options)) as never,
    _stream: (input: unknown, options: CallOptions, passive: boolean) => streamOf(input, options, passive),
    _kind: "ai",
    _conversationFields: () => {
      const sigNow = signatureNow();
      return sigNow.fields.filter((f) => !(f.direction === "input" && f.purpose !== "plain"))
        .map((f) => ({ name: f.name, direction: f.direction, purpose: f.purpose ?? "plain", shape: dataShape(f.shape as Rec) }));
    },
    _recordedInputs: (input: unknown) => {
      const bound = parseInputsNow(input);
      return recordOf(entriesOf(bound).map(([k, v]) => [k, toJson(v)[0]] as const));
    },
    _programInfo: () => program(),
    _exampleTurn: (inputs: Rec, outputs: Rec) => {
      const plan = planFor(settingsNow()).plan;
      const turn = plan.example(sig.prepareInputs(plan.signature, parseInputsNow(inputs)), outputs);
      return copyData(turn.toJSON()) as unknown as Rec;
    },
    _aiTools: core.tools,
    map,
    mapSettled: (items: Iterable<unknown>, options: MapOptions = {}) => each(items, options, false),
    render: async (input: unknown, options: Settings = {}): Promise<Request> => {
      const s = settingsNow(options);
      const [raw, filled] = binder.bind(input, { check: false });
      const bound = checkInputs(core.interface, raw, core.definition.name, droppedFields(fieldsOf(core.interface, signatureNow(s)), core.own, options));
      await binder.parse(bound, filled);
      const shaped = await shape(bound, s, null, options);
      if (isBaked(shaped.settings.lm)) {
        const job = bakedJob(fn, shaped.settings.lm, shaped.settings, bound, null, answer);
        const values = sig.prepareInputs(job.plan.signature, job.inputs);
        return lm15Request(job.plan.render(job.plan.turn(values), { turns: await pastFor(job.plan, null, true) }), job.model, configOf(job.settings));
      }
      const { plan, model, settings } = planFor(shaped.settings, shaped.instruction);
      const values = sig.prepareInputs(plan.signature, bound);
      if (shaped.offered.length) setOwn(values, "tools", shaped.offered.map((t) => ({ name: t.name, description: t.description ?? null, parameters: t.parameters })));
      return lm15Request(plan.render(plan.turn(values), { turns: await pastFor(plan, null, false) }), model, configOf(settings));
    },
    using: (settings: Settings) => {
      const own = { ...core.own, ...settings };
      checkSettings(own, `${core.definition.name}.using`, fieldsOf(core.interface, sig.signature(definitionNow(own), core.state.instructions)));
      return make({ ...core, own, state: copyData(core.state) });
    },
    state: (): State => copyData(core.state),
    loadState: (state: Partial<State>) => {
      core.state = {
        instructions: state.instructions !== undefined ? state.instructions : core.state.instructions,
        demos: state.demos !== undefined ? state.demos.map((d) => demoOf(d, names, answer)) : core.state.demos,
      };
      return fn;
    },
    probeRequest,
    _core: core,
  });
  return fn;
}

export type { Core };

/** An escalation target answering: it follows only its own `escalateTo`. */
const ESCALATING = new Slot<boolean>();

/**
 * An AI function's call that may be shown again with its steps keeps them
 * (lmcc's turn steps as JSON): one that ran tools, or one made in a
 * conversation (calls.md, *Saw*). Showing it again reads them; none is
 * rebuilt from replies.
 */
function stepsOf(call: calllog.Call, pred: Prediction): unknown[] | null {
  const turn = pred.turn as unknown as { steps?: readonly { kind?: string }[]; toJSON(): unknown };
  if (!turn?.steps?.length) return null;
  const ranTools = turn.steps.some((st) => st.kind === "tool");
  if (!ranTools && !call.turnRun) return null;
  try {
    return copyData(((turn.toJSON() as Rec)["steps"] ?? null) as unknown[] | null);
  } catch {
    return null;
  }
}
