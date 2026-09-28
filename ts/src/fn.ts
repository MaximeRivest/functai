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
import * as bridge from "lmcc/lm15";
import type { Request } from "@lm15/lm15";
import type * as calllog from "./calllog.ts";
import { checkLogContent, type CallFields } from "./content.ts";
import { Cancelled, run as runEngine, Prediction, type Router, type Tool } from "./engine.ts";
import { Binder, declareInput, rulesOf, type InputRule } from "./inputs.ts";
import { checkInterface, interfaceSignature, type Interface } from "./interface.ts";
import { bind } from "./layouts.ts";
import { adjustSettings, callCapabilities, defaultModel, defaultRouter, modelString, PROBE } from "./models.ts";
import { runCall } from "./program.ts";
import { readField, type FieldSpec, type InputValueOf, type IsOptional, type ValueOf } from "./shapes.ts";
import { configOf, effective, type Settings } from "./settings.ts";
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

/** Options of `fn.map`: calls in flight at once (default 8), and the call options. */
export interface MapOptions extends CallOptions {
  concurrency?: number;
}

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
  /** The exact request a call would send, without sending it. */
  render(...args: CallArgs<I, Settings>): Request;
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

function demoOf(item: Demo | Rec, inputNames: readonly string[], answer: string): Demo {
  if (typeof item === "object" && item !== null && "signature" in item && "steps" in item) return item as unknown as Demo;   // a recorded turn
  if (typeof item === "object" && item !== null && "inputs" in item && "outputs" in item
    && typeof item["inputs"] === "object" && !inputNames.includes("inputs")) {
    return { inputs: { ...(item["inputs"] as Rec) }, outputs: { ...(item["outputs"] as Rec) } };
  }
  if (Array.isArray(item) && item.length === 2) {
    const [i, o] = item as unknown as [unknown, unknown];
    return {
      inputs: typeof i === "object" && i !== null && !Array.isArray(i) ? { ...(i as Rec) } : { [inputNames[0]!]: i },
      outputs: typeof o === "object" && o !== null && !Array.isArray(o) ? { ...(o as Rec) } : { [answer]: o },
    };
  }
  const inputs: Rec = {};
  const outputs: Rec = {};
  for (const [k, v] of Object.entries(item)) (inputNames.includes(k) ? inputs : outputs)[k] = v;
  return { inputs, outputs };
}

interface Core {
  definition: sig.Definition;
  /** What it takes and gives, as data; its inputs' rules follow it unless `rules` says more (a schema). */
  interface: Interface;
  /** By input name; a loaded function has none, and its rules come from its interface. */
  rules?: Record<string, InputRule>;
  own: Settings;
  tools: readonly Tool[];
  module: string;
  file?: string;
  line?: number;
  saved?: string;
  state: { instructions: string | null; demos: Demo[] };
}

const SETTING_KEYS = new Set(["lm", "router", "temperature", "maxTokens", "topP", "stop", "seed", "config", "adapter", "template",
  "module", "includeFnName", "capabilities", "retries", "apiRetries", "maxSteps", "toolErrors", "logCalls", "logContent", "caller",
  "cacheReplies", "observers", "journal"]);

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
  const rules: Record<string, InputRule> = {};
  const inputs = Object.entries(def.input).map(([field, spec]) => {
    const { field: declared, rule } = declareInput(field, spec, `${name}.input.${field}`, "ai");
    rules[field] = rule;
    return { name: field, shape: declared.shape, desc: declared.desc ?? null, ...(rule.optional ? { optional: true } : {}) };
  });
  const outputSpecs: [string, FieldSpec][] = def.outputs
    ? Object.entries(def.outputs)
    : [["result", (def.output ?? { type: "string" }) as FieldSpec]];
  if (!outputSpecs.length) throw new TypeError(`${name}: outputs is empty`);
  const outputs = outputSpecs.map(([field, spec]) => ({ name: field, ...readField(spec, `${name}.outputs.${field}`) }));
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
  const iface = checkInterface(interfaceOf(definition), `ai("${name}")`, { ai: true });
  checkLogContent(own.logContent, `ai("${name}")`, fieldsOf(iface, signature));
  const core: Core = {
    definition: { ...definition, cot: false, includeName: true }, interface: iface, rules, own, tools: [...(def.tools ?? [])],
    module: moduleName, file: where.file, line: where.line, state: { instructions: def.instructions ?? null, demos: [] },
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

  const rules: Record<string, InputRule> = core.rules ?? rulesOf(core.interface.inputs);
  const binder = new Binder(core.definition.name, names, rules);
  /** A call's argument as inputs by name; inputs left out get their default. */
  const bindInputs = (arg: unknown): Rec => binder.bind(arg)[0];
  const parseInputs = async (arg: unknown): Promise<Rec> => {
    const [bound, filled] = binder.bind(arg);
    await binder.parse(bound, filled);
    return bound;
  };
  const parseInputsNow = (arg: unknown): Rec => {
    const [bound, filled] = binder.bind(arg);
    binder.parse(bound, filled, true);
    return bound;
  };

  /** Worked examples as example turns for this plan (contract/functions.md, "Worked examples"). */
  const pastTurns = (plan: lmcc.Plan): lmcc.Turn[] => {
    const plain = new Set(plan.signature.fields.filter((f) => f.direction === "input" && f.purpose === "plain").map((f) => f.name));
    const kept = new Set(plan.signature.fields.filter((f) => f.direction === "output" && (f.purpose === "plain" || f.purpose === "reasoning")).map((f) => f.name));
    const out: lmcc.Turn[] = [];
    for (const d of core.state.demos) {
      if ("signature" in d && "steps" in d) {
        const recorded = d as unknown as { signature: string };
        if (recorded.signature === plan.fingerprint) {
          try {
            out.push(plan.loadTurn(d));
            continue;
          } catch {
            // read as its inputs and outputs below
          }
        }
      }
      const ins = sig.prepareInputs(plan.signature, Object.fromEntries(Object.entries(d.inputs ?? {}).filter(([k]) => plain.has(k))));
      const outs = Object.fromEntries(Object.entries(d.outputs ?? {}).filter(([k]) => kept.has(k)));
      if (!Object.keys(outs).length) continue;
      try {
        out.push(plan.example(ins, outs));
      } catch (err) {
        if (!lmcc.isRefusal(err)) throw err;
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
    if (core.tools.length) values["tools"] = core.tools.map((t) => ({ name: t.name, description: t.description ?? null, parameters: t.parameters }));
    return plan.render(plan.turn(values), { turns: pastTurns(plan) }).request("probe");
  };

  const version = (): string => {
    const s = settingsNow();
    const key = JSON.stringify(["version", layoutKey(s), s.module ?? null, s.includeFnName ?? null, core.state]);
    let v = cache.get(key) as string | undefined;
    if (!v) {
      let r: string;
      try {
        r = lmcc.sha256(probeRequest());
      } catch (err) {
        if (!lmcc.isRefusal(err)) throw err;
        r = `refused:${(err as lmcc.Refusal).code}`;
      }
      v = lmcc.sha256({ request: r });
      cache.set(key, v);
    }
    return v;
  };

  const route = (s: Settings) => {
    const lm = s.lm ?? defaultModel(env());
    if (!lm) throw new Error("no model configured, and no API key found to pick one: set OPENAI_API_KEY (or ANTHROPIC_API_KEY, GEMINI_API_KEY, …), or name a model: configure({ lm: \"gpt-4.1-mini\" })");
    const model = modelString(lm);
    const router = (s.router ?? defaultRouter()) as Router & { resolve(m: string): { provider: string; model: string } };
    const resolution = router.resolve(model);
    return { model, router, provider: resolution.provider, wire: resolution.model };
  };

  const planFor = <S extends Settings>(given: S) => {
    const r = route(given);
    const s = adjustSettings(given, r.provider, r.wire);
    const caps = callCapabilities(r.provider, r.wire, { temperature: s.temperature, capabilities: s.capabilities });
    const key = JSON.stringify(["plan", layoutKey(s), caps, r.provider, s.module ?? null, s.includeFnName ?? null, core.state.instructions]);
    let plan = cache.get(key) as lmcc.Plan | undefined;
    if (!plan) {
      plan = bind(layout(s), signatureNow(s), caps, r.provider);
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

  const predict = async (arg: unknown, options: CallOptions = {}, stream?: PredictionStream): Promise<Prediction> => {
    const { signal, ...extra } = options;
    if (signal?.aborted || stream?.signal.aborted) throw new Cancelled();
    const bound = await parseInputs(arg);
    const s = settingsNow(extra);
    return runCall<Prediction>({
      program: () => program(s), fields: fieldsOf(core.interface, signatureNow(s)), own: core.own, options: extra as Settings,
      settings: s, stream, signal, inputs: bound,
      body: async (call) => {
        const { plan, model, router, provider, settings } = planFor(s);
        call.provider = provider;
        const pred = await runEngine({
          function: core.definition.name, plan, past: pastTurns(plan), inputs: bound, settings, router, model,
          tools: core.tools, call, answer,
        });
        return { value: pred, outputs: pred.outputs as Rec, shown: pred.answer, returned: pred.answer };
      },
    });
  };

  const map = async (items: Iterable<unknown>, options: MapOptions = {}): Promise<unknown[]> => {
    const { concurrency = 8, ...call } = options;
    const list = [...items];
    const out: unknown[] = new Array(list.length);
    const stop = new AbortController();
    const signal = call.signal ? AbortSignal.any([call.signal, stop.signal]) : stop.signal;
    let next = 0;
    const worker = async () => {
      while (next < list.length && !signal.aborted) {
        const i = next++;
        out[i] = (await predict(list[i], { ...call, signal })).answer;
      }
    };
    try {
      await Promise.all(Array.from({ length: Math.max(1, Math.min(concurrency, list.length || 1)) }, worker));
    } catch (err) {
      stop.abort();                                  // the first failure: start no more
      throw err;
    }
    if (call.signal?.aborted) throw new Cancelled();
    return out;
  };

  const fn = (async (input: unknown, options?: CallOptions) => (await predict(input, options)).answer) as unknown as AIFunction;
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
      get: () => core.state.demos.map((d) => structuredClone(d)),
      set: (items: Array<Demo | Rec>) => { core.state = { ...core.state, demos: (items ?? []).map((d) => demoOf(d, names, answer)) }; },
    },
    version: { get: version },
    signatureId: { get: () => sig.signatureId(signatureNow()) },
    interface: { get: () => structuredClone(core.interface) },
    interfaceId: { value: interfaceId },
    settings: { get: () => ({ ...core.own }) },
    definition: { get: () => core.definition },
    tools: { get: () => core.tools },
    saved: { get: () => core.saved },
  };
  Object.defineProperties(fn, props);
  Object.assign(fn, {
    predict: (input: unknown, options?: CallOptions) => predict(input, options),
    stream: (input: unknown, options: CallOptions = {}) => {
      bindInputs(input);                                 // wrong arguments fail here, not in the stream
      return new PredictionStream<unknown, Prediction>((s) => predict(input, options, s as PredictionStream), (p) => p.answer, options.signal);
    },
    map,
    render: (input: unknown, options: Settings = {}): Request => {
      const s = settingsNow(options);
      const { plan, model, settings } = planFor(s);
      const values = sig.prepareInputs(plan.signature, parseInputsNow(input));
      if (core.tools.length) values["tools"] = core.tools.map((t) => ({ name: t.name, description: t.description ?? null, parameters: t.parameters }));
      return bridge.request(plan.render(plan.turn(values), { turns: pastTurns(plan) }), { model, config: configOf(settings) });
    },
    using: (settings: Settings) => {
      const own = { ...core.own, ...settings };
      checkLogContent(own.logContent, `${core.definition.name}.using`, fieldsOf(core.interface, sig.signature(definitionNow(own), core.state.instructions)));
      return make({ ...core, own, state: structuredClone(core.state) });
    },
    state: (): State => structuredClone(core.state),
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
