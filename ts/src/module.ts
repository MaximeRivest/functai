/**
 * Modules: your code that calls AI functions, as a program with a declared
 * interface (contract/programs.md). A module's call checks its inputs before
 * its code runs and its outputs when it returns, and is followed as one call
 * with the calls it makes as its children (in the call log and in a stream).
 *
 * ```ts
 * const support = module("support", {
 *   description: "Answer a customer's message.",
 *   input: { message: t.string(), tone: t.string({ default: "kind" }) },   // tone may be left out
 *   output: t.string(),
 *   uses: [topic, answer],
 * }, async ({ message, tone }, { signal }) => answer({ message, tone, topic: await topic(message) }, { signal }));
 *
 * await support("Where is my parcel?");
 * support.interface;       // { description, inputs: [...], outputs: [...] }, as every language writes it
 * ```
 */

import * as lmcc from "lmcc";
import type * as calllog from "./calllog.ts";
import type { CallFields } from "./content.ts";
import { Cancelled } from "./engine.ts";
import { defaultsDocument, definedAt, type AnyAIFunction, type CallArgs, type CallOptions } from "./fn.ts";
import { Binder, declareInput, declareOutput, type InputRule, type InputRules } from "./inputs.ts";
import { checkInputs, checkInterface, checkReturned, dataShape, interfaceSignature, recordedInputs, type Interface, type InterfaceField } from "./interface.ts";
import { droppedFields, recordInputs, runCall } from "./program.ts";
import { checkSettings, effective, type Settings } from "./settings.ts";
import type { FieldSpec, InputValueOf, IsOptional, ValueOf } from "./shapes.ts";
import { Stream } from "./stream.ts";
import { copyData, getOwn, setOwn } from "./values.ts";

type Rec = Record<string, unknown>;
type Fields = Record<string, FieldSpec>;
type Simplify<T> = { [K in keyof T]: T[K] } & {};

/** What `module(name, spec, run)` takes besides its code: its interface, what it uses, and settings. */
export interface ModuleSpec<I extends Fields, O extends Fields | undefined, A extends FieldSpec | undefined> extends Settings {
  /** What it does, in words. */
  description?: string;
  /** Its inputs, by name: `{ message: t.string() }`, zod, or any Standard Schema. `{}` for none. */
  input: I;
  /** Its one output (named `result`), what `run` returns. */
  output?: A;
  /** Several outputs, in order (the last is the answer): `run` returns a record of each by name. */
  outputs?: O;
  /** The AI functions and modules it calls: its version changes when they change. */
  uses?: readonly (AnyAIFunction | AnyModule)[];
  /** The code module the call log files it under (`program.module`; default: the file's name, without extension). */
  definedIn?: string;
}

type LeftOutKeys<I> = { [K in keyof I]: undefined extends ValueOf<I[K]> ? K : never }[keyof I];
/** What a module's code gets: each input by name (an input left out with no default is absent). */
export type ModuleInputs<I> = Simplify<
  { [K in Exclude<keyof I, LeftOutKeys<I>>]: ValueOf<I[K]> } & { [K in LeftOutKeys<I>]?: Exclude<ValueOf<I[K]>, undefined> }>;
type RequiredFields<I> = { [K in keyof I]: IsOptional<I[K]> extends true ? never : K }[keyof I];
/** What a call of a module takes, by name. */
export type ModuleArgs<I> = Simplify<
  { [K in RequiredFields<I>]: InputValueOf<I[K]> } & { [K in Exclude<keyof I, RequiredFields<I>>]?: InputValueOf<I[K]> }>;
type IsUnion<T, U = T> = T extends unknown ? ([U] extends [T] ? false : true) : never;
/** Whether an object type has exactly one key (a record type built at run time, `Record<string, …>`, may have several: no). */
type OneKey<O> = [keyof O] extends [never] ? false : string extends keyof O ? false : true extends IsUnion<keyof O> ? false : true;
/**
 * What a module returns: its one output's value (`output`, or `outputs`
 * naming one), or its several outputs by name (programs.md, "Checking
 * values": one output is the value).
 */
export type ModuleResult<O extends Fields | undefined, A extends FieldSpec | undefined> =
  O extends Fields ? (OneKey<O> extends true ? ValueOf<O[keyof O]> : { [K in keyof O]: ValueOf<O[K]> })
    : A extends FieldSpec ? ValueOf<A> : never;

/** What a module's code is given besides its inputs. */
export interface ModuleContext {
  /** Aborted when the call is cancelled (its caller's signal, its stream closed): pass it on to what you await. */
  readonly signal: AbortSignal;
  /** The call's id in the call log. */
  readonly callId: string;
}

/** A module: call it with its inputs, get its result. */
export interface Module<I extends Rec = Rec, R = unknown> {
  (...args: CallArgs<I>): Promise<R>;
  readonly name: string;
  /** The code module the call log files it under. */
  readonly module: string;
  /** What it takes and gives, as data (the same JSON in every language). */
  readonly interface: Interface;
  /** The interface's signature: the call log's `program.interface`. */
  readonly interfaceId: string;
  /** A fingerprint of its code, the AI functions it uses and its interface (contract/calls.md, "Versions"). */
  readonly version: string;
  readonly uses: readonly (AnyAIFunction | AnyModule)[];
  readonly settings: Settings;
  /** Call it and watch it being made: every event of its call and the calls inside it. */
  stream(...args: CallArgs<I>): Stream<R>;
  /** A copy with other settings. */
  using(settings: Settings): Module<I, R>;
}

/** Any module, whatever its types. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyModule = Module<any, any>;

const SETTING_KEYS = new Set(["lm", "router", "temperature", "maxTokens", "topP", "stop", "seed", "config", "adapter", "template",
  "module", "includeFnName", "capabilities", "retries", "apiRetries", "maxSteps", "toolErrors", "logCalls", "logContent", "caller",
  "cacheReplies", "observers", "journal"]);

interface ModuleCore {
  name: string;
  where: string;
  file?: string;
  line?: number;
  iface: Interface;
  rules: InputRules;
  run: (inputs: Rec, context: ModuleContext) => unknown;
  uses: readonly (AnyAIFunction | AnyModule)[];
  own: Settings;
  /** Inputs whose default is given as a function: its code (calls.md, "Versions", "Defaults"). */
  defaultCode?: Readonly<Record<string, string>>;
  /** Its module was taken from its file's name (no `definedIn`): its calls are known by its file too. */
  topLevel?: boolean;
}

/** The code and AI versions a module reaches: its own code, and what it uses (their own reach, for modules). */
function reach(core: ModuleCore): { code: Record<string, string>; ai: Record<string, string> } {
  const code: Record<string, string> = { [`${core.where}:${core.name}`]: "sha256:" + lmcc.sha256Hex(core.run.toString()) };
  const ai: Record<string, string> = {};
  for (const u of core.uses) {
    const inner = (u as { _reach?: () => { code: Record<string, string>; ai: Record<string, string> } })._reach;
    if (inner) {
      const r = inner();
      Object.assign(code, r.code);
      Object.assign(ai, r.ai);
    } else ai[`${u.module}:${u.name}`] = u.version;
  }
  return { code, ai };
}

function makeModule(core: ModuleCore): AnyModule {
  const { iface, name } = core;
  const interfaceId = interfaceSignature(iface);
  // the interface without its defaults (a computed one must not make a new version each day), and each default by its logic
  const plain: Interface = {
    description: iface.description,
    inputs: iface.inputs.map((f) => ({ ...f, shape: dataShape(f.shape) })),
    outputs: iface.outputs.map((f) => ({ ...f, shape: dataShape(f.shape) })),
  };
  const defaults = defaultsDocument(iface, core.defaultCode);
  const version = () => lmcc.sha256({ ...reach(core), interface: plain, ...(defaults ? { defaults } : {}) } as unknown as lmcc.Json);
  const answer = iface.outputs[iface.outputs.length - 1]!.name;
  const program = (): calllog.Program => ({
    name, kind: "module", module: core.where, version: version(), interface: interfaceId, answer,
    ...(core.file ? { file: core.file } : {}), ...(core.line ? { line: core.line } : {}),
  });
  const fields: CallFields = { inputs: iface.inputs.map((f) => f.name), outputs: iface.outputs.map((f) => f.name), added: [] };
  const names = iface.inputs.map((f) => f.name);
  const binder = new Binder(name, names, core.rules);

  const call = async (arg: unknown, options: CallOptions = {}, stream?: Stream): Promise<unknown> => {
    const { signal, ...extra } = options;
    if (signal?.aborted || stream?.signal.aborted) throw new Cancelled();
    const s = effective({ ...core.own, ...extra });
    // Inputs are checked before the code runs; a refusal is the call's outcome, recorded (programs.md). The record and the
    // events hold only the interface's fields: a value given under another name is named by the refusal, never kept.
    let given: Rec = {};
    let inputs: Rec = {};
    let refused: unknown;
    try {
      given = binder.bind(arg, { fill: false, check: false })[0];
      inputs = checkInputs(iface, given, name, droppedFields(fields, core.own, extra as Settings));
      for (const f of iface.inputs) {                     // a left-out input takes its default as the program declared it (a zod default's own value)
        const fill = core.rules.get(f.name)?.fill;
        if (getOwn(given, f.name) === undefined && fill && Object.hasOwn(inputs, f.name)) {
          let value: unknown;
          try {
            value = copyData(fill.value);
          } catch {
            value = fill.value;
          }
          setOwn(inputs, f.name, value);
        }
      }
    } catch (err) {
      refused = err;
      inputs = recordedInputs(iface, given);
    }
    return runCall<unknown>({
      program, fields, own: core.own, options: extra as Settings, settings: s, stream, signal, inputs: recordInputs(names, inputs),
      ...(refused !== undefined ? { refused } : {}),
      body: async (c) => {
        const values: Rec = { ...inputs };                 // the code's own: what it does with them never reaches the record
        await binder.parse(values, new Set(names.filter((n) => getOwn(given, n) === undefined)));
        const cancelled = c.signal ?? new AbortController().signal;
        if (cancelled.aborted) throw new Cancelled();
        const returned = await core.run(values, { signal: cancelled, callId: c.id });
        if (cancelled.aborted) throw new Cancelled();
        const outputs = checkReturned(iface, returned, name, droppedFields(fields, core.own, extra as Settings));
        return { value: returned, outputs };
      },
    });
  };

  const fn = ((input: unknown, options?: CallOptions) => call(input, options)) as unknown as AnyModule;
  Object.defineProperties(fn, {
    name: { value: name },
    module: { value: core.where },
    interface: { get: () => copyData(iface) },
    defaultCode: { get: () => ({ ...(core.defaultCode ?? {}) }) },
    file: { get: () => (core.topLevel ? core.file : undefined) },
    interfaceId: { value: interfaceId },
    version: { get: version },
    uses: { value: core.uses },
    settings: { get: () => ({ ...core.own }) },
  });
  Object.assign(fn, {
    stream: (input: unknown, options: CallOptions = {}) => new Stream((st) => call(input, options, st), options.signal),
    using: (settings: Settings) => {
      const own = { ...core.own, ...settings };
      checkSettings(own, `${name}.using`, fields);
      return makeModule({ ...core, own });
    },
    _reach: () => reach(core),
  });
  return fn;
}

/**
 * A module: your code that calls AI functions, as a program with a declared
 * interface. `input` and `output` (or `outputs`) say what it takes and gives,
 * as `ai()` does, and are checked on every call (`InterfaceError`); `run`
 * gets the inputs by name (a left-out input takes its default), and the call's
 * `signal`. `uses` lists the AI functions and modules it calls, so its
 * version changes when they are improved. A field `t.opaque()` takes values
 * with no JSON form, unchecked.
 */
export function module<I extends Fields, O extends Fields | undefined = undefined, A extends FieldSpec | undefined = undefined>(
  name: string,
  spec: ModuleSpec<I, O, A>,
  run: (inputs: ModuleInputs<I>, context: ModuleContext) => ModuleResult<O, A> | Promise<ModuleResult<O, A>>,
): Module<ModuleArgs<I>, ModuleResult<O, A>> {
  if (typeof name !== "string" || !name) throw new TypeError('module(name, { input, output }, run): the name comes first: module("support", { ... }, run)');
  if (!spec || typeof spec !== "object" || !spec.input || typeof spec.input !== "object") {
    throw new TypeError(`module("${name}", { input: { ... }, output }, run): the inputs are required ({} for none)`);
  }
  if (typeof run !== "function") throw new TypeError(`module("${name}", spec, run): run is the module's code, a function of its inputs`);
  if (spec.output !== undefined && spec.outputs !== undefined) throw new TypeError(`${name}: give output (one) or outputs (several), not both`);
  if (spec.output === undefined && spec.outputs === undefined) {
    throw new TypeError(`module("${name}", { input, output }, run): declare what it returns: output (one) or outputs (several); t.json() for any JSON, t.opaque() for anything`);
  }
  const rules = new Map<string, InputRule>();
  const defaultCode: Record<string, string> = {};
  const inputs: InterfaceField[] = Object.entries(spec.input).map(([field, s]) => {
    const declared = declareInput(field, s, `${name}.input.${field}`, "module");
    rules.set(field, declared.rule);
    if (declared.code !== undefined) setOwn(defaultCode, field, declared.code);
    return declared.field;
  });
  const outputSpecs: [string, FieldSpec][] = spec.outputs ? Object.entries(spec.outputs) : [["result", spec.output as FieldSpec]];
  const outputs: InterfaceField[] = outputSpecs.map(([field, s]) => declareOutput(field, s, `${name}.outputs.${field}`, "module"));
  const iface = checkInterface({ description: spec.description ?? "", inputs, outputs }, { where: `module("${name}")` });
  const own: Settings = {};
  for (const [k, v] of Object.entries(spec)) if (SETTING_KEYS.has(k)) (own as Rec)[k] = v;
  checkSettings(own, `module("${name}")`, { inputs: inputs.map((f) => f.name), outputs: outputs.map((f) => f.name), added: [] });
  const where = definedAt(module);
  return makeModule({
    name, where: spec.definedIn ?? where.module ?? "main", file: where.file, line: where.line, iface, rules,
    run: run as ModuleCore["run"], uses: [...(spec.uses ?? [])], own, defaultCode, topLevel: spec.definedIn === undefined,
  }) as unknown as Module<ModuleArgs<I>, ModuleResult<O, A>>;
}
