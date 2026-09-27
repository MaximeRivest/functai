/**
 * functai for TypeScript and JavaScript: typed functions whose body is a
 * language model call. Write the signature; a model writes the body; you
 * measure how often it is right, and make it better.
 *
 * It follows the FunctAI contract (../contract in the repository), so a
 * function has the same version, the same call log and the same saved form
 * as the same function in Python.
 */

import * as lmcc from "lmcc";
import * as calllog from "./calllog.ts";
import type { AnyAIFunction as AIFunction } from "./fn.ts";
import type { Prediction, Tool } from "./engine.ts";
import { effective, type Settings } from "./settings.ts";
import { readField, type FieldSpec, type ValueOf } from "./shapes.ts";

export { ai, type AIFunction, type AnyAIFunction, type CallOptions, type Column, type Definition, type Demo, type Expected, type Input,
  type InputOf, type MapOptions, type OutputOf, type Row, type State } from "./fn.ts";
export { t, describe, type Shape, type FieldSpec, type StandardSchemaLike, type ValueOf, type ZodLike } from "./shapes.ts";
export { configure, withSettings, type Settings } from "./settings.ts";
export { Prediction, StepLimit, Cancelled, type Tool } from "./engine.ts";
export { Stream, type StreamEvent } from "./stream.ts";
export { evaluate, exactMatch, interval, Evaluation, type EvaluateOptions, type Metric, type RowResult, type Summary } from "./evaluate.ts";
export { labeledFewShot, bootstrapFewShot, type BootstrapOptions, type LabeledOptions } from "./optimize.ts";
export { gepa, type GepaOptions, type GepaResult, type Feedback, type Trial } from "./gepa.ts";
export { load, fromManifest, save, toManifest, LoadRefused } from "./saved.ts";
export { capabilities } from "./models.ts";
export { normalize, casefold } from "./text.ts";
export { VERSION } from "./calllog.ts";
export { Refusal, isRefusal } from "lmcc";

type Rec = Record<string, unknown>;

/**
 * A tool the model may call: your function, with a name, a sentence and its
 * input's types. `tool("lookup_order", { description, input: { order: t.string() } }, ({ order }) => …)`;
 * `order` is typed from its shape.
 */
export function tool<I extends Record<string, FieldSpec>>(
  name: string,
  spec: { description?: string; input: I },
  run: (input: { [K in keyof I]: ValueOf<I[K]> }) => unknown,
): Tool {
  const properties: Rec = {};
  for (const [field, f] of Object.entries(spec.input)) {
    const { shape, desc } = readField(f, `${name}.${field}`);
    properties[field] = desc ? { ...shape, description: desc } : shape;
  }
  // no additionalProperties: Gemini refuses the keyword in function declarations
  return { name, description: spec.description ?? "", parameters: { type: "object", properties, required: Object.keys(spec.input) }, run: run as Tool["run"] };
}

/**
 * Say whether a call's answer is right, and if not, what it should have been
 * (contract/calls.md, "A rating record"). `call` is a Prediction or its
 * `callId`. The rating is written to the call log folder.
 */
export function rate(call: Prediction | string, verdict?: calllog.Verdict, opts: calllog.RateOptions = {}): Rec {
  const id = typeof call === "string" ? call : call.callId;
  return calllog.rating(id, verdict, opts, effective({}));
}

/** One logged call (contract/calls.md), as TypeScript reads it; `record` is the line as written. */
export interface LoggedCall {
  id: string;
  parent: string | null;
  root: string;
  program: calllog.Program;
  started: Date;
  seconds: number;
  /** null when the call was logged without content (`logContent: false`). */
  inputs: Rec | null;
  outputs: Rec | null;
  error: { type: string; message?: string; code?: string } | null;
  model: string | null;
  /** Tokens, summed over the call's requests: `inputTokens`, `outputTokens`, `totalTokens`, … */
  usage: Record<string, number>;
  confidence: number | null;
  /** Who called: `{ evaluation: "…" }`, `{ optimization: "…" }`, `$FUNCTAI_CALLER`'s keys. */
  caller: Rec;
  record: Rec;
}

const camel = (k: string) => k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

function loggedCall(r: Rec): LoggedCall {
  return {
    id: r["id"] as string, parent: (r["parent"] as string | null) ?? null, root: r["root"] as string,
    program: r["program"] as calllog.Program, started: new Date(r["started"] as string), seconds: r["seconds"] as number,
    inputs: (r["inputs"] as Rec | undefined) ?? null, outputs: (r["outputs"] as Rec | undefined) ?? null,
    error: (r["error"] as LoggedCall["error"] | undefined) ?? null, model: (r["model"] as string | null) ?? null,
    usage: Object.fromEntries(Object.entries((r["usage"] ?? {}) as Record<string, number>).map(([k, v]) => [camel(k), v])),
    confidence: (r["confidence"] as number | null | undefined) ?? null, caller: (r["caller"] ?? {}) as Rec, record: r,
  };
}

/** Every logged call (of one function, when given), oldest first. */
export function calls(fn?: AIFunction | string, opts: { folder?: string; since?: Date } = {}): LoggedCall[] {
  const [all] = calllog.read(opts.folder ?? calllog.folderOf(effective({}).logCalls ?? true), { since: opts.since ?? null });
  const name = typeof fn === "string" ? fn : fn?.name;
  const module = typeof fn === "object" || typeof fn === "function" ? fn?.module : undefined;
  return all
    .filter((c) => !name || ((c["program"] as Rec)["name"] === name && (module === undefined || (c["program"] as Rec)["module"] === module)))
    .sort((a, b) => String(a["started"]).localeCompare(String(b["started"])))
    .map(loggedCall);
}

/** Rated calls that could not become rows, and why. */
export interface LeftOut {
  /** Calls of another signature: the inputs or outputs changed since. */
  otherSignature: number;
  /** Calls logged without their content. */
  noContent: number;
  /** Ratings that say neither what the answer was nor what it should have been. */
  noAnswer: number;
}

/**
 * Rows with known answers from people's ratings (contract/calls.md, "Rows
 * with known answers"): the inputs, the right answer, and who said so.
 * Ready for `evaluate` and the optimizers. Pass `records` to read them from
 * somewhere else than the log folder.
 */
export function rated(fn: AIFunction | string, opts: { folder?: string; by?: string; since?: Date; records?: readonly Rec[] } = {}): { rows: Rec[]; leftOut: LeftOut } {
  const records = opts.records
    ? [opts.records.filter((r) => "functai_call" in r), opts.records.filter((r) => "functai_rating" in r)] as [Rec[], Rec[]]
    : calllog.read(opts.folder ?? calllog.folderOf(effective({}).logCalls ?? true), { since: opts.since ?? null });
  const key = typeof fn === "string" ? { name: fn } : { name: fn.name, module: fn.module, signature: fn.signatureId };
  const [rows, left] = calllog.ratedRows(records[0], records[1], { ...key, by: opts.by });
  return { rows, leftOut: { otherSignature: left.other_signature, noContent: left.no_content, noAnswer: left.no_answer } };
}

/**
 * A module: your code that calls AI functions, followed as one call with
 * theirs as its children (in the call log and in a stream). `uses` lists the
 * AI functions it calls, so its version changes when they are improved.
 */
export function module<A extends unknown[], R>(
  name: string, run: (...args: A) => Promise<R>, opts: { uses?: readonly AIFunction[]; definedIn?: string; settings?: Settings } = {},
): ((...args: A) => Promise<R>) & { readonly version: string } {
  const where = opts.definedIn ?? "main";
  const version = () => lmcc.sha256({
    code: { [`${where}:${name}`]: lmcc.sha256Hex(run.toString()) },
    ai: Object.fromEntries((opts.uses ?? []).map((f) => [`${f.module}:${f.name}`, f.version])),
  });
  const program = (): calllog.Program => ({ name, kind: "module", module: where, version: version(), answer: "result" });
  const wrapped = async (...args: A): Promise<R> => {
    const inputs: Rec = {};
    args.forEach((a, i) => { inputs[`arg${i}`] = a; });
    const call = calllog.start(program, effective(opts.settings ?? {}), inputs);
    return calllog.current.run(call, async () => {
      try {
        const out = await run(...args);
        calllog.finish(call, { returned: out, hasReturned: true });
        return out;
      } catch (err) {
        calllog.finish(call, { error: err });
        throw err;
      }
    });
  };
  Object.defineProperty(wrapped, "version", { get: version });
  return wrapped as typeof wrapped & { readonly version: string };
}
