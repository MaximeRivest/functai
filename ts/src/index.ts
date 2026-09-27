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
import { readField, type FieldSpec } from "./shapes.ts";

export { ai, type AIFunction, type AnyAIFunction, type Definition, type Demo, type State } from "./fn.ts";
export { t, describe, type Shape, type FieldSpec, type ValueOf, type ZodLike } from "./shapes.ts";
export { configure, withSettings, type Settings } from "./settings.ts";
export { Prediction, StepLimit, Cancelled, type Tool } from "./engine.ts";
export { Stream, type StreamEvent } from "./stream.ts";
export { evaluate, exactMatch, interval, Evaluation, type EvaluateOptions, type Metric, type RowResult, type Summary } from "./evaluate.ts";
export { labeledFewShot, bootstrapFewShot, type BootstrapOptions } from "./optimize.ts";
export { load, fromManifest, save, toManifest, LoadRefused } from "./saved.ts";
export { capabilities } from "./models.ts";
export { normalize, casefold } from "./text.ts";
export { VERSION } from "./calllog.ts";
export { Refusal, isRefusal } from "lmcc";

type Rec = Record<string, unknown>;

/** A tool the model may call: `tool({ name, description, input: { city: t.string() } }, ({ city }) => …)`. */
export function tool<I extends Record<string, FieldSpec>>(
  spec: { name: string; description?: string; input: I },
  run: (input: { [K in keyof I]: unknown }) => unknown,
): Tool {
  const properties: Rec = {};
  for (const [name, f] of Object.entries(spec.input)) {
    const { shape, desc } = readField(f, `${spec.name}.${name}`);
    properties[name] = desc ? { ...shape, description: desc } : shape;
  }
  // no additionalProperties: Gemini refuses the keyword in function declarations
  return { name: spec.name, description: spec.description ?? "", parameters: { type: "object", properties, required: Object.keys(spec.input) }, run: run as Tool["run"] };
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

/** Every logged call (of one function, when given), oldest first. */
export function calls(fn?: AIFunction | string, opts: { folder?: string; since?: Date } = {}): Rec[] {
  const [all] = calllog.read(opts.folder ?? calllog.folderOf(effective({}).logCalls ?? true), { since: opts.since ?? null });
  const name = typeof fn === "string" ? fn : fn?.name;
  const module = typeof fn === "object" || typeof fn === "function" ? fn?.module : undefined;
  return all
    .filter((c) => !name || ((c["program"] as Rec)["name"] === name && (module === undefined || (c["program"] as Rec)["module"] === module)))
    .sort((a, b) => String(a["started"]).localeCompare(String(b["started"])));
}

/**
 * Rows with known answers from people's ratings (contract/calls.md, "Rows
 * with known answers"): the inputs, the right answer, and who said so.
 * Ready for `evaluate` and the optimizers. Pass `records` to read them from
 * somewhere else than the log folder.
 */
export function rated(fn: AIFunction | string, opts: { folder?: string; by?: string; since?: Date; records?: readonly Rec[] } = {}): { rows: Rec[]; leftOut: calllog.LeftOut } {
  const records = opts.records
    ? [opts.records.filter((r) => "functai_call" in r), opts.records.filter((r) => "functai_rating" in r)] as [Rec[], Rec[]]
    : calllog.read(opts.folder ?? calllog.folderOf(effective({}).logCalls ?? true), { since: opts.since ?? null });
  const key = typeof fn === "string" ? { name: fn } : { name: fn.name, module: fn.module, signature: fn.signatureId };
  const [rows, leftOut] = calllog.ratedRows(records[0], records[1], { ...key, by: opts.by });
  return { rows, leftOut };
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
