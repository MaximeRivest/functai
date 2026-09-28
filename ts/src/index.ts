/**
 * functai for TypeScript and JavaScript: typed functions whose body is a
 * language model call. Write the signature; a model writes the body; you
 * measure how often it is right, and make it better.
 *
 * It follows the FunctAI contract (../contract in the repository), so a
 * function has the same version, the same call log and the same saved form
 * as the same function in Python.
 */

import * as calllog from "./calllog.ts";
import type { AnyAIFunction as AIFunction } from "./fn.ts";
import type { AnyModule } from "./module.ts";
import type { Prediction, Tool } from "./engine.ts";
import { effective, type Settings } from "./settings.ts";
import { readField, type FieldSpec, type ValueOf } from "./shapes.ts";

export { ai, type AIFunction, type AnyAIFunction, type CallOptions, type Column, type Definition, type Demo, type Expected, type Input,
  type InputOf, type MapOptions, type OutputOf, type Row, type State } from "./fn.ts";
export { module, type AnyModule, type Module, type ModuleArgs, type ModuleContext, type ModuleInputs, type ModuleResult,
  type ModuleSpec } from "./module.ts";
export { t, describe, type Json, type Shape, type FieldSpec, type StandardSchemaLike, type ValueOf, type ZodLike } from "./shapes.ts";
export { configure, withSettings, type Settings } from "./settings.ts";
export { SettingError, type LogContent } from "./content.ts";
export { InterfaceError, interfaceSignature, type Interface, type InterfaceCode, type InterfaceField } from "./interface.ts";
export { Prediction, StepLimit, Cancelled, type Tool } from "./engine.ts";
export { Stream, PredictionStream, EventUnknown, views, type EventsOptions, type Form, type View } from "./stream.ts";
export {
  Follower, MemoryStore, Replay, keptLog, replay, resume, settle, receivers, stateOf, positionOf, samePosition,
  type AppendAnswer, type CallState, type ClaimAnswer, type DoneEvent, type ErrorInfo, type EventSource, type EventStore,
  type FailedEvent, type FollowResult, type KeptFields, type LogState, type Position, type ProgramInfo, type ReadAnswer,
  type RequestEvent, type RetryEvent, type SawEntry, type StartedEvent, type StoreCode, type StreamEvent, type TextEvent,
  type ThinkingEvent, type ToolCallEvent, type ToolResultEvent,
} from "./events.ts";
export { JournalError, type Journal, type JournalCode, type JournalSetting, type Observer, type Outcome } from "./log.ts";
export { sawOf, keepsSaw, SawUnknown, type SawCode } from "./saw.ts";
export { evaluate, exactMatch, interval, Evaluation, type EvaluateOptions, type Metric, type RowResult, type Summary } from "./evaluate.ts";
export { labeledFewShot, bootstrapFewShot, type BootstrapOptions, type LabeledOptions } from "./optimize.ts";
export { gepa, type GepaOptions, type GepaResult, type Feedback, type Trial } from "./gepa.ts";
export { load, fromManifest, save, toManifest, describeSaved, LoadRefused } from "./saved.ts";
export { capabilities } from "./models.ts";
export { clearCache, type ReplyCache } from "./cache.ts";
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

/** One logged call (contract/calls.md, formats 1 and 2), as TypeScript reads it; `record` is the line as written. */
export interface LoggedCall {
  id: string;
  parent: string | null;
  root: string;
  program: calllog.Program;
  started: Date;
  seconds: number;
  /** Whether every value was written. */
  content: boolean;
  /** The fields whose values were not written (when `content` is false), by kind. */
  omitted: { inputs: string[]; outputs: string[] } | null;
  /** The inputs written (all of them when `content` is true; null when none was). */
  inputs: Rec | null;
  /** The outputs written (null when none was, or the call failed before an answer). */
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
  const omitted = (r["omitted"] ?? null) as LoggedCall["omitted"];
  return {
    id: r["id"] as string, parent: (r["parent"] as string | null) ?? null, root: r["root"] as string,
    program: r["program"] as calllog.Program, started: new Date(r["started"] as string), seconds: r["seconds"] as number,
    content: r["content"] === true, omitted: r["content"] === true ? null : omitted ?? { inputs: [], outputs: [] },
    inputs: (r["inputs"] as Rec | undefined) ?? null, outputs: (r["outputs"] as Rec | undefined) ?? null,
    error: (r["error"] as LoggedCall["error"] | undefined) ?? null, model: (r["model"] as string | null) ?? null,
    usage: Object.fromEntries(Object.entries((r["usage"] ?? {}) as Record<string, number>).map(([k, v]) => [camel(k), v])),
    confidence: (r["confidence"] as number | null | undefined) ?? null, caller: (r["caller"] ?? {}) as Rec, record: r,
  };
}

/** Every logged call (of one function, when given), oldest first. */
export function calls(fn?: AIFunction | AnyModule | string, opts: { folder?: string; since?: Date } = {}): LoggedCall[] {
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
  /** Calls of another signature or interface: the inputs or outputs changed since. */
  otherSignature: number;
  /** Calls whose inputs were not all written as data. */
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
export function rated(fn: AIFunction | AnyModule | string, opts: { folder?: string; by?: string; since?: Date; records?: readonly Rec[] } = {}): { rows: Rec[]; leftOut: LeftOut } {
  const records = opts.records
    ? [opts.records.filter((r) => "functai_call" in r), opts.records.filter((r) => "functai_rating" in r)] as [Rec[], Rec[]]
    : calllog.read(opts.folder ?? calllog.folderOf(effective({}).logCalls ?? true), { since: opts.since ?? null });
  const key = typeof fn === "string" ? { name: fn }
    : { name: fn.name, module: fn.module, interface: fn.interfaceId, ...("signatureId" in fn ? { signature: (fn as AIFunction).signatureId } : {}) };
  const [rows, left] = calllog.ratedRows(records[0], records[1], { ...key, by: opts.by });
  return { rows, leftOut: { otherSignature: left.other_signature, noContent: left.no_content, noAnswer: left.no_answer } };
}
