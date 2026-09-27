/**
 * Streaming (contract/streaming.md): the same call, watched while it is made.
 * It retries, runs tools, logs and ends exactly as calling the function does;
 * the stream only shows it.
 *
 * ```ts
 * for await (const piece of haiku.stream("the first snow")) process.stdout.write(piece);
 * const s = solve.stream("10 pencils?");
 * for await (const e of s.events()) console.log(e.kind);
 * await s.result;
 * ```
 */

import type { Call } from "./calllog.ts";
import { toJson } from "./calllog.ts";
import { Cancelled, type Prediction, type Watch } from "./engine.ts";

type Rec = Record<string, unknown>;

export type StreamEvent =
  | { kind: "started"; call: string; function: string; parent: string | null; inputs: Rec }
  | { kind: "text"; call: string; function: string; field: string; answer: boolean; text: string }
  | { kind: "thinking"; call: string; function: string; text: string }
  | { kind: "tool_call"; call: string; function: string; id: string; name: string; input: unknown }
  | { kind: "tool_result"; call: string; function: string; id: string; name: string; output: string }
  | { kind: "retry"; call: string; function: string; reason: string; wait: number | null }
  | { kind: "done"; call: string; function: string; value: unknown }
  | { kind: "failed"; call: string; function: string; error: { type: string; message?: string; code?: string } };

function errorOf(err: unknown): { type: string; message?: string; code?: string } {
  const e = err as { name?: string; message?: string; code?: unknown; constructor?: { name?: string } };
  const type = e?.constructor?.name && e.constructor.name !== "Error" ? e.constructor.name : (e?.name ?? "Error");
  return { type, message: e?.message ?? String(err), ...(typeof e?.code === "string" ? { code: e.code } : {}) };
}

/** A call being made and watched. Iterate it for the answer's text as it arrives. */
export class Stream<A = unknown> implements Watch, AsyncIterable<string> {
  private readonly controller = new AbortController();
  private readonly log: StreamEvent[] = [];
  private waiters: Array<() => void> = [];
  private finished = false;
  private outer: string | null = null;
  private answerText = "";
  /** The call's value (the same as calling the function); rejects with its error. */
  readonly result: Promise<A>;
  /** The whole prediction, when it ends. */
  readonly prediction: Promise<Prediction>;

  constructor(start: (watch: Watch) => Promise<Prediction>, signal?: AbortSignal) {
    if (signal) {                                  // the caller's signal closes the stream too
      if (signal.aborted) this.controller.abort();
      else signal.addEventListener("abort", () => this.controller.abort(), { once: true });
    }
    this.prediction = start(this).finally(() => {
      this.finished = true;
      this.wake();
    });
    this.result = this.prediction.then((p) => p.answer as A);
    this.result.catch(() => undefined);           // an unwatched failure is reported by whoever awaits
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  /** The answer so far (restarts after a retry). Provisional: `result` has the typed value. */
  get text(): string {
    return this.answerText;
  }

  /** Stop the call: no new request starts, and it ends with `Cancelled`. */
  close(): void {
    this.controller.abort();
  }

  check(): void {
    if (this.controller.signal.aborted) throw new Cancelled();
  }

  // ---------------------------------------------------------------- what the engine tells it

  private push(e: StreamEvent): void {
    this.log.push(e);
    this.wake();
  }

  private wake(): void {
    const w = this.waiters;
    this.waiters = [];
    for (const f of w) f();
  }

  private id(call: Call): { call: string; function: string } {
    return { call: call.id, function: call.program().name };
  }

  started(call: Call, inputs: Rec): void {
    this.outer ??= call.id;
    const values: Rec = {};
    for (const [k, v] of Object.entries(inputs)) values[k] = toJson(v)[0];
    this.push({ kind: "started", ...this.id(call), parent: call.parent, inputs: values });
  }

  onText(call: Call, field: string, text: string): void {
    const answer = field === call.program().answer;
    if (call.id === this.outer && answer) this.answerText += text;
    this.push({ kind: "text", ...this.id(call), field, answer, text });
  }

  onThinking(call: Call, text: string): void {
    this.push({ kind: "thinking", ...this.id(call), text });
  }

  onToolCall(call: Call, id: string, name: string, input: unknown): void {
    this.push({ kind: "tool_call", ...this.id(call), id, name, input });
  }

  onToolResult(call: Call, id: string, name: string, output: string): void {
    if (call.id === this.outer) this.answerText = "";
    this.push({ kind: "tool_result", ...this.id(call), id, name, output });
  }

  onRetry(call: Call, reason: string, wait: number | null): void {
    if (call.id === this.outer) this.answerText = "";
    this.push({ kind: "retry", ...this.id(call), reason, wait });
  }

  ended(call: Call, value: unknown, error?: unknown): void {
    if (error !== undefined) this.push({ kind: "failed", ...this.id(call), error: errorOf(error) });
    else this.push({ kind: "done", ...this.id(call), value: toJson(value)[0] });
  }

  // ---------------------------------------------------------------- reading

  /** Every event, in order: the calls inside it too, reasoning, tool calls, retries. */
  async *events(): AsyncGenerator<StreamEvent> {
    let i = 0;
    for (;;) {
      while (i < this.log.length) yield this.log[i++]!;
      if (this.finished) {
        while (i < this.log.length) yield this.log[i++]!;
        return;
      }
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
  }

  /** The answer's text, piece by piece (a retry's discarded text included: `result` is the truth). */
  async *[Symbol.asyncIterator](): AsyncGenerator<string> {
    for await (const e of this.events()) {
      if (e.kind === "text" && e.answer && e.call === this.outer) yield e.text;
    }
    await this.result;
  }

  then<T1 = A, T2 = never>(ok?: ((value: A) => T1 | PromiseLike<T1>) | null, fail?: ((reason: unknown) => T2 | PromiseLike<T2>) | null): Promise<T1 | T2> {
    return this.result.then(ok, fail);
  }
}
