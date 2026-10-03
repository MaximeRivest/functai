/**
 * Streaming (contract/streaming.md): the same call, watched while it is made.
 * It retries, runs tools, logs and ends exactly as calling the program does;
 * the stream only shows it, as the call tree's log of events (format 2).
 *
 * ```ts
 * for await (const piece of haiku.stream("the first snow")) process.stdout.write(piece);
 * const s = solve.stream("10 pencils?");
 * for await (const e of s.events()) console.log(e.seq, e.kind);
 * await s;                                                  // the value, as calling it gives
 * for await (const e of s.events({ form: "kept" })) send(e); // what the log may keep
 * ```
 */

import { Cancelled, type Approval } from "./errors.ts";
import { answerHere, waitingHere } from "./tools.ts";
import {
  keptEvent, known, positionOf, Relink, resume, type KeptFields, type Position, type ReadAnswer, type EventSource,
  type StartedEvent, type StreamEvent,
} from "./events.ts";
import { JournalError, type Node, type Watcher } from "./log.ts";
import { View as NamedView, type ViewName } from "./views.ts";

type Rec = Record<string, unknown>;

/**
 * What one kind of reader may see of a log (streaming.md, "Views"). A view
 * shows a call whole or not at all: a shown call's `started`, `request`,
 * `retry`, `done` and `failed` are always shown (a view may leave out the
 * values they hold, never the events), and its other events as `shows`
 * says. It never alters a value it shows.
 */
export interface View {
  /** Whether a call is shown (default: every call). */
  readonly calls?: (started: StartedEvent) => boolean;
  /** An event of a shown call as this view shows it: the same, with values left out, or null (not shown). */
  readonly shows?: (event: StreamEvent) => StreamEvent | null;
}

/** Views every host can use (stage 3 names more). */
export const views = {
  /** A program's boundary: the call the stream was opened on, its start and its end (streaming.md: until stage 3, a module's view). */
  boundary: (call: string): View => ({ calls: (s) => s.call === call, shows: () => null }),
} as const;

/** Which form of the log to read: the whole log (every value), or the kept form (what `logContent` lets the log keep). */
export type Form = "whole" | "kept";

export interface EventsOptions {
  /** `"whole"` (the default) or `"kept"`. */
  readonly form?: Form;
  /** A view made from that form: one of your own, or a named one (`"outside"`: a caller who sees only the program's boundary). */
  readonly view?: View | ViewName;
  /** Resume after this event (its position in this form); null or absent: from the first. */
  readonly after?: Position | null;
}

/** A read of a form refused: the stream does not have that event in that form (`event-unknown`). */
export class EventUnknown extends Error {
  readonly code = "event-unknown";
  constructor(message: string) {
    super(message);
    this.name = "EventUnknown";
  }
}

const ALWAYS = new Set(["started", "request", "retry", "done", "failed"]);

/** A form made from the whole events, one at a time. */
class Former {
  private readonly link = new Relink();
  private readonly shown = new Map<string, boolean>();
  private readonly opts: EventsOptions;
  private readonly named: NamedView | null;
  constructor(opts: EventsOptions, answerFrom: string | null = null) {
    this.opts = opts;
    this.named = typeof opts.view === "string" ? new NamedView(opts.view, { answerFrom }) : null;
  }

  take(e: StreamEvent, keep: KeptFields, program: { kind: string; answer: string }): StreamEvent | null {
    let x: Rec | null = this.opts.form === "kept" ? keptEvent(e, keep, program) : known(e);
    if (!x) return null;
    if (this.named) return this.named.apply(x) as unknown as StreamEvent | null;
    const view = this.opts.view as View | undefined;
    if (view) {
      if (e.kind === "started") this.shown.set(e.call, view.calls ? view.calls(x as StartedEvent) : true);
      if (!this.shown.get(e.call)) return null;
      const shown = view.shows ? view.shows(x as unknown as StreamEvent) : x as unknown as StreamEvent;
      if (shown) x = known(shown);
      else if (!ALWAYS.has(e.kind)) return null;
    }
    return this.link.take(x) as unknown as StreamEvent;
  }
}

/**
 * A call being made and watched. Iterate it for its answer's text as it
 * arrives; `events()` for every event of the call and the calls inside it;
 * await it (or `result`) for its value. Closing it cancels the call.
 */
export class Stream<A = unknown> implements AsyncIterable<string>, PromiseLike<A>, Watcher, EventSource {
  /** The writer whose process gives these events: this one, the first. */
  readonly writer = 1;
  private readonly controller = new AbortController();
  private readonly log: { event: StreamEvent; node: Node }[] = [];
  private readonly link = new Relink();
  private waiters: Array<() => void> = [];
  private finished = false;
  private outer: string | null = null;
  private answerField: string | null = null;
  private answerText = "";
  /** The call's value (the same as calling the program); rejects with its error. */
  readonly result: Promise<A>;
  /** Read for its result only: the calls it watches are not streamed from the provider. */
  readonly passive: boolean;
  /** @internal A module's answer as it is written: the AI function whose answer text is the module's (the outside view). */
  answerFrom: string | null = null;
  private readonly calls = new Set<string>();

  constructor(start: (stream: Stream<A>) => Promise<A>, signal?: AbortSignal, passive = false) {
    this.passive = passive;
    let onAbort: (() => void) | null = null;
    if (signal) {                                  // the caller's signal closes the stream too, until the call ends
      if (signal.aborted) this.controller.abort();
      else signal.addEventListener("abort", onAbort = () => this.controller.abort(), { once: true });
    }
    let started: Promise<A>;
    try {
      started = start(this);
    } catch (err) {
      started = Promise.reject(err);
    }
    this.result = started.finally(() => {
      if (onAbort) signal!.removeEventListener("abort", onAbort);   // a long-lived signal keeps no finished stream
      this.finished = true;
      this.wake();
    });
    this.result.catch(() => undefined);           // an unwatched failure is reported by whoever awaits
  }

  /** Aborted when the stream is closed. */
  get signal(): AbortSignal {
    return this.controller.signal;
  }

  /** The id of the call the stream was opened on (null until it starts). */
  get callId(): string | null {
    return this.outer;
  }

  /** The log's id (its outermost call's), once the call started. */
  get tree(): string | null {
    return this.log[0]?.event.tree ?? null;
  }

  /** The answer so far: its text since the call's latest request (streaming.md, law 3). Provisional: `result` has the typed value. */
  get text(): string {
    return this.answerText;
  }

  /** Stop the call: no new request starts, and it ends with `Cancelled`. */
  close(): void {
    this.controller.abort();
  }

  /** The tool calls this stream's call waits for a person to approve (`approve` rules, a plugin's `ask`). */
  get waiting(): Approval[] {
    return waitingHere().filter((a) => this.calls.has(a.call));
  }

  /**
   * Say yes to a tool call the call waits for (`s.waiting[0]`, its invocation
   * number, or nothing for the only one): the tool runs and the call goes on.
   */
  approve(approval?: Approval | number | null, opts: { by?: string | null } = {}): void {
    this.answer(approval ?? null, true, null, opts.by ?? null);
  }

  /** Say no: the model is told the person did not allow it (and why), and may try something else. */
  deny(approval?: Approval | number | null, reason?: string | null, opts: { by?: string | null } = {}): void {
    this.answer(approval ?? null, false, reason ?? null, opts.by ?? null);
  }

  private answer(approval: Approval | number | null, allowed: boolean, reason: string | null, by: string | null): void {
    const pending = this.waiting;
    let target: Approval | undefined;
    if (approval === null) {
      if (pending.length !== 1) throw new TypeError(`${pending.length} tool calls wait here: name one (s.waiting)`);
      target = pending[0];
    } else {
      const inv = typeof approval === "number" ? approval : approval.invocation;
      const call = typeof approval === "number" ? null : approval.call;
      target = pending.find((a) => a.invocation === inv && (call === null || a.call === call));
    }
    if (!target) throw new TypeError(`no tool call waits here for ${JSON.stringify(approval)}`);
    answerHere(target.call, target.invocation, allowed, { reason, by, plugin: target.plugin });
  }

  /** @internal */
  check(): void {
    if (this.controller.signal.aborted) throw new Cancelled();
  }

  /** @internal Given each whole event of the calls it watches (by the log). */
  receive(e: StreamEvent, node: Node): void {
    if (e.kind === "started") this.calls.add(e.call);
    if (this.outer === null && e.kind === "started") {
      this.outer = e.call;
      this.answerField = e.program.answer;
    }
    if (e.call === this.outer) {
      if (e.kind === "request" || e.kind === "retry") this.answerText = "";
      else if (e.kind === "text" && e.field === this.answerField) this.answerText += e.text;
    }
    this.log.push({ event: this.link.take(e as unknown as Rec) as unknown as StreamEvent, node });
    this.wake();
  }

  private wake(): void {
    const w = this.waiters;
    this.waiters = [];
    for (const f of w) f();
  }

  private project(opts: EventsOptions, upTo = this.log.length): StreamEvent[] {
    const former = new Former(opts, this.answerFrom);
    const out: StreamEvent[] = [];
    for (const { event, node } of this.log.slice(0, upTo)) {
      const x = former.take(event, node.keep, node.program);
      if (x) out.push(x);
    }
    return out;
  }

  /**
   * The events of a form after one this reader has (streaming.md,
   * "Resuming"): this process gives any form of its log while it runs.
   * `event-unknown` when that form has no such event.
   */
  read(tree: string | null, after: Position | null, opts: Omit<EventsOptions, "after"> = {}): ReadAnswer {
    if (tree !== null && this.tree !== null && tree !== this.tree) return after === null ? { events: [] } : { refuses: "event-unknown" };
    return resume(this.project(opts), after);
  }

  /**
   * Every event, in order, as it happens: the call's and those of the calls
   * inside it (a stream opened inside a tree shows the tree's numbers, its
   * first `after` null). `form: "kept"` gives what the log may keep; `view`
   * a view made from the form; `after` resumes after an event of that form
   * (`EventUnknown` when the form has none).
   */
  async *events(opts: EventsOptions = {}): AsyncGenerator<StreamEvent> {
    const former = new Former(opts, this.answerFrom);
    let skipping = opts.after !== undefined && opts.after !== null;
    if (skipping && "refuses" in this.read(null, opts.after!, opts)) {
      throw new EventUnknown(`this stream has no event ${opts.after!.writer}:${opts.after!.seq} in that form`);
    }
    let i = 0;
    for (;;) {
      while (i < this.log.length) {
        const { event, node } = this.log[i++]!;
        const x = former.take(event, node.keep, node.program);
        if (!x) continue;
        if (skipping) {
          const p = positionOf(x);
          if (p.writer === opts.after!.writer && p.seq === opts.after!.seq) skipping = false;
          continue;
        }
        yield x;
      }
      if (this.finished && i >= this.log.length) return;
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
  }

  /** The answer's text, piece by piece (a retry's discarded text included: `result` is the truth). */
  async *[Symbol.asyncIterator](): AsyncGenerator<string> {
    for await (const e of this.events()) {
      if (e.kind === "text" && e.call === this.outer && e.field === this.answerField) yield e.text;
    }
    await this.result;
  }

  then<T1 = A, T2 = never>(ok?: ((value: A) => T1 | PromiseLike<T1>) | null, fail?: ((reason: unknown) => T2 | PromiseLike<T2>) | null): Promise<T1 | T2> {
    return this.result.then(ok, fail);
  }
}

/** A stream of an AI function's call: its `prediction` has every output, the turn and the replies. */
export class PredictionStream<A = unknown, P = unknown> extends Stream<A> {
  /** The whole prediction, when it ends. */
  readonly prediction: Promise<P>;

  constructor(start: (stream: Stream<A>) => Promise<P>, answer: (p: P) => A, signal?: AbortSignal, passive = false) {
    let prediction!: Promise<P>;
    super((s) => {
      try {
        prediction = start(s);
      } catch (err) {
        prediction = Promise.reject(err);
      }
      // a journal that did not keep the end: the error holds what this reader would have got (the answer)
      return prediction.then(answer, (err: unknown) => {
        throw err instanceof JournalError ? err.withDone((p) => answer(p as P)) : err;
      });
    }, signal, passive);
    prediction.catch(() => undefined);
    this.prediction = prediction;
  }
}
