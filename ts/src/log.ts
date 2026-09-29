/**
 * The writer of a call tree's log (contract/streaming.md): one log per tree,
 * numbered as its events happen, given to the streams opened in it, to
 * observers (the kept form) and to the tree's journal (the kept form, with
 * acknowledged conditional appends: best effort, or required, with barriers
 * at the tree's start, before each tool and at its end).
 *
 * Nothing here waits without a bound, and nothing a receiver does reaches
 * the call: every append has a deadline and its resends back off; a barrier
 * stops at its deadline or when the call is cancelled; observers are given
 * events later, each from its own bounded queue, each its own copy.
 */

import { iso } from "./calllog.ts";
import { copyData } from "./values.ts";
import {
  keptEvent, positionOf, Relink, settle, type EventStore, type KeptFields, type Position, type StreamEvent,
} from "./events.ts";

type Rec = Record<string, unknown>;

/**
 * Gets the kept form of every event of every call in its scope, in order.
 * A function is called soon after each event is made, never on the call's
 * own turn (from a queue drained between the call's steps; an observer that
 * returns a promise is not awaited); an object with `postMessage` (a
 * `Worker`, a `MessagePort`, a `BroadcastChannel`) is posted each event, so
 * it handles them on another thread. Each observer gets its own copy, from
 * its own queue, with its share of the time given to observers: one that is
 * slow falls behind alone. One that throws or rejects is warned about once
 * and given no more events; one that falls more than 10,000 events behind
 * loses events (it then sees a gap in `after`).
 */
export type Observer = ((event: StreamEvent) => void | PromiseLike<void>) | { postMessage(event: StreamEvent): void };

/**
 * A journal: a store that keeps whole trees' kept logs while they are
 * written. A store alone is best effort; `{ store, mode: "required" }` makes
 * calls wait until their events are kept (at the tree's start, before each
 * tool, at its end), at most `timeout` milliseconds at each.
 */
export type Journal = EventStore | JournalSetting;

export interface JournalSetting {
  readonly store: EventStore;
  /** `"best-effort"` (the default): the call never waits. `"required"`: it waits at three barriers, and raises `JournalError` when they fail. */
  readonly mode?: "required" | "best-effort";
  /** Sends of one append, after the first, before giving up on it for now (default 2). */
  readonly retries?: number;
  /** Most events in one append (default 64; 1 sends each event alone). */
  readonly batch?: number;
  /**
   * Milliseconds (default 30,000): the longest an append is waited for (its
   * `signal` aborts then, and no answer came), and the longest a barrier
   * waits before the call goes on as if the journal did not answer.
   * `Infinity` waits for ever.
   */
  readonly timeout?: number;
  /** Milliseconds before the first resend of an append; each next one waits twice as long, up to 2 s (default 50). */
  readonly backoff?: number;
}

/** A journal setting as the rules read it: its store and mode (two are the same when both are), and how it sends. */
export interface ResolvedJournal {
  readonly store: EventStore;
  readonly mode: "required" | "best-effort";
  readonly retries: number;
  readonly batch: number;
  readonly timeout: number;
  readonly backoff: number;
}

const isStore = (x: unknown): x is EventStore => typeof x === "object" && x !== null
  && typeof (x as EventStore).append === "function" && typeof (x as EventStore).read === "function";

const count = (v: unknown, least: number) => typeof v === "number" && Number.isInteger(v) && v >= least;
const millis = (v: unknown, least: number) => typeof v === "number" && !Number.isNaN(v) && v >= least;

/**
 * A `journal` setting as its store, mode and sending (null: no journal).
 * A setting that could only fail later refuses here (`TypeError`).
 */
export function journalOf(j: Journal | null, where = "journal"): ResolvedJournal | null {
  if (j === null) return null;
  const defaults = { mode: "best-effort" as const, retries: 2, batch: 64, timeout: 30_000, backoff: 50 };
  if (isStore(j)) return { store: j, ...defaults };
  if (typeof j !== "object" || !isStore((j as JournalSetting).store)) {
    throw new TypeError(`${where}: a store ({ append, read }), or { store, mode: "required" | "best-effort" }; null for none`);
  }
  const s = j as JournalSetting;
  const known = new Set(["store", "mode", "retries", "batch", "timeout", "backoff"]);
  const extra = Object.keys(s).filter((k) => !known.has(k));
  if (extra.length) throw new TypeError(`${where}: a journal setting has no ${extra.join(", ")} (it has store, mode, retries, batch, timeout, backoff)`);
  const mode = s.mode ?? "best-effort";
  if (mode !== "required" && mode !== "best-effort") throw new TypeError(`${where}: mode is "required" or "best-effort", not ${JSON.stringify(mode)}`);
  if (s.retries !== undefined && !count(s.retries, 0)) throw new TypeError(`${where}: retries is a whole number of at least 0`);
  if (s.batch !== undefined && !count(s.batch, 1)) throw new TypeError(`${where}: batch is a whole number of at least 1`);
  if (s.timeout !== undefined && !(millis(s.timeout, 1) && (s.timeout <= MAX_DELAY || s.timeout === Infinity))) {
    throw new TypeError(`${where}: timeout is milliseconds, from 1 to ${MAX_DELAY} (about 24.8 days), or Infinity to wait for ever`);
  }
  if (s.backoff !== undefined && !(millis(s.backoff, 0) && s.backoff <= MAX_DELAY)) throw new TypeError(`${where}: backoff is milliseconds, from 0 to ${MAX_DELAY}`);
  return {
    store: s.store, mode, retries: s.retries ?? defaults.retries, batch: s.batch ?? defaults.batch,
    timeout: s.timeout ?? defaults.timeout, backoff: s.backoff ?? defaults.backoff,
  };
}

/** Whether something can be an observer: a function, or an object with `postMessage`. */
export function isObserver(o: unknown): o is Observer {
  return typeof o === "function" || (typeof o === "object" && o !== null && typeof (o as { postMessage?: unknown }).postMessage === "function");
}

/** An `observers` setting, checked (`TypeError` for one that could only fail later). */
export function checkObservers(value: unknown, where = "observers"): void {
  if (value === undefined || value === null) return;
  if (!Array.isArray(value)) throw new TypeError(`${where}: observers is a list: observers: [(e) => socket.send(JSON.stringify(e))]`);
  for (const o of value) {
    if (!isObserver(o)) throw new TypeError(`${where}: an observer is a function of one event, or an object with postMessage (a Worker, a MessagePort); not ${o === null ? "null" : typeof o}`);
  }
}

export type JournalCode = "journal-policy" | "journal-scope" | "journal-barrier" | "journal-end";

/**
 * A call's outcome as `JournalError` holds it: what the call would have
 * given its caller (its value: the answer for `fn(x)`, the `Prediction` for
 * `fn.predict(x)`), or its error.
 */
export type Outcome = { readonly done: unknown } | { readonly failed: unknown };

/**
 * A journal decided a call (streaming.md, "Keeping a log while it is
 * written"). `journal-policy`: the layers around the tree break the journal
 * policy (refused before it runs). `journal-scope`: a required journal set
 * only inside a tree. `journal-barrier`: a required journal did not confirm
 * the events before the code or a tool ran. `journal-end`: it did not confirm
 * the call's end: `outcome` is what the call did (what it would have given
 * you, or its error), `event` names the event that records it, and
 * `journal` says whether the journal refused it or did not answer
 * (`settle()` finds out which).
 */
export class JournalError extends Error {
  readonly code: JournalCode;
  readonly journal?: "refused" | "unknown";
  readonly event?: Position;
  readonly outcome?: Outcome;
  readonly tree?: string;
  private readonly store?: EventStore;

  constructor(code: JournalCode, message: string, opts: {
    journal?: "refused" | "unknown"; event?: Position; outcome?: Outcome; tree?: string; store?: EventStore;
  } = {}) {
    super(message);
    this.name = "JournalError";
    this.code = code;
    if (opts.journal) this.journal = opts.journal;
    if (opts.event) this.event = opts.event;
    if (opts.outcome) this.outcome = opts.outcome;
    if (opts.tree) this.tree = opts.tree;
    Object.defineProperty(this, "store", { value: opts.store, enumerable: false });
  }

  /** @internal The same error, its value as another caller gets it (the answer of a `Prediction`). */
  withDone(f: (value: unknown) => unknown): JournalError {
    if (!this.outcome || !("done" in this.outcome)) return this;
    const copy = new JournalError(this.code, this.message, {
      ...(this.journal ? { journal: this.journal } : {}), ...(this.event ? { event: this.event } : {}),
      ...(this.tree ? { tree: this.tree } : {}), store: this.store, outcome: { done: f(this.outcome.done) },
    });
    copy.stack = this.stack;
    return copy;
  }

  /**
   * For `journal-end`: read the journal and say what became of the call's
   * end: `"kept"`, `"not-kept"` (the log is unfinished; final only once you
   * have claimed it) or `"another-end"` (another writer ended the log). The
   * store did not answer the end, and may not answer this read: `signal`
   * (`AbortSignal.timeout(5000)`) stops waiting, rejecting with its reason.
   */
  async settle(opts: { signal?: AbortSignal } = {}): Promise<"kept" | "not-kept" | "another-end"> {
    if (!this.store || !this.tree || !this.event) throw new Error("only a journal-end error can be settled");
    return settle(this.store, this.tree, this.event, opts);
  }
}

// ------------------------------------------------------------------ timers

type Timer = ReturnType<typeof setTimeout>;

/** A timer; `hold: false` lets the process end while it waits (a best-effort journal's resends). Longer than `MAX_DELAY`: none (for ever). */
function timer(ms: number, fn: () => void, hold: boolean): Timer | null {
  if (!Number.isFinite(ms) || ms > MAX_DELAY) return null;
  const t = setTimeout(fn, ms);
  if (!hold) (t as { unref?: () => void }).unref?.();
  return t;
}

/** Wait `ms` (for ever when longer than a timer can), or until `signal` aborts. */
const pause = (ms: number, hold: boolean, signal: AbortSignal) => new Promise<void>((resolve) => {
  if (ms <= 0 || signal.aborted) {
    resolve();
    return;
  }
  const done = () => {
    if (t !== null) clearTimeout(t);
    signal.removeEventListener("abort", done);
    resolve();
  };
  const t = timer(ms, done, hold);
  signal.addEventListener("abort", done, { once: true });
});

/** Later, after the current turn and the promise jobs it queued: a macrotask. */
const later: (fn: () => void) => void = typeof globalThis.setImmediate === "function"
  ? (fn) => { globalThis.setImmediate(fn); }
  : (fn) => { setTimeout(fn, 0); };

// ------------------------------------------------------------------ journals

type Status = "confirmed" | "refused" | "unanswered";

interface Waiter {
  readonly n: number;
  readonly done: (s: Status) => void;
}

/** The writers sending now (for `flush`). */
const busy = new Set<JournalWriter>();
/** The writers holding events not confirmed after a round gave up (for `flush`: it sends them again). */
const holding = new Set<JournalWriter>();

/** The longest a timer can wait (setTimeout's limit, about 24.8 days); a longer wait is for ever. */
export const MAX_DELAY = 2_147_483_647;
/** The longest wait between a writer's later rounds, in milliseconds (they start 1 s apart, and double). */
const LATER_MAX = 60_000;
/**
 * The most events all journal writers of the process may hold unconfirmed
 * together. Past it, the writer holding the oldest gives up on its log for
 * good, then the next, until they hold no more: what writers hold for a
 * store that stays down is bounded by this, not by time. A writer that
 * gives up lets go of everything it holds at once: its round ends, and the
 * append under way is aborted (the copies that append gave the store are the
 * store's: one that ignores the abort and never answers keeps them).
 */
let heldLimit = 100_000;
/** Events all writers hold unconfirmed now, and the writers holding some, the one holding the oldest first. */
let held = 0;
const holders = new Set<JournalWriter>();
/** Events writers gave up on for good, and how many of them a `flush()` has reported (it says false once for those since). */
let lostEvents = 0;
let lostReported = 0;

/** @internal For tests: set how many unconfirmed events all writers may hold together; returns the previous limit. */
export function holdAtMost(n: number): number {
  const before = heldLimit;
  heldLimit = n;
  return before;
}

type Trouble = { fail(what: string): void; lost(what: string): void; ok(): void };

/**
 * Sends a tree's kept events to its journal, in order, each append naming
 * where it goes (`after`). It keeps every event not confirmed and sends it
 * again (after `backoff`, doubling); after a refusal it sends nothing more.
 * Each append is given fresh copies, and waited for at most `timeout`: a
 * store that throws, never answers, or answers something else than kept or
 * duplicate never holds the writer, nor reaches the call. When a round gives
 * up, it tries again on its own later (after 1 s, then 2, 4, … up to 60 s
 * apart, for as long as the process lives: a tree's last events have no
 * next event to carry them), at the next event, and when `flush()` asks;
 * those later rounds never keep the process alive. Time alone never makes
 * it give up: only memory does. When all writers together hold more than
 * the limit (100,000 events), the one holding the oldest gives up on its log
 * for good (warned about; the log is kept at least up to the events it
 * confirmed; the next `flush()` says false) and sends nothing more to it.
 */
export class JournalWriter {
  private pending: { e: Rec; n: number }[] = [];
  private added = 0;
  private confirmedN = 0;
  private running = false;
  /** A later round, scheduled after one gave up; and how many came since the writer last had an answer. */
  private retryTimer: Timer | null = null;
  private laterRounds = 0;
  private readonly waiters = new Set<Waiter>();
  private idlers: Array<() => void> = [];
  refused = false;
  readonly journal: ResolvedJournal;
  private readonly trouble: Trouble;
  /** Set once it gave up on what it held for good: it sends nothing more to this log. */
  private abandoned = false;
  /** Stops the round under way (its wait for an append, its pause before the next): aborted when the writer gives up. */
  private halt: AbortController | null = null;
  /** Appends made (a batch is one), for tests and hosts measuring a store. */
  appends = 0;

  constructor(journal: ResolvedJournal, trouble: Trouble = { fail: () => undefined, lost: () => undefined, ok: () => undefined }) {
    this.journal = journal;
    this.trouble = trouble;
  }

  /** Queue an event (in the kept form; a copy is taken); returns its number in this writer, for `confirm`. */
  add(e: Rec): number {
    const n = ++this.added;
    if (this.refused || this.abandoned) return n;
    if (!this.pending.length) holders.add(this);
    this.pending.push({ e: copyData(e), n });
    held++;
    this.laterRounds = 0;                            // a new event: the writer tries again, and later again, from the start
    // past the limit, the writers holding the oldest events give up on their logs (this one, if it holds the oldest)
    for (const w of holders) {
      if (held <= heldLimit) break;
      w.abandon();
    }
    this.start();
    return n;
  }

  /** Let go of the first `k` events it holds (confirmed), or of all of them. */
  private release(k = this.pending.length): void {
    const gone = Math.min(k, this.pending.length);
    this.pending.splice(0, gone);
    held -= gone;
    if (!this.pending.length) holders.delete(this);
  }

  /** Send what waits now (an event came, or `flush` asks), unless it is sending already: whether a round began. */
  start(): boolean {
    if (this.running || this.refused || this.abandoned || !this.pending.length) return false;
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.running = true;                             // set before any store code runs: a store that throws at once cannot leave it stale
    busy.add(this);
    queueMicrotask(() => { void this.pump(); });
    return true;
  }

  /** Send what waits, in order, batch by batch, until it is all confirmed, one is refused, or a round gives up. */
  private async pump(): Promise<void> {
    let gaveUp = false;
    const halt = this.halt = new AbortController();
    try {
      while (this.pending.length && !this.refused && !this.abandoned) {
        const sending = this.pending.slice(0, this.journal.batch);
        const answer = await this.round(sending, halt.signal);
        if (answer === "abandoned" || this.abandoned) break;          // it gave up on the log meanwhile: nothing more is sent
        if (answer === "unanswered") {
          gaveUp = true;
          this.trouble.fail("the journal did not answer (the log is kept at least up to the events it confirmed; the rest is sent again later)");
          break;
        }
        if (answer === "refused") break;
        this.release(sending.length);
        this.confirmedN = sending[sending.length - 1]!.n;
        this.laterRounds = 0;
        this.trouble.ok();
        for (const w of this.waiters) if (w.n <= this.confirmedN) w.done("confirmed");
      }
    } catch {
      gaveUp = true;                                 // the writer's own fault: nothing is known of what was kept
    } finally {
      this.halt = null;
      this.running = false;
      busy.delete(this);
      if (this.pending.length && !this.refused && !this.abandoned) {
        holding.add(this);
        this.later();
      } else holding.delete(this);
      const status: Status = this.refused ? "refused" : gaveUp || this.abandoned ? "unanswered" : "confirmed";
      for (const w of this.waiters) w.done(w.n <= this.confirmedN ? "confirmed" : status);
      const idlers = this.idlers;
      this.idlers = [];
      for (const f of idlers) f();
      settled();
    }
  }

  /** After a round gave up: another, later, on its own (never holding the process), 1 s after, then twice as long each time, at most 60 s. */
  private later(): void {
    if (this.retryTimer !== null) return;
    const wait = Math.min(1000 * 2 ** Math.min(this.laterRounds++, 16), LATER_MAX);
    this.retryTimer = timer(wait, () => {
      this.retryTimer = null;
      this.start();
    }, false);
  }

  /**
   * Give up on what it holds for good (all writers hold more than the limit,
   * and it holds the oldest): it sends nothing more to this log. The round
   * under way ends now: its wait for an append stops, the append's signal
   * aborts, and no pause or resend follows.
   */
  private abandon(): void {
    const n = this.pending.length;
    this.release();
    this.abandoned = true;
    this.halt?.abort();
    holding.delete(this);
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    for (const w of this.waiters) if (w.n > this.confirmedN) w.done("unanswered");
    lostEvents += n;
    this.trouble.lost(`${n} event${n === 1 ? "" : "s"} of a log the journal has not confirmed are given up: journal writers held more than ${heldLimit.toLocaleString("en-US")} unconfirmed events, and this log's were the oldest (the log is kept at least up to the events it confirmed; an append under way is aborted, and nothing more is sent to it)`);
  }

  /** Whether it holds events the journal has not confirmed, and may still send them (not refused, not given up). */
  get unconfirmed(): boolean {
    return this.pending.length > 0 && !this.refused && !this.abandoned;
  }

  /** One append of `sending`, sent again after no answer (backing off): its answer; `"abandoned"` when the writer gave up on the log meanwhile (`halt`). */
  private async round(sending: readonly { e: Rec; n: number }[], halt: AbortSignal): Promise<Status | "abandoned"> {
    const { retries, backoff } = this.journal;
    let wait = backoff;
    for (let attempt = 0; attempt <= retries; attempt++) {
      if (attempt) {
        await pause(wait, this.journal.mode === "required", halt);
        wait = Math.min(wait * 2, 2000);
      }
      if (halt.aborted) return "abandoned";
      const answer = await this.attempt(sending.map((p) => copyData(p.e)), halt);
      if (answer === GAVE_UP) return "abandoned";
      if (answer === NO_ANSWER) continue;
      if (answer === "kept" || answer === "duplicate") return "confirmed";
      this.refuse(answer, sending[0]!.e);
      return "refused";
    }
    return "unanswered";
  }

  /**
   * One append, waited for at most `timeout`: its answer, NO_ANSWER (it
   * threw, rejected, or did not answer in time), or GAVE_UP (`halt` aborted:
   * the writer gave up on the log; the append's signal aborts).
   */
  private attempt(events: Rec[], halt: AbortSignal): Promise<unknown> {
    const { store, timeout, mode } = this.journal;
    this.appends++;
    return new Promise((resolve) => {
      let over = false;
      const controller = new AbortController();
      const finish = (v: unknown) => {
        if (over) return;
        over = true;
        if (t !== null) clearTimeout(t);
        halt.removeEventListener("abort", stop);
        resolve(v);
      };
      const t = timer(timeout, () => {
        finish(NO_ANSWER);
        controller.abort(new Error(`the journal did not answer within ${timeout} ms`));
      }, mode === "required");
      const stop = () => {
        finish(GAVE_UP);
        controller.abort(new Error("the journal writer gave up on this log: journal writers held too many unconfirmed events"));
      };
      halt.addEventListener("abort", stop, { once: true });
      let answered: Promise<unknown>;
      try {
        answered = Promise.resolve(store.append(events, { signal: controller.signal }));
      } catch {
        answered = Promise.resolve(NO_ANSWER);
      }
      answered.then(finish, () => finish(NO_ANSWER));
    });
  }

  private refuse(answer: unknown, first: Rec): void {
    this.refused = true;
    this.release();
    const a = answer as { refuses?: unknown; event?: { writer?: unknown; seq?: unknown } } | null;
    const code = typeof a === "object" && a !== null && typeof a.refuses === "string" ? a.refuses : null;
    const at = typeof a === "object" && a !== null && typeof a.event === "object" && a.event !== null
      && typeof a.event.writer === "number" && typeof a.event.seq === "number"
      ? `${a.event.writer}:${a.event.seq}` : `${first["writer"]}:${first["seq"]}`;
    this.trouble.fail(code
      ? `the journal refused event ${at} (${code}); nothing more is sent to it for this log`
      : `the journal answered ${describe(answer)} to event ${at}, which is neither kept, duplicate nor a refusal; nothing more is sent to it for this log`);
  }

  /**
   * Wait until every event up to `n` is confirmed: `"confirmed"`, `"refused"`,
   * `"unanswered"` (the sends that carried it, or an event before it, gave
   * up, or `timeout` passed), or `"cancelled"` (the signal aborted).
   */
  confirm(n: number, signal?: AbortSignal): Promise<Status | "cancelled"> {
    if (this.confirmedN >= n) return Promise.resolve("confirmed");
    if (this.refused) return Promise.resolve("refused");
    if (!this.running || this.abandoned) return Promise.resolve("unanswered");          // the last round gave up on it, or the writer on the log
    if (signal?.aborted) return Promise.resolve("cancelled");
    return new Promise((resolve) => {
      let t: Timer | null = null;
      const onAbort = () => done("cancelled");
      const done = (s: Status | "cancelled") => {
        if (!this.waiters.delete(w)) return;
        if (t !== null) clearTimeout(t);
        signal?.removeEventListener("abort", onAbort);
        resolve(s);
      };
      const w: Waiter = { n, done };
      this.waiters.add(w);
      t = timer(this.journal.timeout, () => done("unanswered"), true);
      signal?.addEventListener("abort", onAbort, { once: true });
    });
  }

  /** Resolves when nothing is being sent. */
  idle(): Promise<void> {
    return this.running ? new Promise((resolve) => this.idlers.push(resolve)) : Promise.resolve();
  }

  /** Whether it is sending now. */
  get busy(): boolean {
    return this.running;
  }
}

const NO_ANSWER = Symbol("no answer");
const GAVE_UP = Symbol("gave up");

function describe(v: unknown): string {
  try {
    const s = JSON.stringify(v);
    return s === undefined ? String(v) : s.length > 80 ? s.slice(0, 77) + "..." : s;
  } catch {
    return Object.prototype.toString.call(v);
  }
}

/**
 * A journal's trouble, warned about once per outage of its store (again after
 * it has answered well); and, apart, once per outage, that a writer gave up
 * on events for good.
 */
const troubled = new WeakSet<EventStore>();
const lostWarned = new WeakSet<EventStore>();
function journalTrouble(store: EventStore, tree: string): Trouble {
  return {
    fail(what) {
      if (troubled.has(store)) return;
      troubled.add(store);
      console.warn(`functai: ${what} (log ${tree}); calls go on. Further trouble with this journal is not reported until it answers again.`);
    },
    lost(what) {
      if (lostWarned.has(store)) return;
      lostWarned.add(store);
      console.warn(`functai: ${what} (log ${tree}). Further events given up on with this journal are not reported until it answers again.`);
    },
    ok() {
      troubled.delete(store);
      lostWarned.delete(store);
    },
  };
}

// ------------------------------------------------------------------ observers

/**
 * Each observer's events waiting for it, in the order they were made: one
 * lane per observer, so one that is slow falls behind (and past the bound
 * loses events) alone, never the others beside it.
 */
interface Lane {
  readonly o: Observer;
  items: (StreamEvent | undefined)[];
  head: number;
}
const lanes = new Map<Observer, Lane>();
let draining = false;
/** Where the next turn starts, so no lane always goes first. */
let turn = 0;
const broken = new WeakSet<object>();
const warnedObservers = new WeakSet<object>();
/** The most events one observer may have waiting; past it, its events are dropped (it sees a gap). */
export const OBSERVER_QUEUE = 10_000;
/** How long one turn of giving events to observers may run, in milliseconds, before the process gets its turn back. */
const SLICE = 8;

const clock = () => (globalThis.performance ? globalThis.performance.now() : Date.now());
const observerName = (o: Observer) => (typeof o === "function" && o.name ? `observer ${o.name}` : "an observer");
const waiting = (l: Lane) => l.items.length - l.head;

function warnObserver(o: Observer, message: string): void {
  if (warnedObservers.has(o)) return;
  warnedObservers.add(o);
  console.warn(`functai: ${observerName(o)} ${message}`);
}

function fail(o: Observer, err: unknown): void {
  broken.add(o);
  lanes.delete(o);
  warnObserver(o, `failed (${(err as Error)?.message ?? String(err)}); it gets no more events`);
}

/** Queue an event (its own copy) for an observer. */
function deliver(o: Observer, e: StreamEvent): void {
  if (broken.has(o)) return;
  let lane = lanes.get(o);
  if (!lane) lanes.set(o, lane = { o, items: [], head: 0 });
  if (waiting(lane) >= OBSERVER_QUEUE) {
    warnObserver(o, `is ${OBSERVER_QUEUE} events behind: events are dropped (it sees a gap in after)`);
    return;
  }
  lane.items.push(e);
  if (!draining) {
    draining = true;
    later(drain);
  }
}

/**
 * Give observers what waits for them, for one slice of time, then let the
 * process go on. Each observer with events waiting gets its share of the
 * slice (what one leaves unused goes to the next), and at least one event
 * each turn: a fast observer empties its lane every turn, whatever a slow
 * one beside it does.
 */
function drain(): void {
  const start = clock();
  const all = [...lanes.values()];
  const n = all.length;
  const first = n ? turn++ % n : 0;
  for (let i = 0; i < n; i++) {
    const lane = all[(first + i) % n]!;
    const until = clock() + Math.max(0, start + SLICE - clock()) / (n - i);
    do {
      const e = lane.items[lane.head]!;
      lane.items[lane.head++] = undefined;
      give(lane.o, e);
      if (broken.has(lane.o)) break;
    } while (waiting(lane) > 0 && clock() < until);
    if (broken.has(lane.o)) continue;
    if (!waiting(lane)) lanes.delete(lane.o);
    else if (lane.head > 1024 && lane.head * 2 > lane.items.length) {
      lane.items = lane.items.slice(lane.head);             // a long backlog: let go of what was given
      lane.head = 0;
    }
  }
  if (lanes.size) {
    later(drain);
    return;
  }
  turn = 0;
  draining = false;
  settled();
}

function give(o: Observer, e: StreamEvent): void {
  try {
    if (typeof o === "function") {
      const r = o(e);
      if (r && typeof (r as PromiseLike<void>).then === "function") (r as PromiseLike<void>).then(undefined, (err: unknown) => fail(o, err));
    } else {
      o.postMessage(e);
    }
  } catch (err) {
    fail(o, err);
  }
}

// ------------------------------------------------------------------ flush

let waitingForIdle: Array<() => void> = [];
const idleNow = () => !draining && busy.size === 0;

function settled(): void {
  if (!idleNow()) return;
  const w = waitingForIdle;
  waitingForIdle = [];
  for (const f of w) f();
}

/** Whether every journal writer has had every event it was given confirmed (or was refused: it sends nothing more). */
const allKept = () => ![...holding].some((w) => w.unconfirmed);

/**
 * Wait until every observer has been given the events made so far, and
 * every journal has tried to keep them: a writer still holding events a
 * journal did not confirm sends them again now (at most `timeout`
 * milliseconds in all; default 5,000). True when all of it was done in
 * time and every journal confirmed every event it was sent (a journal that
 * refused one, which it is sent nothing more, counts as done); false when
 * time ran out, a journal still holds events it did not confirm, or a
 * writer gave up on events for good since the previous `flush()` returned
 * (or since the process began): each loss makes one `flush()` say false.
 * For a process that is about to end, and for tests.
 */
export async function flush(opts: { timeout?: number } = {}): Promise<boolean> {
  const timeout = opts.timeout ?? 5000;
  const deadline = clock() + timeout;
  const asked = new Set<JournalWriter>();
  /** Whether no writer gave up on events since the last flush said so; this flush says it now. */
  const noneLost = () => {
    const none = lostEvents === lostReported;
    lostReported = lostEvents;
    return none;
  };
  for (;;) {
    // each writer holding what was not confirmed sends it once more: those holding now, and those whose round gives up
    // meanwhile; one sending already (a round begun before this flush) is asked once that round is over
    for (const w of [...holding]) {
      if (asked.has(w) || !w.unconfirmed) continue;
      if (w.start()) asked.add(w);
    }
    if (!await idleWithin(deadline - clock())) {
      noneLost();
      return false;
    }
    if (![...holding].some((w) => w.unconfirmed && !asked.has(w))) return noneLost() && allKept();
  }
}

/** Resolves true once observers and journals are idle, or false after `ms`. */
function idleWithin(ms: number): Promise<boolean> {
  if (idleNow()) return Promise.resolve(true);
  if (ms <= 0) return Promise.resolve(false);
  return new Promise<boolean>((resolve) => {
    let t: Timer | null = null;
    const done = () => {
      if (t !== null) clearTimeout(t);
      resolve(true);
    };
    waitingForIdle.push(done);
    t = timer(ms, () => {
      waitingForIdle = waitingForIdle.filter((f) => f !== done);
      resolve(idleNow());
    }, true);
  });
}

// ------------------------------------------------------------------ a tree's log

/** One call in its tree's log: what its events need. */
export interface Node {
  readonly id: string;
  readonly name: string;
  readonly parent: Node | null;
  readonly log: TreeLog;
  /** Which of its fields the kept form keeps. */
  readonly keep: KeptFields;
  readonly program: { kind: string; answer: string };
  /** Its observers: those of every layer around it, and its parent's. */
  readonly observers: readonly Observer[];
  /** The streams watching it: those opened on it, and its parent's. */
  readonly streams: readonly Watcher[];
}

/** What a stream is to the log: it receives the whole events of the calls it watches. */
export interface Watcher {
  receive(e: StreamEvent, node: Node): void;
}

/** An event as made: the event, its kept form (null when the kept form leaves it out), and its number in the journal writer (0 when none). */
export interface Made {
  readonly event: StreamEvent;
  readonly kept: Rec | null;
  readonly n: number;
}

/**
 * One call tree's log in this process: it numbers every event made (writer
 * 1: this process began the tree), and gives each to the streams and
 * observers of its call and to the journal.
 */
export class TreeLog {
  readonly writer = 1;
  private seq = 0;
  private last: Position | null = null;
  private lastAt = "";
  private readonly observed = new Map<Observer, Relink>();
  private readonly toJournal = new Relink();
  readonly journalWriter: JournalWriter | null;
  readonly tree: string;
  readonly journal: ResolvedJournal | null;

  constructor(tree: string, journal: ResolvedJournal | null) {
    this.tree = tree;
    this.journal = journal;
    this.journalWriter = journal ? new JournalWriter(journal, journalTrouble(journal.store, tree)) : null;
  }

  /** Whether anything receives this call's events (then streams read replies in pieces). */
  watched(node: Node): boolean {
    return node.streams.length > 0 || node.observers.length > 0 || this.journalWriter !== null;
  }

  private stamp(): string {
    const now = iso(globalThis.performance ? globalThis.performance.timeOrigin + globalThis.performance.now() : Date.now());
    this.lastAt = now > this.lastAt ? now : this.lastAt;
    return this.lastAt;
  }

  /** Make and number an event of a call; its kept form goes to the journal (a copy). */
  make(node: Node, kind: StreamEvent["kind"], fields: Rec): Made {
    const e = {
      functai_event: 2, kind, tree: this.tree, writer: this.writer, seq: ++this.seq, after: this.last, at: this.stamp(),
      call: node.id, function: node.name, ...fields,
    } as unknown as StreamEvent;
    this.last = positionOf(e);
    const kept = node.observers.length || this.journalWriter ? keptEvent(e, node.keep, node.program) : null;
    let n = 0;
    if (this.journalWriter && kept) n = this.journalWriter.add(this.toJournal.take(kept));
    return { event: e, kept, n };
  }

  /** Give an event to the streams of its call, and (each its own copy of the kept form, later) to its observers. */
  show(made: Made, node: Node): void {
    for (const s of node.streams) s.receive(made.event, node);
    if (!made.kept) return;
    for (const o of node.observers) {
      if (broken.has(o)) continue;
      let link = this.observed.get(o);
      if (!link) this.observed.set(o, link = new Relink());
      deliver(o, link.take(copyData(made.kept)) as unknown as StreamEvent);
    }
  }

  /** Make an event and show it. */
  emit(node: Node, kind: StreamEvent["kind"], fields: Rec): Made | null {
    if (!this.watched(node)) return null;
    const made = this.make(node, kind, fields);
    this.show(made, node);
    return made;
  }

  /** Whether a required journal must confirm the tree's events before the call goes on. */
  get required(): boolean {
    return this.journal?.mode === "required";
  }

  /**
   * A required journal's barrier after event `made` (the tree's start, a
   * tool call): wait until every event up to it is confirmed (at most the
   * journal's `timeout`), then go on; else `"cancelled"` (the signal
   * aborted: the caller stops the call), or `JournalError`
   * (`journal-barrier`): the code or the tool does not run.
   */
  async barrier(made: Made | null, signal?: AbortSignal): Promise<"passed" | "cancelled"> {
    if (!this.required || !this.journalWriter || made === null) return "passed";
    const status = await this.journalWriter.confirm(made.n, signal);
    if (status === "cancelled") return "cancelled";
    if (status !== "confirmed") {
      // the message is the contract's (cases/events/journal-*: the failed event holds it)
      throw new JournalError("journal-barrier", `the journal did not keep event ${made.event.seq}`, { tree: this.tree, event: positionOf(made.event) });
    }
    return "passed";
  }

  /**
   * The tree's end: make the outermost call's `done` or `failed`; with a
   * required journal, show it only once confirmed (at most the journal's
   * `timeout`: the outcome is decided, and a cancelled caller still waits for
   * it to be kept, or for the deadline). Returns what the journal said when
   * it did not confirm it.
   */
  async end(node: Node, kind: "done" | "failed", fields: Rec): Promise<{ journal: "refused" | "unknown"; event: Position } | null> {
    if (!this.watched(node)) return null;
    const made = this.make(node, kind, fields);
    if (!this.required || !this.journalWriter) {
      this.show(made, node);
      return null;
    }
    const status = await this.journalWriter.confirm(made.n);
    if (status === "confirmed") {
      this.show(made, node);
      return null;
    }
    return { journal: status === "refused" ? "refused" : "unknown", event: positionOf(made.event) };
  }
}
