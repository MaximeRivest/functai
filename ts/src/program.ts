/**
 * One call of any program (an AI function or a module), from its start to
 * its record: which values its log keeps, which observers and journal its
 * tree gets, its `started` event, a required journal's barriers at the
 * tree's start and end, its terminal event, and its line in the call log.
 */

import * as calllog from "./calllog.ts";
import { ACTIVE_TURN, Call, errorJson, type Program, type TurnRunLike } from "./calllog.ts";
import { keptFields, type CallFields } from "./content.ts";
import { Cancelled } from "./errors.ts";
import { receivers, type ReceiverLayer } from "./events.ts";
import { env } from "./host.ts";
import { journalOf, JournalError, TreeLog, type Node, type Observer, type ResolvedJournal, type Watcher } from "./log.ts";
import { checkSettings, layersOf, type Settings } from "./settings.ts";
import { copyData, entriesOf, getOwn, recordOf, setOwn, toJson } from "./values.ts";

type Rec = Record<string, unknown>;

/** What a call's body gives back: the value the caller gets, its outputs by name, and what the log's `done` holds. */
export interface Ended<R> {
  readonly value: R;
  readonly outputs: Rec;
  /** The call's value as its program gives it (an AI function's answer): the `done` event's, and `JournalError`'s outcome (default: `value`). */
  readonly shown?: unknown;
  /** AI functions: what the code returned, when it is not the answer as the model gave it. */
  readonly returned?: unknown;
}

export interface CallSpec<R> {
  readonly program: () => Program;
  readonly fields: CallFields;
  /** The program's own settings (the closest layer), and this call's options (a block around it). */
  readonly own: Settings;
  readonly options: Settings;
  /** The settings it runs with (every layer merged). */
  readonly settings: Settings;
  /** A stream opened on this call. */
  readonly stream?: Watcher & { readonly signal: AbortSignal };
  readonly signal?: AbortSignal;
  /**
   * The inputs by their interface's names, as the `started` event and the
   * record hold them (only the fields' names: a value given under another
   * name is refused, never kept): `recordInputs`, taken by the caller before
   * anything (a schema's parsing, the code) can change them.
   */
  readonly inputs: RecordedInputs;
  /** The call is refused before its code runs (an input its interface does not take): this error is its outcome. */
  readonly refused?: unknown;
  readonly body: (call: Call) => Promise<Ended<R>>;
  /** The program object itself: a conversation's turn knows its own call by it. */
  readonly self?: object;
  /**
   * What the call is shown as context, worked out before it starts (its
   * `saw`): a conversation's earlier turns, a helper's memory, a row asked
   * again (conversations.ts).
   */
  readonly prepare?: (call: Call) => void | Promise<void>;
}

/**
 * A turn stopped to wait for a person (contract/tools.md): the call and every
 * call it is inside stop without ending: no `failed` event, no record (the
 * log stays unfinished, for the process that resumes it).
 */
export class TurnWaiting extends Error {
  readonly code = "turn-waiting";
  readonly approval: unknown;
  constructor(message: string, approval: unknown) {
    super(message);
    this.name = "TurnWaiting";
    this.approval = approval;
  }
}

/**
 * The turn a call made now runs in: the turn being started (its own call
 * comes first), else its parent's.
 */
export function turnRunFor(parent: Call | undefined): TurnRunLike | null {
  const active = ACTIVE_TURN.get();
  if (active && !active.rootTaken) return active;
  if (parent) return parent.turnRun as TurnRunLike | null;
  return active ?? null;
}

const combine = (...signals: (AbortSignal | undefined)[]): AbortSignal | undefined => {
  const all = signals.filter((s): s is AbortSignal => s !== undefined);
  return all.length <= 1 ? all[0] : AbortSignal.any(all);
};

/** The journal a layer sets: absent (undefined), none (null), or a store and a mode. */
function journalAt(s: Settings): ResolvedJournal | null | undefined {
  return s.journal === undefined ? undefined : journalOf(s.journal);
}

const unique = <T>(xs: readonly T[]) => [...new Set(xs)];

/** Each recorded input: its JSON (or description), its size, whether it is a description. */
export type RecordedInputs = Readonly<Record<string, readonly [unknown, number, boolean]>>;

/**
 * The inputs a call records, by its interface's names, as JSON now: what a
 * schema's parsing or the code later does to the values (even in place,
 * inside a nested object) changes nothing here. An input left out is absent.
 */
export function recordInputs(names: readonly string[], inputs: Readonly<Rec>): RecordedInputs {
  const out: Record<string, readonly [unknown, number, boolean]> = {};
  for (const n of names) {
    const value = getOwn(inputs, n);
    if (value !== undefined) setOwn(out, n, toJson(value));
  }
  return out;
}

/** How a call's body ended: with a value, or with what it threw (which may be anything, `undefined` included). */
type Outcome<R> = { readonly ok: true; readonly ended: Ended<R> } | { readonly ok: false; readonly error: unknown };

/**
 * The fields a `logContent` layer in effect drops for a call with these
 * settings: an error message never quotes their values (programs.md, "The
 * message"). When unsure, every field.
 */
export function droppedFields(fields: CallFields, own: Settings, options: Settings = {}): string[] {
  try {
    const layers = layersOf(own, options);
    const keep = keptFields(fields, layers.map((l) => l.settings.logContent).filter((v) => v !== undefined && v !== null) as never,
      env()["FUNCTAI_LOG_CONTENT"] ?? null);
    return Object.keys(keep).filter((n) => getOwn(keep, n) !== true);
  } catch {
    return ["*"];
  }
}

/** Run one call of a program: its events, its journal's barriers, its record. */
export async function runCall<R>(spec: CallSpec<R>): Promise<R> {
  checkSettings(spec.options, "a call's options");               // a block around one call: what every block may set
  const parent = calllog.current.get();
  const layers = layersOf(spec.own, spec.options);
  const program = spec.program();
  let folder: string | null = null;
  try {
    folder = calllog.folderOf(spec.settings.logCalls);
  } catch (err) {
    calllog.warnOnce(`start:${(err as Error).name}`, `calls are not logged: ${(err as Error).message}`);
  }
  const keep = keptFields(spec.fields, layers.map((l) => l.settings.logContent).filter((v) => v !== undefined && v !== null) as never,
    env()["FUNCTAI_LOG_CONTENT"] ?? null);
  const turnRun = turnRunFor(parent);
  const signal = combine(spec.signal, spec.stream?.signal, parent?.signal, turnRun?.signal);
  const call = new Call(spec.program, parent, folder, spec.fields, keep, calllog.callerOf(spec.settings), signal);
  call.turnRun = turnRun;
  if (turnRun && !turnRun.rootTaken && spec.self !== undefined && spec.self === turnRun.self) {
    turnRun.rootTaken = true;
    call.mintedId = turnRun.turn;                    // the turn's id, minted before the call (conversations.md)
  }
  if (!parent) {
    const remote = calllog.REMOTE_PARENT.get();
    if (remote?.id) {                               // the caller's call, in another process's log (serving.md)
      call.remoteParent = remote.id;
      remote.id = null;
    }
  } else call.invocation = calllog.INVOCATION.get() ?? null;
  turnRun?.started(call);
  // what the record and the started event hold: the fields' values, as JSON, taken before anything could change them
  const recorded: Record<string, readonly [unknown, number, boolean]> = {};
  for (const n of spec.fields.inputs) {
    const got = getOwn(spec.inputs, n);
    if (got !== undefined) setOwn(recorded, n, got);
  }
  call.inputsJson = recorded;

  // Receivers: observers add up (and a call's are its parent's too); the tree's journal is decided when its outermost call starts.
  const receiverLayers: ReceiverLayer<Observer, unknown>[] = layers.map((l) => ({
    where: l.where, observers: l.settings.observers ?? [],
    ...(l.settings.programObservers !== undefined && l.settings.programObservers !== null
      ? { programObservers: l.settings.programObservers } : {}),
    ...(journalAt(l.settings) !== undefined ? { journal: journalAt(l.settings) } : {}),
  }));
  const got = receivers(receiverLayers);
  let refusal: unknown = null;
  let log: TreeLog;
  const later = !parent && turnRun ? turnRun.laterWriter(call) : null;
  if (!parent) {
    log = new TreeLog(call.id, got.journal as ResolvedJournal | null, later ?? {});
    if (later) {                                    // a turn resumed: this process continues its log
      call.writer = later.writer;
      log.replaying = true;
      call.requests = later.requests;
    }
    if (got.refused) {
      refusal = new JournalError("journal-policy",
        `${program.name}: its own settings replace or remove a journal the host set, or a closer layer replaces, weakens or removes a required one`);
    }
  } else {
    log = parent.node.log;
    const set = receiverLayers.find((l) => l.journal !== undefined)?.journal as ResolvedJournal | null | undefined;
    const tree = log.journal;
    if (set && !(tree && tree.store === set.store && tree.mode === set.mode)) {
      if (set.mode === "required") {
        refusal = new JournalError("journal-scope", `${program.name}: a required journal is set only inside a call tree (it keeps whole trees): set it around the outermost call`);
      } else {
        calllog.warnOnce(`journal-scope:${program.name}`, `a journal set only inside a call tree keeps nothing of it (${program.name}); set it around the outermost call`);
      }
    }
  }
  const node: Node = {
    id: call.id, name: program.name, parent: parent?.node ?? null, log,
    keep: {
      inputs: Object.fromEntries(spec.fields.inputs.map((n) => [n, getOwn(keep, n)!])),
      outputs: Object.fromEntries(spec.fields.outputs.map((n) => [n, getOwn(keep, n)!])),
    },
    program: { kind: program.kind, answer: program.answer },
    observers: unique([...(parent?.node.observers ?? []), ...got.observers,
      ...(!parent && turnRun ? [turnRun.eventsSink(call)].filter((o) => o !== null) : [])]),
    streams: [...(parent?.node.streams ?? []), ...(spec.stream ? [spec.stream] : [])],
  };
  call.node = node;
  if (spec.prepare) {
    try {
      await spec.prepare(call);                    // what it is shown as context (its saw), before it starts
    } catch (err) {
      refusal ??= err as JournalError;              // (a conversation's refusal: the call's outcome)
    }
  }

  const started = log.emit(node, "started", {
    parent: call.parent, root: call.root, program,
    inputs: recordOf(entriesOf(recorded).map(([k, [json]]) => [k, copyData(json)])), content: true, saw: copyData(call.saw),
    ...(call.invocation !== null ? { invocation: call.invocation } : {}),
  });

  // whether the body ended well is said by `ok`, never by the error's value: code may throw undefined (Promise.reject())
  let outcome: Outcome<R>;
  try {
    if (refusal !== null) throw refusal;
    if (spec.refused !== undefined) throw spec.refused;
    if (!parent && await log.barrier(started, call.signal) === "cancelled") throw new Cancelled();
    if (call.signal?.aborted) throw new Cancelled();
    // the body: this call is current; a call it makes is not made by a tool unless a tool of its own makes it
    const ended = await calllog.current.run(call, () => calllog.INVOCATION.run(null, () => spec.body(call)));
    // closed before it ended: the call is cancelled, whatever its code did after (streaming.md, "Closing")
    if (call.signal?.aborted) throw new Cancelled();
    call.outputs = ended.outputs;
    outcome = { ok: true, ended };
  } catch (err) {
    outcome = { ok: false, error: err };
  }

  if (!outcome.ok && outcome.error instanceof TurnWaiting) {
    // a turn stopped to wait for a person: nothing ended; its log stays unfinished, for the process that resumes it
    throw outcome.error;
  }
  if (outcome.ok) call.value = outcome.ended.value;
  const kind = outcome.ok ? "done" : "failed";
  const fields = outcome.ok
    ? { value: toJson(outcome.ended.shown !== undefined ? outcome.ended.shown : outcome.ended.value)[0] }
    : { error: errorJson(outcome.error) };
  let unconfirmed: Awaited<ReturnType<TreeLog["end"]>> = null;
  if (!parent) unconfirmed = await log.end(node, kind, fields);
  else log.emit(node, kind, fields);
  if (unconfirmed) call.journal = unconfirmed.journal;
  calllog.write(call, outcome.ok
    ? { failed: false, returned: outcome.ended.returned !== undefined ? outcome.ended.returned : outcome.ended.value }
    : { failed: true, error: outcome.error });
  turnRun?.ended(call, !outcome.ok);
  if (unconfirmed) {
    throw new JournalError("journal-end",
      `the journal ${unconfirmed.journal === "refused" ? "refused" : "did not answer for"} the end of ${program.name}'s call (event ${unconfirmed.event.writer}:${unconfirmed.event.seq}); its outcome is in err.outcome`,
      {
        journal: unconfirmed.journal, event: unconfirmed.event, tree: log.tree, store: log.journal!.store,
        outcome: outcome.ok ? { done: outcome.ended.value } : { failed: outcome.error },
      });
  }
  if (!outcome.ok) throw outcome.error;
  return outcome.ended.value;
}
