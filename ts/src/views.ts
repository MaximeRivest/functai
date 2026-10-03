/**
 * Views: what one kind of reader may see of a call tree's log
 * (contract/streaming.md, *Views*).
 *
 * - `full`: every event and value (only the process running the tree has it);
 * - `kept`: what `logContent` lets be kept (a store's form);
 * - `outside`: a caller who sees only the program's boundary (a served
 *   program's customer): its `started` (its program object without its file
 *   and line), its answer's text as it is written, the approvals addressed to
 *   it, and its `done`, or its `failed` with the error's type and code and no
 *   message. Never a helper's answer, a tool call or result, a thinking, or
 *   why a request was retried.
 *
 * A view keeps each event's `writer` and `seq` and sets `after` to the event
 * before it in the view. It never alters a value it shows and never invents
 * one: a module's answer shown as it is written is its `answerFrom` helper's
 * text, re-addressed to the module (`call`, `function`, `field`), each call of
 * that helper beginning as a `request` of the module, so a second call
 * empties the first's text.
 */

import { copyData } from "./values.ts";

type Rec = Record<string, unknown>;

/** The views every host can name. */
export type ViewName = "full" | "kept" | "outside";

const PROGRAM = new Set(["name", "kind", "module", "version", "signature", "interface", "answer"]);

/** A view made event by event: `apply(event)` gives the event as the view shows it (a new object), or null when it leaves it out. */
export class View {
  readonly name: ViewName;
  readonly answerFrom: string | null;
  private root: string | null = null;
  private rootFunction: string | null = null;
  private rootKind = "ai";
  private answer = "result";
  private readonly forwarded = new Map<string, boolean>();
  private readonly callerAsked = new Set<string>();
  private requests = 0;
  private last: { writer: number; seq: number } | null = null;

  constructor(name: ViewName, opts: { answerFrom?: string | { name: string } | null } = {}) {
    if (name !== "full" && name !== "kept" && name !== "outside") throw new TypeError(`a view is "full", "kept" or "outside"; not ${JSON.stringify(name)}`);
    this.name = name;
    const a = opts.answerFrom ?? null;
    this.answerFrom = a === null ? null : typeof a === "string" ? a : a.name;
  }

  private link(e: Rec): Rec {
    e["after"] = this.last;
    this.last = { writer: e["writer"] as number, seq: e["seq"] as number };
    return e;
  }

  apply(event: Readonly<Rec>): Rec | null {
    const e = copyData(event) as Rec;
    if (this.name !== "outside") return this.link(e);
    const kind = e["kind"] as string;
    const call = e["call"] as string;
    if (this.root === null) {
      if (kind !== "started") return null;
      this.root = call;
      this.rootFunction = e["function"] as string;
      const program = (e["program"] ?? {}) as Rec;
      this.rootKind = (program["kind"] ?? "ai") as string;
      this.answer = (program["answer"] ?? "result") as string;
    }
    return call === this.root ? this.atRoot(e, kind) : this.inside(e, kind);
  }

  private atRoot(e: Rec, kind: string): Rec | null {
    switch (kind) {
      case "started":
        e["program"] = Object.fromEntries(Object.entries((e["program"] ?? {}) as Rec).filter(([k]) => PROGRAM.has(k)));
        delete e["invocation"];
        return this.link(e);
      case "request":
        if (this.rootKind !== "ai") return null;
        this.requests = Math.max(this.requests, Number(e["request"] ?? 0));
        return this.link(e);
      case "retry":
        if (this.rootKind !== "ai") return null;
        delete e["reason"];
        e["content"] = false;
        return this.link(e);
      case "text":
        return e["answer"] ? this.link(e) : null;
      case "approval":
      case "approved":
        return this.approval(e);
      case "done":
        return this.link(e);
      case "failed":
        e["error"] = Object.fromEntries(Object.entries((e["error"] ?? {}) as Rec).filter(([k]) => k === "type" || k === "code"));
        e["content"] = false;
        return this.link(e);
      default:
        return null;
    }
  }

  private approval(e: Rec): Rec | null {
    const key = `${e["call"]}|${e["invocation"]}`;
    if (e["kind"] === "approval") {
      if (e["to"] !== "caller") return null;
      this.callerAsked.add(key);
    } else if (!this.callerAsked.has(key)) return null;
    e["call"] = this.root;
    e["function"] = this.rootFunction;
    return this.link(e);
  }

  private inside(e: Rec, kind: string): Rec | null {
    if (kind === "approval" || kind === "approved") return this.approval(e);
    if (this.rootKind === "ai" || this.answerFrom === null) return null;
    const call = e["call"] as string;
    if (kind === "started") {
      this.forwarded.set(call, e["function"] === this.answerFrom);
      return this.forwarded.get(call) ? this.asRequest(e) : null;
    }
    if (!this.forwarded.get(call)) return null;
    if (kind === "request" || kind === "retry") return this.asRequest(e);
    if (kind === "text" && e["answer"]) {
      e["call"] = this.root;
      e["function"] = this.rootFunction;
      e["field"] = this.answer;
      return this.link(e);
    }
    return null;
  }

  /** A forwarded helper's new request (or its start): the module's answer starts again, as a `request` of the module. */
  private asRequest(e: Rec): Rec {
    this.requests += 1;
    const out: Rec = {};
    for (const k of ["functai_event", "tree", "writer", "seq", "at"]) if (k in e) out[k] = e[k];
    Object.assign(out, { kind: "request", call: this.root, function: this.rootFunction, request: this.requests, model: null });
    return this.link(out);
  }
}

/** A whole form of a log as the outside view shows it. */
export function outside(events: Iterable<Readonly<Rec>>, opts: { answerFrom?: string | { name: string } | null } = {}): Rec[] {
  const v = new View("outside", opts);
  const out: Rec[] = [];
  for (const e of events) {
    const x = v.apply(e);
    if (x) out.push(x);
  }
  return out;
}
