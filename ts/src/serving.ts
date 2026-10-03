/**
 * Serving a program over HTTP (contract/serving.md).
 *
 * ```ts
 * await serve(team, { port: 8080, keys: "keys.txt" });         // Node's own HTTP server
 * const service = new Service(team, { keys: [process.env.TEAM_KEY!] });
 * export default { fetch: service.fetch };                      // Deno, Bun, Cloudflare Workers, Hono, a Next.js route
 * ```
 *
 * What a caller sees is the program's boundary (the `outside` view): its
 * answer, its answer's text as it is written, approvals addressed to it, and
 * its end; never a helper's answer, a tool's input or output, a thinking, or
 * an error's message. The owner watches everything in their own log.
 *
 * Routes (JSON in, JSON out; `Authorization: Bearer <key>` when the service
 * has keys): `GET /interface`, `GET /openapi.json`, `GET /` (a form),
 * `POST /call`, `POST /stream` (Server-Sent Events), and a conversation's
 * turns: `POST /conversations/<id>/turns`, `GET /conversations/<id>/turns`,
 * `GET …/turns/<turn>`, `GET …/turns/<turn>/events`, `POST …/turns/<turn>/stop`,
 * `POST …/turns/<turn>/approvals/<invocation>`.
 *
 * A caller that is itself a FunctAI call sends `FunctAI-Parent: <its call
 * id>`: the served call's record names it as its `parent`, so the two logs
 * make one call tree.
 */

import * as calllog from "./calllog.ts";
import { InterfaceError, checkInputs } from "./interface.ts";
import { ConversationError, FunctAIError, ServeError } from "./errors.ts";
import { builtin } from "./host.ts";
import { withSettings, type Settings } from "./settings.ts";
import { APPROVALS_TO } from "./tools.ts";
import { parseData, toJson, writeData } from "./values.ts";
import type { Conversation, ConversationOptions, Turn, TurnStream } from "./conversations.ts";
import type { Stream } from "./stream.ts";
import type { Interface } from "./interface.ts";

type Rec = Record<string, unknown>;

/** The interface's format number when it is served (outside a saved manifest). */
export const FORMAT = 1;
/** Bytes a request may send. */
const MAX_BODY = 16 * 1024 * 1024;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** What can be served: an AI function or a module (made by `ai()` or `module()`). */
export interface Servable {
  readonly name: string;
  readonly interface: Interface;
  readonly version: string;
  stream(input: unknown, options?: Rec): Stream;
  conversation(id?: string | null, options?: ConversationOptions): Conversation;
  using(settings: Settings): Servable;
  /** @internal */
  _programInfo(): calllog.Program;
}

/** Options of a service. */
export interface ServeOptions {
  /** Bearer keys a caller must send: a list, one key, or a file of one key per line (`#` comments skipped). None: no key, which `serve` allows only on this machine. */
  keys?: readonly string[] | string | null;
  /** Where conversations are kept (as `fn.conversation({ store })`); null keeps them in this process's memory. */
  store?: ConversationOptions["store"];
  /** The model every call uses, bound when serving starts. */
  lm?: string | null;
  /** Who answers a tool call the program's `approve` rule asks about: the owner (in their own process), or the caller, through the approvals route. */
  approvals?: "owner" | "caller";
}

const enc = new TextEncoder();
const json = (status: number, data: unknown, headers: Record<string, string> = {}): Response =>
  new Response(writeData(data), { status, headers: { "content-type": "application/json", ...headers } });

function errorResponse(status: number, err: unknown, message = false): Response {
  const e = err as { name?: string; code?: unknown; field?: unknown; message?: string };
  const out: Rec = { type: e?.name ?? "Error" };
  if (typeof e?.code === "string") out["code"] = e.code;
  if (err instanceof InterfaceError && e.field !== null && e.field !== undefined) out["field"] = e.field;
  if (message) out["message"] = e?.message ?? String(err);
  return json(status, { error: out });
}

function readKeys(keys: ServeOptions["keys"]): string[] {
  if (keys === undefined || keys === null) return [];
  if (typeof keys === "string") {
    const fs = builtin("node:fs");
    if (fs?.existsSync(keys) && fs.statSync(keys).isFile()) {
      return fs.readFileSync(keys, "utf8").split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
    }
    return [keys];
  }
  return [...keys].map(String);
}

/** Two texts compared in constant time (their lengths aside). */
function sameKey(a: string, b: string): boolean {
  const x = enc.encode(a);
  const y = enc.encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return diff === 0;
}

/** An event's position as an SSE id: `<writer>-<seq>`. */
export const positionId = (e: Rec) => `${e["writer"]}-${e["seq"]}`;

/** `<writer>-<seq>` back to a position (null when absent or not one). */
export function parsePosition(text: string | null | undefined): { writer: number; seq: number } | null {
  const m = /^\s*(\d+)-(\d+)\s*$/.exec(text ?? "");
  return m ? { writer: Number(m[1]), seq: Number(m[2]) } : null;
}

const sse = (e: Rec) => enc.encode(`id: ${positionId(e)}\nevent: ${e["kind"]}\ndata: ${writeData(e)}\n\n`);

function eventStream(events: AsyncIterable<Rec>, onCancel?: () => void): Response {
  const it = events[Symbol.asyncIterator]();
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const next = await it.next();
        if (next.done) controller.close();
        else controller.enqueue(sse(next.value));
      } catch {
        controller.close();                                  // the call's own end is in the events
      }
    },
    cancel() {
      onCancel?.();                                          // a reader gone stops the call
      void it.return?.();
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream", "cache-control": "no-cache" } });
}

/**
 * A program as an HTTP service, independent of any server: `fetch(request)`
 * answers one request (the Web's own `Request` and `Response`); `serve()`
 * runs Node's HTTP server around it.
 */
export class Service {
  readonly program: Servable;
  readonly keys: readonly string[];
  readonly store: ConversationOptions["store"];
  readonly lm: string | null;
  readonly approvals: "owner" | "caller";
  private readonly conversations = new Map<string, Conversation>();

  constructor(program: Servable, opts: ServeOptions = {}) {
    if (!program || typeof program.stream !== "function" || !program.interface) throw new TypeError("serve an AI function or a module");
    const opaque = [...program.interface.inputs, ...program.interface.outputs].filter((f) => f.opaque).map((f) => f.name);
    if (opaque.length) {
      throw new ServeError("serve-opaque", `${program.name} cannot be served: ${opaque.join(", ")} may hold values with no JSON form, and only JSON crosses HTTP. Give ${opaque.length > 1 ? "them" : "it"} a type`);
    }
    const approvals = opts.approvals ?? "owner";
    if (approvals !== "owner" && approvals !== "caller") throw new TypeError(`approvals go to "owner" or "caller", not ${JSON.stringify(approvals)}`);
    this.program = opts.lm ? program.using({ lm: opts.lm }) : program;
    this.keys = readKeys(opts.keys);
    this.store = opts.store ?? null;
    this.lm = opts.lm ?? null;
    this.approvals = approvals;
    this.fetch = this.fetch.bind(this);
  }

  /** The program, described: `{ functai_interface: 1, name, kind, version, interface, answer, model, approvals }`. */
  describe(): Rec {
    const info = this.program._programInfo();
    return { functai_interface: FORMAT, name: this.program.name, kind: info.kind, version: info.version, interface: this.program.interface,
      answer: info.answer, model: this.lm, approvals: this.approvals };
  }

  /** The same, as OpenAPI 3.1. */
  openapi(): Rec {
    const iface = this.program.interface;
    const ins = { type: "object", properties: Object.fromEntries(iface.inputs.map((f) => [f.name, f.shape])), required: iface.inputs.filter((f) => !f.optional).map((f) => f.name) };
    const outs = { type: "object", properties: Object.fromEntries(iface.outputs.map((f) => [f.name, f.shape])) };
    const body = { required: true, content: { "application/json": { schema: { type: "object", properties: { inputs: ins }, required: ["inputs"] } } } };
    return {
      openapi: "3.1.0", info: { title: this.program.name, version: this.program.version, description: iface.description ?? "" },
      paths: {
        "/call": { post: { requestBody: body, responses: { "200": { description: "the answer", content: { "application/json": { schema: {
          type: "object", properties: { call: { type: "string" }, outputs: outs, value: {} } } } } } } } },
        "/stream": { post: { requestBody: body, responses: { "200": { description: "Server-Sent Events", content: { "text/event-stream": {} } } } } },
      },
      components: { securitySchemes: { key: { type: "http", scheme: "bearer" } } },
      security: this.keys.length ? [{ key: [] }] : [],
    };
  }

  /** A form, which asks for the key itself. */
  form(): string {
    const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
    const iface = this.program.interface;
    const rows = iface.inputs.map((f) => `<label>${esc(f.name)}<br><textarea name="${esc(f.name)}" rows="3"></textarea></label><br>`).join("");
    return `<!doctype html><meta charset=utf-8><title>${esc(this.program.name)}</title>`
      + "<style>body{font:16px system-ui;max-width:40em;margin:2em auto}textarea{width:100%}</style>"
      + `<h1>${esc(this.program.name)}</h1><p>${esc(iface.description ?? "")}</p>`
      + `<form id=f>${rows}<label>key <input name=__key type=password></label> <button>Ask</button></form><pre id=out></pre><script>`
      + "f.onsubmit=async e=>{e.preventDefault();const d=new FormData(f),inputs={};"
      + "for(const[k,v]of d)if(k!='__key'){try{inputs[k]=JSON.parse(v)}catch{inputs[k]=v}}"
      + "const r=await fetch('call',{method:'POST',headers:{'content-type':'application/json',"
      + "authorization:'Bearer '+d.get('__key')},body:JSON.stringify({inputs})});"
      + "out.textContent=JSON.stringify(await r.json(),null,2)}</script>";
  }

  private authorized(headers: Headers): boolean {
    if (!this.keys.length) return true;
    const got = headers.get("authorization") ?? "";
    if (!got.toLowerCase().startsWith("bearer ")) return false;
    const given = got.slice(7).trim();
    let ok = false;
    for (const k of this.keys) ok = sameKey(given, k) || ok;    // every key compared: no early exit
    return ok;
  }

  /** Answer one request. Mount it anywhere the Web's `fetch` handlers run. */
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const path = "/" + url.pathname.replace(/^\/+|\/+$/g, "");
    try {
      if (request.method === "GET" && path === "/") return new Response(this.form(), { status: 200, headers: { "content-type": "text/html; charset=utf-8" } });
      if (!this.authorized(request.headers)) return json(401, { error: { type: "Unauthorized" } });
      if (request.method === "GET" && path === "/interface") return json(200, this.describe());
      if (request.method === "GET" && path === "/openapi.json") return json(200, this.openapi());
      let data: Rec = {};
      if (request.method === "POST") {
        const text = await request.text();
        if (text.length > MAX_BODY) return json(413, { error: { type: "TooLarge" } });
        if (text) {
          try {
            data = parseData(text) as Rec;
          } catch {
            return json(400, { error: { type: "BadRequest", message: "the body is not JSON" } });
          }
          if (typeof data !== "object" || data === null || Array.isArray(data)) return json(400, { error: { type: "BadRequest", message: "the body is a JSON object" } });
        }
      }
      const header = request.headers.get("functai-parent");
      const parent = header && UUID.test(header) ? header : null;
      const parts = path.split("/").filter(Boolean);
      if (request.method === "POST" && path === "/call") return await this.call(data, parent);
      if (request.method === "POST" && path === "/stream") return this.stream(data, parent, request.signal);
      if (parts[0] === "conversations" && parts.length >= 3 && parts[2] === "turns") {
        return await this.conversation(request.method, parts[1]!, parts.slice(3), data, request.headers, parent);
      }
      return json(404, { error: { type: "NotFound" } });
    } catch (err) {
      if (err instanceof InterfaceError) return errorResponse(422, err, err.code === "interface-input");
      if (err instanceof ConversationError) {
        const status = ({ "turn-unknown": 404, "conversation-id": 400, "conversation-busy": 409 } as Record<string, number>)[err.code] ?? 409;
        return errorResponse(status, err, true);
      }
      if (err instanceof FunctAIError) return errorResponse(409, err);
      return errorResponse(500, err);                        // the type, never the message (outside view)
    }
  }

  /** The request's inputs, checked against the interface before anything runs (`interface-input`, 422). */
  private inputs(data: Rec): Rec {
    const inputs = data["inputs"] ?? {};
    if (typeof inputs !== "object" || inputs === null || Array.isArray(inputs)) {
      throw new InterfaceError("interface-input", null, "inputs is a JSON object of the program's inputs");
    }
    checkInputs(this.program.interface, inputs as Rec, this.program.name, []);
    return inputs as Rec;
  }

  /** Run `fn` as a call made for this service's caller: its caller, where approvals go, the remote caller's call as its parent. */
  private scope<R>(parent: string | null, fn: () => R): R {
    return withSettings({ caller: { kind: "api" } }, () => APPROVALS_TO.run(this.approvals, () => calllog.REMOTE_PARENT.run({ id: parent }, fn)));
  }

  private async call(data: Rec, parent: string | null): Promise<Response> {
    const inputs = this.inputs(data);
    const s = this.scope(parent, () => this.program.stream(inputs, {}));
    const value = await s.result;
    const prediction = (s as unknown as { prediction?: Promise<{ outputs: Rec }> }).prediction;
    const names = this.program.interface.outputs.map((f) => f.name);
    let outputs: Rec;
    if (prediction) {
      const p = await prediction;
      outputs = Object.fromEntries(names.filter((n) => Object.hasOwn(p.outputs, n)).map((n) => [n, toJson(p.outputs[n])[0]]));
    } else outputs = names.length === 1 ? { [names[0]!]: toJson(value)[0] } : Object.fromEntries(names.map((n) => [n, toJson((value as Rec)?.[n])[0]]));
    return json(200, { call: s.callId, outputs, value: toJson(value)[0] });
  }

  private stream(data: Rec, parent: string | null, signal: AbortSignal): Response {
    const inputs = this.inputs(data);
    const s = this.scope(parent, () => this.program.stream(inputs, { signal }));
    s.result.catch(() => undefined);
    return eventStream(s.events({ view: "outside" }) as unknown as AsyncIterable<Rec>, () => s.close());
  }

  /** The conversation of this id (as the program's, in this service's store). */
  conversation_(cid: string): Conversation {
    let chat = this.conversations.get(cid);
    if (!chat) {
      chat = this.program.conversation(cid, { store: this.store });
      if (this.conversations.size > 1000) this.conversations.clear();
      this.conversations.set(cid, chat);
    }
    return chat;
  }

  private turnJson(t: Turn): Rec {
    const names = new Set(this.program.interface.outputs.map((f) => f.name));
    const out: Rec = { turn: t.id, parent: t.parent, state: t.state, inputs: t.inputs };
    if (t.state === "done") {
      out["outputs"] = Object.fromEntries(Object.entries(t.outputs).filter(([k]) => names.has(k)));
      out["value"] = toJson(t.result)[0];
    }
    if (t.state === "waiting" && this.approvals === "caller") {
      out["waiting"] = t.waiting.map((a) => ({ invocation: a.invocation, name: a.name, input: toJson(a.input)[0], path: a.path }));
    }
    if (t.error) out["error"] = Object.fromEntries(Object.entries(t.error).filter(([k]) => k === "type" || k === "code"));
    return out;
  }

  private async conversation(method: string, cid: string, rest: string[], data: Rec, headers: Headers, parent: string | null): Promise<Response> {
    let chat = this.conversation_(cid);
    if (method === "POST" && !rest.length) {
      const inputs = this.inputs(data);
      if (data["after"]) chat = chat.continueFrom(String(data["after"]));
      const s: TurnStream = this.scope(parent, () => chat.stream(inputs, { requestId: (data["request_id"] ?? null) as string | null }));
      s.result.catch(() => undefined);
      if (data["wait"]) {
        try {
          await s.result;
        } catch {
          // the turn's state says how it went
        }
      }
      const t = await (await s.turn).refresh();
      return json(201, { ...this.turnJson(t), conversation: cid });
    }
    if (method === "GET" && !rest.length) return json(200, { conversation: cid, turns: (await chat.turns()).map((t) => this.turnJson(t)) });
    const turn = await chat.turn(rest[0]!);
    if (method === "GET" && rest.length === 1) return json(200, this.turnJson(turn));
    if (method === "GET" && rest[1] === "events" && rest.length === 2) {
      const after = parsePosition(headers.get("last-event-id"));
      return eventStream(turn.events({ after, view: "outside" }));
    }
    if (method === "POST" && rest[1] === "stop" && rest.length === 2) {
      await turn.stop();
      return json(202, { turn: turn.id, stopping: true });
    }
    if (method === "POST" && rest.length === 3 && rest[1] === "approvals") {
      if (this.approvals !== "caller") return json(403, { error: { type: "Forbidden", message: "approvals go to the owner" } });
      const verdict = data["verdict"];
      if (verdict !== "yes" && verdict !== "no") return json(400, { error: { type: "BadRequest", message: "verdict is 'yes' or 'no'" } });
      const inv = Number(rest[2]);
      await this.scope(parent, async () => {
        if (verdict === "yes") await turn.approve(inv, { resume: false });
        else await turn.deny(inv, (data["reason"] ?? null) as string | null, { resume: false });
        const again = await chat.turn(turn.id);
        // the turn resumes once nothing else waits, in the caller's scope: approvals stay addressed to it
        if (!again.waiting.length) void again.resume().catch(() => undefined);
      });
      return json(202, { turn: turn.id, verdict });
    }
    return json(404, { error: { type: "NotFound" } });
  }

  /**
   * Run Node's HTTP server around this service. Without keys it listens only
   * on this machine (127.0.0.1, ::1, localhost). Resolves once listening, to
   * the server (`server.close()` stops it).
   */
  async serve(opts: { host?: string; port?: number } = {}): Promise<import("node:http").Server> {
    const host = opts.host ?? "127.0.0.1";
    if (!this.keys.length && !["127.0.0.1", "::1", "localhost"].includes(host)) {
      throw new ServeError("serve-keys", `serving on ${host} lets anyone on the network call ${this.program.name} and spend your model budget: give keys (keys: "keys.txt"), or serve on 127.0.0.1`);
    }
    const http = builtin("node:http" as never) as typeof import("node:http") | null;
    if (!http) throw new Error("serve() needs Node's HTTP server; elsewhere, mount service.fetch");
    const server = http.createServer((req, res) => void this.node(req, res));
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(opts.port ?? 8080, host, () => resolve());
    });
    return server;
  }

  /** One request from Node's server, through `fetch`. */
  private async node(req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse): Promise<void> {
    const controller = new AbortController();
    res.on("close", () => { if (!res.writableFinished) controller.abort(); });
    try {
      const chunks: Buffer[] = [];
      let size = 0;
      for await (const c of req) {
        size += (c as Buffer).length;
        if (size > MAX_BODY) {
          res.writeHead(413, { "content-length": "0", connection: "close" }).end();
          return;
        }
        chunks.push(c as Buffer);
      }
      const headers = new Headers();
      for (const [k, v] of Object.entries(req.headers)) if (typeof v === "string") headers.set(k, v);
      const body = req.method === "GET" || req.method === "HEAD" ? undefined : Buffer.concat(chunks);
      const response = await this.fetch(new Request(`http://${req.headers.host ?? "localhost"}${req.url ?? "/"}`, {
        method: req.method ?? "GET", headers, body, signal: controller.signal,
      }));
      res.writeHead(response.status, Object.fromEntries(response.headers.entries()));
      if (!response.body) {
        res.end();
        return;
      }
      const reader = response.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(value);
      }
      res.end();
    } catch {
      if (!res.headersSent) res.writeHead(500, { "content-type": "application/json" });
      res.end();
    }
  }
}

/**
 * Serve a program over HTTP: its interface, calls, streams and conversations,
 * to callers who see only its boundary. Resolves to Node's server once it
 * listens (`server.close()` stops it). Without `keys` it listens only on this
 * machine. Elsewhere than Node, mount `new Service(program).fetch`.
 */
export function serve(program: Servable, opts: ServeOptions & { host?: string; port?: number } = {}): Promise<import("node:http").Server> {
  const { host, port, ...rest } = opts;
  return new Service(program, rest).serve({ ...(host ? { host } : {}), ...(port !== undefined ? { port } : {}) });
}
