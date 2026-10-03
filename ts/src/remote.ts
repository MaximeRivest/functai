/**
 * A program served elsewhere, used like a local one (contract/serving.md).
 *
 * ```ts
 * const team = await remote("https://lambda.example/programs/team", { key: process.env.TEAM_KEY });
 * await team({ message: "I was charged twice for order B-2210." });     // "billing"
 * await team.map(tickets, { concurrency: 8 });                          // a table, as with a local program
 * ```
 *
 * Its interface is the server's (`GET /interface`): its inputs are bound
 * and checked here before anything is sent, and what comes back is checked
 * against its outputs. Each call is logged here (`program.kind` `"remote"`,
 * `program.remote` the URL) and there; the server's record names this call
 * as its `parent`, so the two logs make one call tree.
 */

import { RemoteError } from "./errors.ts";
import { checkInterface, InterfaceError, type Interface } from "./interface.ts";
import { module, type AnyModule } from "./module.ts";
import { parseData, toJson, writeData } from "./values.ts";
import type { CallOptions } from "./fn.ts";

type Rec = Record<string, unknown>;

async function request(url: string, key: string | null, body: unknown, opts: { timeout: number; parent?: string | null; signal?: AbortSignal; stream?: boolean }): Promise<globalThis.Response> {
  const headers: Record<string, string> = { accept: opts.stream ? "text/event-stream" : "application/json" };
  if (body !== undefined) headers["content-type"] = "application/json";
  if (key) headers["authorization"] = `Bearer ${key}`;
  if (opts.parent) headers["functai-parent"] = opts.parent;
  const signals = [AbortSignal.timeout(opts.timeout), ...(opts.signal ? [opts.signal] : [])];
  const resp = await fetch(url, { method: body === undefined ? "GET" : "POST", headers, body: body === undefined ? undefined : writeData(body), signal: opts.stream ? opts.signal : AbortSignal.any(signals) });
  if (resp.status >= 400) {
    let err: Rec = {};
    try {
      err = ((parseData(await resp.text()) as Rec)["error"] ?? {}) as Rec;
    } catch {
      err = {};
    }
    const code = (err["code"] as string | undefined) ?? `remote-${resp.status}`;
    const text = (err["message"] as string | undefined) ?? `${url} answered ${resp.status} (${err["type"] ?? resp.statusText})`;
    if (code === "interface-input" || code === "interface-output") throw new InterfaceError(code, (err["field"] ?? null) as string | null, text);
    throw new RemoteError(code, text, { status: resp.status, type: (err["type"] ?? "") as string });
  }
  return resp;
}

/** A served program, called like a local one: an AI function's or a module's surface, its code a request to the server. */
export type RemoteProgram = AnyModule & {
  /** Where it is served. */
  readonly url: string;
  /** The server's description of it (`GET /interface`). */
  readonly described: Rec;
  /** The served call, watched: the server's outside view, as the contract's JSON events (not logged here). */
  events(input: unknown, options?: CallOptions): AsyncGenerator<Rec>;
};

/**
 * A program served elsewhere (`serve(program)`, `functai serve`), used like a
 * local one: called, mapped over rows, evaluated, its calls logged here too.
 * `key`: the key the server asks for; `timeout`: milliseconds to wait for one
 * answer (default 120,000). Rejects with `RemoteError` when the server
 * refuses (`remote-401` for a wrong key) or is not a FunctAI program.
 */
export async function remote(url: string, opts: { key?: string | null; timeout?: number } = {}): Promise<RemoteProgram> {
  const base = url.replace(/\/+$/, "");
  const key = opts.key ?? null;
  const timeout = opts.timeout ?? 120_000;
  const described = parseData(await (await request(base + "/interface", key, undefined, { timeout })).text()) as Rec;
  if (described["functai_interface"] !== 1) {
    throw new RemoteError("remote-format", `${url} does not describe a FunctAI program this version reads (functai_interface ${JSON.stringify(described["functai_interface"] ?? null)})`);
  }
  const iface = checkInterface(described["interface"] as Interface, { ai: described["kind"] === "ai", where: `remote("${url}")` });
  const names = iface.outputs.map((f) => f.name);
  const shapeOf = (f: Interface["inputs"][number]) => ({ shape: f.shape, ...(f.desc ? { desc: f.desc } : {}), ...(f.optional ? { optional: true } : {}) });
  const program = module(described["name"] as string, {
    _remote: { url: base, version: described["version"] as string },
    description: iface.description ?? "",
    input: Object.fromEntries(iface.inputs.map((f) => [f.name, shapeOf(f)])) as never,
    outputs: Object.fromEntries(iface.outputs.map((f) => [f.name, { shape: f.shape, ...(f.desc ? { desc: f.desc } : {}) }])) as never,
    definedIn: "functai.remote",
  }, async (inputs: Rec, { signal, callId }) => {
    const resp = await request(base + "/call", key, { inputs: Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, toJson(v)[0]])) },
      { timeout, parent: callId, signal });
    const got = parseData(await resp.text()) as Rec;
    const outputs = (got["outputs"] ?? {}) as Rec;
    return (names.length === 1 ? outputs[names[0]!] : Object.fromEntries(names.map((n) => [n, outputs[n]]))) as never;
  }) as unknown as AnyModule;
  Object.defineProperties(program, { url: { value: base }, described: { value: described } });
  Object.assign(program, {
    async *events(input: unknown, options: CallOptions = {}): AsyncGenerator<Rec> {
      const inputs = (program as unknown as { _recordedInputs(i: unknown): Rec })._recordedInputs(input);
      const resp = await request(base + "/stream", key, { inputs }, { timeout, stream: true, ...(options.signal ? { signal: options.signal } : {}) });
      const reader = resp.body!.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += value;
        let end: number;
        while ((end = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, end);
          buffer = buffer.slice(end + 2);
          const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
          if (data) yield parseData(data) as Rec;
        }
      }
    },
    conversation: () => {
      throw new TypeError(`a remote program's conversations are kept by its server: POST ${base}/conversations/<id>/turns`);
    },
  });
  return program as RemoteProgram;
}
