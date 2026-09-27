/** A fake lm15 router: answers each request from a script, records every request. No network. */

import { Message, Request, Response, responseToEvents, streamDelta, streamEnd, streamStart, toolCall, type StreamEvent } from "@lm15/lm15";

export type Reply = string | { text?: string; calls?: { id: string; name: string; input: Record<string, unknown> }[]; finish?: string };

export class FakeRouter {
  readonly requests: Request[] = [];
  private readonly replies: Reply[];
  readonly responder: ((request: Request, i: number) => Reply) | null;
  readonly provider: string;
  /** Cut each streamed text into pieces of this many characters. */
  readonly piece: number;

  constructor(replies: Reply[] = [], responder: ((request: Request, i: number) => Reply) | null = null, provider = "openai", piece = 3) {
    this.replies = [...replies];
    this.responder = responder;
    this.provider = provider;
    this.piece = piece;
  }

  resolve(model: string): { provider: string; model: string } {
    const i = model.indexOf(":");
    return i > 0 ? { provider: model.slice(0, i), model: model.slice(i + 1) } : { provider: this.provider, model };
  }

  private reply(request: Request): Response {
    const i = this.requests.length;
    this.requests.push(request);
    const r = this.responder ? this.responder(request, i) : this.replies.shift();
    if (r === undefined) throw new Error("the fake router has no more replies");
    const spec = typeof r === "string" ? { text: r } : r;
    const parts = [
      ...(spec.text !== undefined ? [{ type: "text", text: spec.text }] : []),
      ...(spec.calls ?? []).map((c) => toolCall(c.id, c.name, c.input as never)),
    ];
    return new Response({
      model: request.model, message: Message.assistant(parts as never), finishReason: (spec.finish ?? (spec.calls?.length ? "tool_call" : "stop")) as never,
      usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
    });
  }

  async complete(request: Request): Promise<Response> {
    return this.reply(request);
  }

  async *stream(request: Request): AsyncIterable<StreamEvent> {
    const response = this.reply(request);
    for (const e of responseToEvents(response)) {
      const d = e.type === "delta" ? (e.delta as { type: string; text?: string }) : null;
      if (d && d.type === "text" && d.text && d.text.length > this.piece) {
        for (let k = 0; k < d.text.length; k += this.piece) {
          await new Promise((r) => setTimeout(r, 0));
          yield streamDelta({ ...d, text: d.text.slice(k, k + this.piece) } as never);
        }
        continue;
      }
      yield e;
    }
  }
}

/** A router that has no stream: replies arrive whole. */
export function whole(router: FakeRouter): { resolve: FakeRouter["resolve"]; complete: FakeRouter["complete"]; requests: Request[] } {
  return { resolve: (m) => router.resolve(m), complete: (r) => router.complete(r), requests: router.requests };
}

export { Request, streamStart, streamEnd };
