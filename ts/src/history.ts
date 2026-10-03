/**
 * The last requests this process sent (or answered from its reply cache), in
 * memory: what went to the provider and what came back, for a quick look
 * while developing. Every call, kept on disk: the call log
 * (`configure({ logCalls: true })`, `calls()`).
 */

import { Request, Response } from "@lm15/lm15";

/** One request sent (or answered from the cache) and its reply. */
export interface HistoryRecord {
  readonly function: string;
  readonly model: string;
  /** Exactly what went to the provider (lm15's request). */
  readonly request: Request;
  /** What came back; null when the provider failed. */
  readonly response: Response | null;
  readonly cached: boolean;
  readonly error: string | null;
  readonly timestamp: Date;
}

const LIMIT = 500;
const records: HistoryRecord[] = [];

/** @internal */
export function keep(r: Omit<HistoryRecord, "timestamp">): void {
  records.push({ ...r, timestamp: new Date() });
  if (records.length > LIMIT) records.splice(0, records.length - LIMIT);
}

/** The last `n` requests FunctAI sent (or answered from its cache), oldest first; the last 500 are kept, in this process only. */
export function inspectHistory(n = 1): HistoryRecord[] {
  return n > 0 ? records.slice(-n) : [];
}

/** Forget them. */
export function clearHistory(): void {
  records.length = 0;
}

type Part = { type?: string; text?: string; name?: string; input?: unknown; id?: string; content?: unknown; value?: unknown };

function partText(p: Part): string {
  switch (p.type) {
    case "text":
      return p.text ?? "";
    case "thinking":
      return `[thinking]\n${p.text ?? ""}`;
    case "tool_call":
      return `[tool call ${p.name ?? "?"}(${JSON.stringify(p.input ?? {})})]`;
    case "tool_result":
      return `[tool result ${p.id ?? ""}] ${Array.isArray(p.content) ? (p.content as Part[]).map(partText).join(" ") : String(p.content ?? "")}`;
    case "data":
      return JSON.stringify(p.value ?? null);
    default:
      return `[${p.type ?? "part"}]${p.text ? ` ${p.text}` : ""}`;
  }
}

const messageText = (m: { parts?: readonly unknown[] }) => (m.parts ?? []).map((p) => partText(p as Part)).join("\n");

/**
 * The last `n` model calls as readable text: every message sent, the reply,
 * the finish reason and the tokens. `console.log(phistory())`.
 */
export function phistory(n = 1): string {
  const out = inspectHistory(n).map((r) => {
    const req = Request.toJSON(r.request) as { system?: unknown; messages?: { role: string; parts?: unknown[] }[]; tools?: { name?: string }[] };
    const lines = [`[${r.timestamp.toISOString().slice(0, 19)}] ${r.function} → ${r.model}${r.cached ? " (from cache)" : ""}`, ""];
    if (req.system) lines.push("System message:", "", typeof req.system === "string" ? req.system : messageText({ parts: req.system as unknown[] }), "");
    for (const m of req.messages ?? []) lines.push(`${m.role[0]!.toUpperCase()}${m.role.slice(1)} message:`, "", messageText(m), "");
    if (req.tools?.length) lines.push(`Tools: ${req.tools.map((t) => t.name ?? "?").join(", ")}`, "");
    if (r.response) {
      const res = Response.toJSON(r.response) as { message?: { parts?: unknown[] }; finish_reason?: string; usage?: { input_tokens?: number; output_tokens?: number } };
      lines.push("Response:", "", messageText(res.message ?? {}), "", `(finish: ${res.finish_reason}; tokens in ${res.usage?.input_tokens ?? "?"}, out ${res.usage?.output_tokens ?? "?"})`);
    } else if (r.error) lines.push("Error:", "", r.error);
    return lines.join("\n").trimEnd();
  });
  return out.length ? out.join(`\n\n${"─".repeat(60)}\n\n`) : "(no model calls yet)";
}
