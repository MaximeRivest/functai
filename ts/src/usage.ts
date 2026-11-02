// Token counts (contract/calls.md, *Token counts*). lm15 2 gives each counter
// one meaning on every provider: input includes cached tokens, output includes
// reasoning. A sum over model calls keeps a counter only when every call
// reported it; counts lm15 1 wrote are shown only when they cannot mislead.

import { getOwn, setOwn } from "./values.ts";

/**
 * Token counts summed over model calls: a counter is in the sum only when
 * every call reported it, so a sum is exact or absent, never a part of the
 * total. No calls: `{}`.
 */
export function sumUsage(parts: Iterable<Record<string, number>>): Record<string, number> {
  let total: Record<string, number> | null = null;
  for (const part of parts) {
    const next: Record<string, number> = {};
    if (total === null) for (const [k, v] of Object.entries(part)) setOwn(next, k, v);
    else for (const [k, v] of Object.entries(total)) {
      const more = getOwn(part, k);
      if (more !== undefined) setOwn(next, k, v + more);
    }
    total = next;
  }
  return total ?? {};
}

const count = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

/**
 * Token counts an earlier run kept, with the meaning lm15 2 gives them.
 * `lm15`: the lm15 that wrote them (a call record's `process.lm15`), or
 * null when the writer did not say (then lm15 1). lm15 1 kept each
 * provider's own counting (Anthropic's input left the cache out, Gemini's
 * and xAI's output the reasoning): its counts are kept only when
 * `total_tokens` is `input_tokens + output_tokens` and no cache count is
 * above 0; otherwise every count is unknown (`{}`).
 */
export function usageRead(usage: unknown, lm15: unknown): Record<string, number> {
  const counts: Record<string, number> = {};
  if (usage && typeof usage === "object") for (const [k, v] of Object.entries(usage)) if (count(v)) setOwn(counts, k, v);
  const major = typeof lm15 === "string" ? Number.parseInt(lm15.split(".")[0]!, 10) : NaN;
  if (major >= 2) return counts;
  const [i, o, t] = [getOwn(counts, "input_tokens"), getOwn(counts, "output_tokens"), getOwn(counts, "total_tokens")];
  if (i === undefined || o === undefined || t === undefined || t !== i + o) return {};
  if ((getOwn(counts, "cache_read_tokens") ?? 0) > 0 || (getOwn(counts, "cache_write_tokens") ?? 0) > 0) return {};
  return counts;
}
