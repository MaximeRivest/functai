/**
 * The reply cache: an identical request (the model, the messages, every
 * setting sent) is answered with the reply it got before, without a model
 * call. Off unless `cacheReplies` is set; the same rule as Python's
 * `cache_replies`. Two rules keep it honest:
 *
 * - only a reply that was read is kept: an unreadable one is forgotten, so a
 *   passing failure never becomes a permanent one;
 * - it answers identical requests identically, sampling included. Leave it
 *   off where you want several different samples of one request.
 *
 * `cacheReplies: true` keeps replies in this process's memory (the last
 * 20,000). Any store with `get`, `set` and `delete` works instead, a `Map`
 * included, or a store shared between processes (Redis, a KV store...): it
 * is given plain JSON under string keys.
 */

import { Request, Response } from "@lm15/lm15";
import * as lmcc from "lmcc";
import { warnOnce } from "./calllog.ts";

type Rec = Record<string, unknown>;

/** Where replies are kept: a `Map` works, or any store with these three (sync or async). */
export interface ReplyCache {
  get(key: string): unknown;
  set(key: string, reply: Rec): unknown;
  delete(key: string): unknown;
}

/** In memory, least recently used out first. */
class Memory implements ReplyCache {
  private readonly data = new Map<string, Rec>();
  private readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get(key: string): Rec | undefined {
    const hit = this.data.get(key);
    if (hit !== undefined) {
      this.data.delete(key);
      this.data.set(key, hit);
    }
    return hit;
  }

  set(key: string, reply: Rec): void {
    this.data.delete(key);
    this.data.set(key, reply);
    while (this.data.size > this.capacity) this.data.delete(this.data.keys().next().value!);
  }

  delete(key: string): void {
    this.data.delete(key);
  }

  clear(): void {
    this.data.clear();
  }
}

const memory = new Memory(20_000);

/** Forget every reply in the in-memory cache (`cacheReplies: true`). */
export function clearCache(): void {
  memory.clear();
}

/** The store a setting names, or null when replies are not cached. */
export function cacheOf(setting: unknown): ReplyCache | null {
  if (setting === true) return memory;
  if (setting && typeof setting === "object" && typeof (setting as ReplyCache).get === "function") return setting as ReplyCache;
  return null;
}

/** The key of a request: everything it sends. */
export function replyKey(request: Request): string {
  return `functai:reply:${lmcc.sha256(Request.toJSON(request) as lmcc.Json)}`;
}

// A store that fails is not a reason to fail the call: it is skipped, with one warning.
const skip = (what: string, err: unknown) => warnOnce(`cache:${what}`, `the reply cache could not ${what} (${(err as Error)?.message ?? err}); calling the model instead`);

export async function lookup(cache: ReplyCache, key: string): Promise<Response | null> {
  try {
    let hit = await cache.get(key);
    if (typeof hit === "string") hit = JSON.parse(hit);         // a store that keeps text
    return hit && typeof hit === "object" ? Response.fromJSON(hit as never) : null;
  } catch (err) {
    skip("read", err);
    return null;
  }
}

export async function keep(cache: ReplyCache, key: string, response: Response): Promise<void> {
  try {
    await cache.set(key, Response.toJSON(response) as Rec);
  } catch (err) {
    skip("write", err);
  }
}

export async function forget(cache: ReplyCache, key: string): Promise<void> {
  try {
    await cache.delete(key);
  } catch (err) {
    skip("forget", err);
  }
}
