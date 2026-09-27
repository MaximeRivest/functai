/**
 * What this package asks of the host, reached without importing it: the
 * package runs in browsers and workers too, where there is no file system
 * and no process. On Node (and Deno, Bun), `process.getBuiltinModule` gives
 * the built-in modules synchronously.
 */

type Builtins = {
  "node:fs": typeof import("node:fs");
  "node:os": typeof import("node:os");
  "node:path": typeof import("node:path");
  "node:async_hooks": typeof import("node:async_hooks");
};

const proc = (globalThis as { process?: { getBuiltinModule?: (id: string) => unknown; env?: Record<string, string | undefined>; pid?: number; version?: string; platform?: string } }).process;

/** A built-in module of the host, or null where there is none (a browser). */
export function builtin<K extends keyof Builtins>(id: K): Builtins[K] | null {
  try {
    return (proc?.getBuiltinModule?.(id) as Builtins[K] | undefined) ?? null;
  } catch {
    return null;
  }
}

/** The environment variables, or an empty record where there are none. */
export function env(): Record<string, string | undefined> {
  return proc?.env ?? {};
}

export function pid(): number | null {
  return proc?.pid ?? null;
}

export function runtime(): string {
  const versions = (globalThis as { process?: { versions?: Record<string, string> } }).process?.versions;
  if (versions?.["bun"]) return `bun ${versions["bun"]}`;
  if (versions?.["deno"]) return `deno ${versions["deno"]}`;
  if (versions?.["node"]) return `node ${versions["node"]}`;
  const nav = (globalThis as { navigator?: { userAgent?: string } }).navigator;
  return nav?.userAgent ?? "unknown";
}

export function platform(): string {
  return proc?.platform ?? "web";
}

/**
 * Context that follows a call through `await`: Node's AsyncLocalStorage when
 * there is one; elsewhere, a plain slot that is right for calls made one at a
 * time (the parent of a call made in parallel may then be missed).
 */
export class Context<T> {
  private readonly als: { getStore(): T | undefined; run<R>(store: T, fn: () => R): R } | null;
  private slot: T | undefined;

  constructor() {
    const hooks = builtin("node:async_hooks");
    this.als = hooks ? new hooks.AsyncLocalStorage<T>() : null;
  }

  get(): T | undefined {
    return this.als ? this.als.getStore() : this.slot;
  }

  run<R>(value: T, fn: () => R): R {
    if (this.als) return this.als.run(value, fn);
    const before = this.slot;
    this.slot = value;
    try {
      const out = fn();
      if (out instanceof Promise) {
        return out.finally(() => { this.slot = before; }) as R;
      }
      this.slot = before;
      return out;
    } catch (err) {
      this.slot = before;
      throw err;
    }
  }
}
