/**
 * A program's inputs, from how TypeScript declares them to the interface
 * (programs.md) and back: which inputs a caller may leave out, the value a
 * left-out input takes, and binding a call's argument to inputs by name.
 */

import { bindValue, InterfaceError, type InterfaceField } from "./interface.ts";
import { allowsNull, isOpaque, readField, saysOptional, standardOf, type FieldSpec, type StandardResult, type StandardSchemaLike } from "./shapes.ts";
import { byCodePoint, copyData, entriesOf, getOwn, jsonForm, setOwn } from "./values.ts";

type Rec = Record<string, unknown>;

/**
 * Whether a validation's result is to be awaited: any thenable, not only
 * this realm's `Promise` (a schema from a `vm` context, an iframe or a
 * polyfill returns its own).
 */
const isThenable = (r: unknown): r is PromiseLike<StandardResult> =>
  (typeof r === "object" || typeof r === "function") && r !== null && typeof (r as { then?: unknown }).then === "function";

/** What a call does with one input. */
export interface InputRule {
  readonly optional: boolean;
  /**
   * The value a left-out input takes (absent: it stays left out, a module's
   * only); `compute`, a default given as a function, gives it at each call.
   */
  readonly fill?: { readonly value: unknown; readonly compute?: () => unknown };
  /** A Standard Schema that checks and parses a given value (zod, valibot, …). */
  readonly schema: StandardSchemaLike | null;
}

/** One declared input: its interface field, and how a call treats it. */
export interface DeclaredInput {
  readonly field: InterfaceField;
  readonly rule: InputRule;
  /** A default given as a function: its source, as the version counts it (calls.md, "Versions", "Defaults"). */
  readonly code?: string;
}

/**
 * A default function's code as a version counts it (calls.md, "Versions",
 * "Defaults"): the expression it returns (after `=>`, or a lone `return`'s),
 * each run of white space one space; its whole text when it is written
 * otherwise. `() => today()` is `today()`, as Python's `today()` is.
 */
export function defaultCode(f: (...args: never[]) => unknown): string {
  const text = Function.prototype.toString.call(f).replace(/\s+/g, " ").trim();
  const arrow = /^(?:async )?\( ?\) ?=> ?([\s\S]+)$/.exec(text);
  const body = arrow ? arrow[1]!.trim() : (/^(?:async )?function ?\w* ?\( ?\) ?(\{[\s\S]*\})$/.exec(text)?.[1] ?? null);
  if (body === null) return text;
  if (!body.startsWith("{")) return body;
  const ret = /^\{ ?return ([\s\S]*?);? ?\}$/.exec(body);
  return ret ? ret[1]!.trim() : text;
}

/**
 * An input as declared (`t.string()`, zod, JSON Schema, `{ shape, desc }`).
 * A Standard Schema says itself whether it may be left out (zod's
 * `.optional()`, `.default(x)`); a plain shape may be left out when it has a
 * `default`, or allows null (`t.optional(...)`: then null). An AI function
 * sends every input, so its optional ones always take a value (null when
 * none is given: the shape then accepts null); a module's `.optional()`
 * input with no default stays left out.
 */
export function declareInput(name: string, spec: FieldSpec, where: string, program: "ai" | "module"): DeclaredInput {
  const { shape: written, desc } = readField(spec, where, name);
  // a default given as a function (`t.string({ default: () => today() })`) is computed at each call that leaves the
  // input out; the interface holds the value it gives now, and the version counts its code
  let read = written;
  let compute: (() => unknown) | undefined;
  let code: string | undefined;
  if (typeof written["default"] === "function") {
    compute = written["default"] as () => unknown;
    code = defaultCode(compute);
    read = { ...written, default: compute() };
  }
  const opaque = isOpaque(spec);
  if (opaque && program === "ai") {
    throw new InterfaceError("interface-malformed", name, `${where}: an AI function's input is sent to a model: it cannot be opaque`);
  }
  const schema = standardOf(spec);
  const said = saysOptional(spec);
  let optional = false;
  let fill: { value: unknown; compute?: () => unknown } | undefined;
  if (said !== undefined) {
    optional = said;
    if (said && Object.hasOwn(read, "default")) fill = { value: copyData(read["default"]) };
  } else if (schema) {
    const probe = schema["~standard"].validate(undefined);
    if (isThenable(probe)) Promise.resolve(probe).catch(() => undefined);     // an async schema cannot say at definition time: required
    else if (!probe.issues) {
      optional = true;
      if (probe.value !== undefined) fill = { value: probe.value };
    }
  } else if (Object.hasOwn(read, "default")) {
    optional = true;
    fill = { value: copyData(read["default"]) };
  } else if (!opaque && allowsNull(read)) {
    optional = true;
    fill = { value: null };
  }
  let shape = read;
  if (optional && !fill && program === "ai") fill = { value: null };
  if (fill && fill.value === null && !allowsNull(shape) && Object.keys(shape).length) {
    const { default: d, ...rest } = shape;
    shape = { anyOf: [rest, { type: "null" }], ...(d !== undefined ? { default: d } : {}) };
  }
  if (fill && !Object.hasOwn(shape, "default")) {
    const json = jsonForm(fill.value);
    if (json === undefined) {
      throw new InterfaceError("interface-malformed", name, `${where}: its default has no JSON form, so no other language can call it with the input left out`);
    }
    shape = { ...shape, default: json };
  }
  if (Object.hasOwn(shape, "default") && !opaque && shape["default"] !== null) {
    // bound as a given value is (programs.md, "Binding"): "5" for an integer is 5; one that does not bind stays, and is refused
    const [ok, bound] = bindValue(shape["default"], { name, shape });
    const json = ok ? jsonForm(bound) : undefined;
    if (json !== undefined) {
      shape = { ...shape, default: json };
      if (fill && !compute) fill = { value: bound };
    }
  }
  if (compute && fill) fill = { value: fill.value, compute };
  const field: InterfaceField = {
    name, shape, ...(desc ? { desc } : {}), ...(opaque ? { opaque: true as const } : {}), ...(optional ? { optional: true as const } : {}),
  };
  return { field, rule: { optional, ...(fill ? { fill } : {}), schema }, ...(code ? { code } : {}) };
}

/**
 * An output as declared: its interface field. An output cannot be
 * `optional` (a program gives every output), and an AI function's cannot be
 * opaque (a model writes JSON): both refuse `interface-malformed`.
 */
export function declareOutput(name: string, spec: FieldSpec, where: string, program: "ai" | "module"): InterfaceField {
  if (saysOptional(spec) === true) {
    throw new InterfaceError("interface-malformed", name, `${where}: an output cannot be optional (a program gives every output)`);
  }
  const opaque = isOpaque(spec);
  if (opaque && program === "ai") {
    throw new InterfaceError("interface-malformed", name, `${where}: an AI function's output is written by a model: it cannot be opaque`);
  }
  const { shape, desc } = readField(spec, where, name);
  return { name, shape, ...(desc ? { desc } : {}), ...(opaque ? { opaque: true as const } : {}) };
}

/**
 * Each input's rule, by name. A `Map`, never an object: a field name is data
 * (`__proto__`, `toString` and `constructor` are ASCII identifiers too), and
 * an object's lookup would find its prototype's members.
 */
export type InputRules = ReadonlyMap<string, InputRule>;

/** The rules of inputs read from an interface (a loaded program): optional as it says, filled with its default. */
export function rulesOf(fields: readonly InterfaceField[]): InputRules {
  return new Map(fields.map((f) => [f.name, {
    optional: f.optional === true,
    ...(Object.hasOwn(f.shape, "default") ? { fill: { value: f.shape["default"] } } : {}),
    schema: null,
  }]));
}

const plainObject = (x: unknown): x is Rec => typeof x === "object" && x !== null && !Array.isArray(x)
  && (Object.getPrototypeOf(x) === Object.prototype || Object.getPrototypeOf(x) === null);

/**
 * Binds a call's argument to inputs by name: the record, or (exactly one
 * required input) that input's value alone. Only the argument's own
 * properties are its inputs: an inherited member (`toString`,
 * `constructor`) is never a value, and an own `__proto__` is one.
 */
export class Binder {
  private readonly required: string[];
  private readonly name: string;
  private readonly names: readonly string[];
  private readonly rules: InputRules;

  constructor(name: string, names: readonly string[], rules: InputRules) {
    this.name = name;
    this.names = names;
    this.rules = rules;
    for (const n of names) if (!rules.has(n)) throw new Error(`${name}: input ${n} has no rule`);
    this.required = names.filter((n) => !this.rule(n).optional);
  }

  private rule(n: string): InputRule {
    return this.rules.get(n)!;
  }

  /** The inputs given by name (`undefined` is left out), and whether each left-out one takes a value. */
  bind(arg: unknown, opts: { fill?: boolean; check?: boolean } = {}): [Rec, Set<string>] {
    const { names, required, name } = this;
    // an object is the inputs by name when every key is an input's, or it holds the one required input's name; else it is that input's value
    const keyed = arg === undefined
      || (plainObject(arg) && (required.length !== 1 || Object.keys(arg).every((k) => names.includes(k)) || Object.hasOwn(arg, required[0]!)));
    if (!keyed) {
      if (required.length === 1) {
        const one: Rec = {};
        setOwn(one, required[0]!, arg);
        return this.bind(one, opts);
      }
      throw new InterfaceError("interface-input", null, `${name} takes its inputs by name: ${name}({ ${names.join(", ")} })`);
    }
    const given = (arg ?? {}) as Rec;
    const out: Rec = {};
    const filled = new Set<string>();
    if (opts.check !== false) {
      const unknown = Object.keys(given).filter((k) => getOwn(given, k) !== undefined && !names.includes(k)).sort(byCodePoint);
      if (unknown.length) throw new InterfaceError("interface-input", unknown[0]!, `${name} has no input ${unknown.join(", ")} (its inputs: ${names.join(", ") || "none"})`);
      const missing = required.filter((k) => getOwn(given, k) === undefined);
      if (missing.length) throw new InterfaceError("interface-input", missing[0]!, `${name} needs ${missing.join(", ")}`);
    }
    for (const n of names) {
      const value = getOwn(given, n);
      if (value !== undefined) setOwn(out, n, value);
      else if (opts.fill !== false && this.rule(n).fill) {
        const fill = this.rule(n).fill!;
        setOwn(out, n, fill.compute ? fill.compute() : copyData(fill.value));
        filled.add(n);
      }
    }
    if (opts.check === false) for (const [k, v] of entriesOf(given)) if (!names.includes(k) && v !== undefined) setOwn(out, k, v);
    return [out, filled];
  }

  /**
   * Given values checked and parsed by their Standard Schemas (defaults,
   * transforms); a filled value is already the schema's. Every validation
   * started is awaited or observed on every path (a refusal from one never
   * leaves another's rejection unhandled); the first refusal, in the
   * interface's order, is the one thrown.
   */
  parse(bound: Rec, skip: Set<string>, sync = false): Promise<void> | void {
    const outcomes: Array<{ n: string; r: StandardResult | Promise<StandardResult> } | { n: string; error: unknown }> = [];
    const pending = new Set<string>();
    for (const n of this.names) {
      const schema = this.rule(n).schema;
      if (!schema || skip.has(n) || !Object.hasOwn(bound, n)) continue;
      try {
        const got: unknown = schema["~standard"].validate(getOwn(bound, n));
        if (isThenable(got)) {
          const r = Promise.resolve(got);                            // this realm's promise, whatever realm the schema's is from
          r.catch(() => undefined);                                  // observed now: whichever way this call goes
          pending.add(n);
          outcomes.push({ n, r });
        } else outcomes.push({ n, r: got as StandardResult });
      } catch (error) {
        outcomes.push({ n, error });
      }
    }
    const accept = (n: string, r: StandardResult) => {
      if (r.issues) throw new InterfaceError("interface-input", n, `${this.name}: input ${n}: ${r.issues.map((i) => i.message).join("; ")}`);
      setOwn(bound, n, r.value);
    };
    const later = pending.size > 0;
    if (later && sync) throw new TypeError(`${this.name}: an input's schema validates asynchronously; render cannot wait for it (predict can)`);
    if (!later) {
      for (const o of outcomes) {
        if ("error" in o) throw o.error;
        accept(o.n, o.r as StandardResult);
      }
      return;
    }
    return (async () => {
      const settled = await Promise.allSettled(outcomes.map((o) => ("error" in o ? Promise.reject(o.error) : Promise.resolve(o.r))));
      settled.forEach((x, i) => {
        if (x.status === "rejected") throw x.reason;
        accept(outcomes[i]!.n, x.value);
      });
    })();
  }
}
