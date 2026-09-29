/**
 * Saved programs (contract/saved.md). `load` runs an AI function saved in
 * any language, from its folder's `functai.json`, after checking it sends
 * exactly what was saved; it refuses, with a reason, what it cannot run
 * (code of its own, tools, a baked model). `save` writes an AI function
 * defined here as a folder another language's loader reads.
 */

import * as lmcc from "lmcc";
import { Config, stringifyJson } from "@lm15/lm15";
import { builtin } from "./host.ts";
import { fieldsOf, make, type AnyAIFunction as AIFunction } from "./fn.ts";
import type { AnyModule } from "./module.ts";
import { interfaceSignature, malformed, type Interface, type InterfaceField } from "./interface.ts";
import { REGISTRY } from "./layouts.ts";
import { passes } from "./schema.ts";
import { checkSettings, type Settings } from "./settings.ts";
import * as sig from "./signature.ts";
import { VERSION } from "./calllog.ts";
import { copyData, parseData, setOwn, writeData } from "./values.ts";
import * as bridge from "lmcc/lm15";

type Rec = Record<string, unknown>;

export const FORMAT = 1;
const LANGUAGE = "typescript";

/** A saved program this loader will not run; `code` says why (contract/saved.md). */
export class LoadRefused extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(`[${code}] ${message}`);
    this.code = code;
    this.name = "LoadRefused";
  }
}

const refuse = (code: string, message: string): never => {
  throw new LoadRefused(code, message);
};

const SNAKE: Record<string, keyof Settings> = {
  retries: "retries", api_retries: "apiRetries", max_steps: "maxSteps", tool_errors: "toolErrors",
  capabilities: "capabilities", log_content: "logContent",
};

/** A manifest's format, then its form (contract/schema/saved.schema.json, with interface.schema.json for every node's interface). */
function checkForm(m: unknown): Rec {
  if (typeof m !== "object" || m === null || Array.isArray(m)) refuse("saved-malformed", "functai.json is not a JSON object");
  const manifest = m as Rec;
  if (manifest["functai_saved"] !== FORMAT) {
    refuse("saved-format", `functai.json is format ${JSON.stringify(manifest["functai_saved"])}; this loader reads format ${FORMAT}`);
  }
  if (!passes("saved", manifest)) refuse("saved-malformed", "functai.json does not pass the saved manifest's schema (contract/schema/saved.schema.json)");
  return manifest;
}

/** The node asked for (the entry by default). */
function nodeOf(m: Rec, key: string): Rec {
  const node = (m["nodes"] as Rec)[key] as Rec | undefined;
  if (!node) refuse("saved-malformed", `functai.json has no node ${JSON.stringify(key)}`);
  return node!;
}

/** The plain fields of an AI node's signature, as the interface they describe (saved.md: a node written before it had one). */
function signatureInterface(data: Rec): Interface {
  const signature = data["signature"] as { instructions: string; fields: Rec[] };
  const plain = signature.fields.filter((f) => ((f["purpose"] as string | null) ?? "plain") === "plain");
  const field = (f: Rec): InterfaceField => ({
    name: f["name"] as string, shape: f["shape"] as Rec, ...(f["desc"] ? { desc: f["desc"] as string } : {}),
    ...(typeof f["type"] === "string" ? { type: f["type"] as string } : {}),
  });
  return {
    description: signature.instructions,
    inputs: plain.filter((f) => f["direction"] === "input").map(field),
    outputs: plain.filter((f) => f["direction"] === "output").map(field),
  };
}

/** An AI node's interface, checked (programs.md; saved.md step 6): refused when it breaks the rules or promises other data than its signature. */
function checkedInterface(key: string, node: Rec): Interface | null {
  if (!("interface" in node)) return null;
  const fault = malformed(node["interface"], { ai: node["kind"] === "ai" });
  if (fault) refuse("interface-malformed", `${key}: its interface is refused${fault.field ? ` at field ${fault.field}` : ""}: ${fault.why}`);
  const iface = node["interface"] as Interface;
  if (node["kind"] === "ai" && interfaceSignature(iface) !== interfaceSignature(signatureInterface(node["ai"] as Rec))) {
    refuse("saved-differs", `${key}: its interface describes other data than its signature takes and gives`);
  }
  return iface;
}

/**
 * What a saved node takes and gives, without loading or running anything
 * (contract/saved.md, "Describing without loading"): its interface, checked;
 * for an AI node written before nodes had one, the interface its signature
 * gives (its instruction as the description: do not show that to outside
 * callers as it is). Refuses (`LoadRefused`) as saved.md says.
 */
export function describeSaved(from: string | Rec, opts: { node?: string } = {}): Interface {
  const m = checkForm(typeof from === "string" ? readManifest(from)[0] : from);
  const key = opts.node ?? (m["entry"] as string);
  const node = nodeOf(m, key);
  if (node["kind"] !== "ai" && node["kind"] !== "module") refuse("saved-not-ai", `${key} is a ${node["kind"]}: plain code has no interface`);
  const iface = checkedInterface(key, node);
  if (iface) return copyData(iface);
  if (node["kind"] === "module") refuse("saved-no-interface", `${key} is a module saved before nodes had an interface: what it takes and gives is not known`);
  return signatureInterface(node["ai"] as Rec);
}

/**
 * An AI function from a saved manifest (the parsed `functai.json`). `node`
 * names one by key (`module:name`); the default is the entry. Parse it with
 * `lmcc.parseJson`, which keeps each object's members in the order written:
 * `JSON.parse` lists integer-like names (`"10"`) first, so a value holding
 * one would be sent in another order than was saved.
 */
export function fromManifest(manifest: unknown, opts: { node?: string; savedId?: string } = {}): AIFunction {
  const m = checkForm(manifest);
  const language = (m["language"] as string | undefined) ?? "python";
  const key = opts.node ?? (m["entry"] as string);
  const node = nodeOf(m, key);
  if (node!["kind"] !== "ai") {
    refuse("saved-not-ai", `${key} is ${node!["kind"] === "module" ? "a module" : `a ${node!["kind"]}`}: code in ${language}, which this loader cannot run. Its AI functions load by key.`);
  }
  const data = node!["ai"] as Rec;
  if (!data || typeof data !== "object") refuse("saved-malformed", `${key} has no "ai" entry`);
  if (!("body" in data) || data["body"] !== null) {
    refuse("saved-code", `${key} runs code of its own beside the model (written in ${language}); only ${language} can run it`);
  }
  if (Array.isArray(data["tools"]) && data["tools"].length) {
    refuse("saved-tools", `${key} has tools (${(data["tools"] as unknown[]).map((t) => JSON.stringify(t)).join(", ")}): a tool is code`);
  }
  const settingsIn = (data["settings"] ?? {}) as Rec;
  for (const [k, v] of Object.entries(settingsIn)) {
    if (typeof v === "object" && v !== null && ("baked" in v || "node" in v)) {
      refuse("saved-model", `${key}: setting ${k} is ${JSON.stringify(v)}, not something this loader can reach`);
    }
  }
  const signature = lmcc.signatureFromDict(data["signature"] as Record<string, unknown>);
  const inputs: sig.FieldDef[] = [];
  const outputs: sig.FieldDef[] = [];
  let cot = false;
  // each field's type as saved: the signature's fingerprint names it, and a recorded turn is replayed only under its own signature
  const types: Record<string, string> = {};
  for (const f of signature.fields) {
    if (f.type) setOwn(types, f.name, f.type);
    if (f.direction === "input" && f.purpose === "plain") inputs.push({ name: f.name, shape: f.shape as Rec, desc: f.desc ?? null });
    else if (f.direction === "output" && f.purpose === "plain") outputs.push({ name: f.name, shape: f.shape as Rec });
    else if (f.purpose === "reasoning") cot = true;
    else refuse("saved-tools", `${key}: field ${f.name} (${f.purpose}) needs tools`);
  }
  const own: Settings = {};
  if (typeof settingsIn["lm"] === "string") own.lm = settingsIn["lm"] as string;
  if (settingsIn["module"] === "cot" || cot) own.module = "cot";
  if (settingsIn["include_fn_name_in_instructions"] === false) own.includeFnName = false;
  if (settingsIn["adapter"] !== undefined && settingsIn["adapter"] !== null) own.adapter = settingsIn["adapter"];
  for (const [snake, camel] of Object.entries(SNAKE)) if (settingsIn[snake] !== undefined && settingsIn[snake] !== null) (own as Rec)[camel] = settingsIn[snake];
  if (Array.isArray(data["template"])) own.template = data["template"] as Rec[];
  const config = data["config"] && Object.keys(data["config"] as Rec).length ? Config.fromJSON(bridge.toLm15(data["config"]) as never) : undefined;
  if (config) {
    const { temperature, maxTokens, topP, stop, seed, ...rest } = config as Rec;
    if (temperature !== undefined) own.temperature = temperature as number;
    if (maxTokens !== undefined) own.maxTokens = maxTokens as number;
    if (topP !== undefined) own.topP = topP as number;
    if (stop !== undefined) own.stop = stop as string[];
    if (seed !== undefined) own.seed = seed as number;
    if (Object.keys(rest).length) own.config = rest;
  }
  const state = (data["state"] ?? { instructions: null, demos: [] }) as { instructions: string | null; demos: Rec[] };
  // its optional inputs, and the default each is sent with, come from its interface (none without one)
  const declared = "interface" in node ? node["interface"] as Interface : null;
  const withDefaults = inputs.map((f) => {
    const d = declared?.inputs.find((x) => x.name === f.name);
    return d?.optional ? { ...f, shape: { ...f.shape, ...(Object.hasOwn(d.shape, "default") ? { default: d.shape["default"] } : {}) }, optional: true } : f;
  });
  const definition: sig.Definition & { written: string } = {
    name: node!["name"] as string, description: "", inputs: withDefaults, outputs, cot, tools: false,
    includeName: own.includeFnName !== false, written: signature.instructions, types,
  };
  const fallback = signatureInterface(data);
  const iface: Interface = declared && !malformed(declared, { ai: true }) ? declared : fallback;
  const fn = make({
    definition, interface: iface, own, tools: [], module: node!["module"] as string, saved: opts.savedId,
    state: { instructions: state.instructions ?? null, demos: [] },
  });
  // its own policy, as a definition's is checked: a misspelt field would write the very value it keeps out (calls.md, "Content")
  checkSettings(own, `${key} (loaded)`, fieldsOf(iface, fn.signature));
  fn.demos = (state.demos ?? []) as never;
  // it must send what was saved (contract/saved.md, "Loading", step 6)
  const probes = (data["probes"] ?? []) as Rec[];
  const want = ((data["fingerprints"] as Rec | undefined)?.["requests"] ?? []) as string[];
  probes.forEach((probe, i) => {
    let got: string;
    try {
      got = lmcc.sha256(fn.probeRequest(probe));
    } catch (err) {
      if (!lmcc.isRefusal(err)) throw err;
      got = `refused:${(err as lmcc.Refusal).code}`;
    }
    if (want[i] !== undefined && got !== want[i]) {
      refuse("saved-differs", `${key}: for probe ${i} (${JSON.stringify(probe).slice(0, 200)}) it would send ${got}, but ${want[i]} was saved`);
    }
  });
  if (typeof data["version"] === "string" && data["version"] !== fn.version) {
    refuse("saved-differs", `${key}: its version here is ${fn.version}, but ${data["version"]} was saved`);
  }
  checkedInterface(key, node);
  return fn;
}

/** A folder's (or a functai.json's) manifest, parsed, and its text. */
function readManifest(path: string): [unknown, string] {
  const fs = builtin("node:fs");
  const p = builtin("node:path");
  if (!fs || !p) throw new Error("reading a saved folder needs a file system; parse functai.json yourself and pass the manifest");
  const file = fs.statSync(path).isDirectory() ? p.join(path, "functai.json") : path;
  const text = fs.readFileSync(file, "utf8");
  try {
    return [parseData(text), text];              // members in the order written (JSON.parse lists integer-like names first)
  } catch (err) {
    return refuse("saved-malformed", `${file}: ${(err as Error).message}`);
  }
}

/** An AI function from a saved folder (or its functai.json): Node, Deno, Bun. */
export function load(path: string, opts: { node?: string } = {}): AIFunction {
  const [manifest, text] = readManifest(path);
  return fromManifest(manifest, { node: opts.node, savedId: "sha256:" + lmcc.sha256Hex(text) });
}

/** An AI function's `ai` entry (contract/saved.md). */
function aiEntry(fn: AIFunction): Rec {
  const s = fn.settings;
  const settings: Rec = { module: s.module ?? "predict", include_fn_name_in_instructions: s.includeFnName !== false };
  if (typeof s.lm === "string") settings["lm"] = s.lm;
  if (s.adapter !== undefined && s.adapter !== null) {
    settings["adapter"] = s.adapter instanceof lmcc.Adapter ? lmcc.dump(s.adapter, REGISTRY) : s.adapter;
  }
  for (const [snake, camel] of Object.entries(SNAKE)) if (s[camel] !== undefined && s[camel] !== null) settings[snake] = s[camel];
  if (fn.tools.length) throw new LoadRefused("saved-tools", `${fn.name} has tools: a tool is code, and a saved folder carries none from TypeScript yet`);
  const config: Rec = { ...(s.config ?? {}) };
  for (const k of ["temperature", "maxTokens", "topP", "stop", "seed"] as const) if (s[k] !== undefined && s[k] !== null) config[k] = s[k];
  const state = fn.state();
  const sample = sig.sampleInputs(fn.signature);
  const probes: Rec[] = [sample];
  for (const d of state.demos.slice(0, 3)) {
    const inputs = (d as { inputs: Rec }).inputs;
    if (!probes.some((p) => lmcc.jsonEqual(p as lmcc.Json, inputs as lmcc.Json))) probes.push(inputs);
  }
  const requests = probes.map((p) => {
    try {
      return lmcc.sha256(fn.probeRequest(p));
    } catch (err) {
      if (!lmcc.isRefusal(err)) throw err;
      return `refused:${(err as lmcc.Refusal).code}`;
    }
  });
  return {
    settings, config: Object.keys(config).length ? JSON.parse(stringifyJson(Config.toJSON(config as Config))) : {},
    template: s.template ? [...s.template] : null, tools: [], teacher: null, state, requires: [],
    signature: lmcc.signatureToDict(fn.signature), probes,
    fingerprints: { signature: lmcc.signatureFingerprint(fn.signature), requests },
    body: null, version: fn.version,
  };
}

const isModule = (p: AIFunction | AnyModule): p is AnyModule => Array.isArray((p as AnyModule).uses) && !("signatureId" in p);

/**
 * The manifest of a program defined here: the part every language reads
 * (contract/saved.md). An AI function is its node; a module is its node
 * (with its interface, so any language can describe it; its code runs only
 * here) and a node for each AI function and module it uses, which other
 * languages load by key.
 */
export function toManifest(program: AIFunction | AnyModule): Rec {
  const nodes: Rec = {};
  const add = (p: AIFunction | AnyModule) => {
    const key = `${p.module}:${p.name}`;
    if (key in nodes) return;
    if (isModule(p)) {
      nodes[key] = { kind: "module", module: p.module, name: p.name, interface: p.interface };
      for (const u of p.uses) add(u);
    } else {
      nodes[key] = { kind: "ai", module: p.module, name: p.name, interface: p.interface, ai: aiEntry(p) };
    }
  };
  add(program);
  return {
    functai_saved: FORMAT, language: LANGUAGE, entry: `${program.module}:${program.name}`,
    created: new Date().toISOString().replace(/\.\d+Z$/, "+00:00"), functai: VERSION, nodes,
  };
}

/** Write a program (an AI function, or a module and what it uses) to a folder (its functai.json), for any language's loader. */
export function save(fn: AIFunction | AnyModule, folder: string): string {
  const fs = builtin("node:fs");
  const p = builtin("node:path");
  if (!fs || !p) throw new Error("save needs a file system; use toManifest and write it yourself");
  fs.mkdirSync(folder, { recursive: true });
  const file = p.join(folder, "functai.json");
  fs.writeFileSync(file, writeData(toManifest(fn), 1) + "\n");     // every value's members in its order, as Python writes them
  return file;
}
