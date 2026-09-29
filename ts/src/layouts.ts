/**
 * Layouts: how a call is written as messages and how the reply is read back
 * (contract/functions.md, "The layout"). A layout is an lmcc adapter; the
 * three named ones are the contract's artifacts, carried as data.
 */

import * as lmcc from "lmcc";
import { install } from "lmcc/std";
import { LAYOUTS, MODELS } from "./generated/contract.ts";
import { JUDGMENT_ONLY } from "./models.ts";
import { copyData } from "./values.ts";

/** The registry every plan binds with: lmcc's standard pack and functai's reply reader. */
export const REGISTRY = new lmcc.Registry();
install(REGISTRY);

/** The whole reply is the one output's value: a template with no output pattern. */
class ReplyReader extends lmcc.Reader {
  constructor(spec: Record<string, unknown>) {
    super();
    if (Object.keys(spec).some((k) => k !== "kind")) {
      lmcc.refuse("entry-malformed", "reader: functai_reply takes only 'kind'", { fix: { action: "edit-entry", path: "reader" } });
    }
    this.spec = { ...spec };
  }
  split(text: string, fieldNames: string[]): Record<string, string> {
    if (fieldNames.length !== 1) {
      lmcc.refuse("parse-ambiguous", `the template has no output pattern, so the reply can hold one output, not ${fieldNames.length}: ${JSON.stringify(fieldNames)}`);
    }
    return { [fieldNames[0]!]: lmcc.strip(text) };
  }
  join(spelled: [string, string][]): string {
    return spelled.map(([, text]) => text).join("\n");
  }
}
REGISTRY.registerReader("functai_reply", (spec) => new ReplyReader(spec), { version: "1.0.0", existOk: true });

const NAMED: Record<string, "xml" | "chat" | "json"> = {
  xml: "xml", default: "xml", tags: "xml", chat: "chat", chatadapter: "chat", json: "json", jsonadapter: "json",
};

const loaded = new Map<string, lmcc.Adapter>();

/** A named layout (`"xml"`, `"chat"`, `"json"`, …), an lmcc adapter, or an adapter artifact. */
export function resolveAdapter(adapter: unknown): lmcc.Adapter {
  if (adapter instanceof lmcc.Adapter) return adapter;
  if (adapter === null || adapter === undefined) return resolveAdapter("xml");
  if (typeof adapter === "string") {
    const key = adapter.toLowerCase().replaceAll("-", "").replaceAll(" ", "").replaceAll("_", "");
    const name = Object.hasOwn(NAMED, key) ? NAMED[key] : undefined;
    if (!name) throw new Error(`unknown adapter ${JSON.stringify(adapter)}; use "xml", "chat", "json", an lmcc adapter, or template: [...]`);
    let a = loaded.get(name);
    if (!a) {
      a = lmcc.load(copyData(LAYOUTS[name]), { registry: REGISTRY });
      loaded.set(name, a);
    }
    return a;
  }
  if (typeof adapter === "object") return lmcc.load(copyData(adapter) as Record<string, unknown>, { registry: REGISTRY });
  throw new TypeError(`adapter must be "xml", "chat", "json", an lmcc adapter or an artifact, not ${typeof adapter}`);
}

const XML = LAYOUTS.xml as { transports: Record<string, unknown>; formats: Record<string, unknown> };

/** A layout from a template the function wrote: functai's formats and transports. */
export function templateAdapter(messages: readonly Record<string, unknown>[], reader = "derived"): lmcc.Adapter {
  const msgs = messages.map((m, i) => {
    if ("directive" in m || ("role" in m && "text" in m)) return { ...m };
    if ("role" in m && typeof m["content"] === "string") return { role: m["role"], text: m["content"] };
    throw new TypeError(`template[${i}]: expected system(...), user(...), assistant(...), turns(), or {role, content}`);
  });
  if (!msgs.some((m) => "directive" in m)) {
    let last = -1;
    msgs.forEach((m, i) => { if (m["role"] === "user") last = i; });
    msgs.splice(last < 0 ? msgs.length : last, 0, { directive: "turns" });
  }
  return lmcc.adapter({
    name: "functai_template", messages: msgs, formats: copyData(XML.formats),
    transports: copyData(XML.transports), reader: { kind: reader },
  });
}

const INPUT_TAGS = "{% for f in inputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}";

function judgment(signature: lmcc.Signature): [lmcc.Adapter, lmcc.Signature] {
  const data = lmcc.signatureToDict(signature) as { instructions: string; fields: Record<string, unknown>[] };
  const inputs = data.fields.filter((f) => f["direction"] === "input" && (f["purpose"] ?? "plain") === "plain");
  const outputs = data.fields.filter((f) => f["direction"] === "output");
  const doc = data.instructions;
  const fields = data.fields.map((f) => {
    if (f["direction"] !== "output") return f;
    const desc = (f["desc"] as string | null) || (doc ? (outputs.length === 1 ? doc : `${doc} (${f["name"]})`) : null);
    return { ...f, desc };
  });
  const body = inputs.length === 1 ? `{${inputs[0]!["name"]}}` : INPUT_TAGS;
  const adapter = lmcc.adapter({
    name: "functai_judgment", messages: [{ directive: "turns" }, { role: "user", text: body }],
    reader: { kind: "json_object" }, formats: copyData(XML.formats),
  });
  return [adapter, lmcc.signatureFromDict({ instructions: doc, fields })];
}

export interface Layout {
  adapter?: unknown;
  template?: readonly Record<string, unknown>[] | null;
}

/** Bind a layout to a signature for a model's facts; every refusal fires here, before anything is sent. */
export function bind(layout: Layout, signature: lmcc.Signature, caps: Record<string, unknown>, provider: string): lmcc.Plan {
  if (layout.template) {
    try {
      return templateAdapter(layout.template).bind(signature, caps, { registry: REGISTRY });
    } catch (err) {
      const r = err as lmcc.Refusal;
      if (!lmcc.isRefusal(err) || r.code !== "not-readable" || (r.fix as { path?: string } | null)?.path !== "template") throw err;
    }
    const plan = templateAdapter(layout.template, "functai_reply").bind(signature, caps, { registry: REGISTRY });
    const visible = signature.fields.filter((f) => f.direction === "output" && f.purpose === "plain");
    if (visible.length !== 1) {
      throw new lmcc.Refusal("not-readable",
        `the template has no output pattern, so the reply can only be one output, but this function has ${visible.length}: ${JSON.stringify(visible.map((f) => f.name))}. Add the pattern to the template`,
        { fix: { action: "edit-template", path: "template" } });
    }
    return plan;
  }
  if ((layout.adapter === undefined || layout.adapter === null) && JUDGMENT_ONLY.has(provider)) {
    const [adapter, sig] = judgment(signature);
    return adapter.bind(sig, caps, { registry: REGISTRY });
  }
  return resolveAdapter(layout.adapter).bind(signature, caps, { registry: REGISTRY });
}

export { MODELS };
