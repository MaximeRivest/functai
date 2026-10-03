/**
 * The stage-1 contract cases TypeScript passes (contract/README.md, "Which
 * cases each language passes"): programs/, content/, saw/ (read). Each case
 * is read from its file and run through the implementation, as
 * contract/cases/README.md says.
 */

import assert from "node:assert/strict";
import { shownTurn } from "../src/saw.ts";
import { test } from "node:test";
import {
  ai, checkInterface, configure, InterfaceError, keepsSaw, module, sawOf, SawUnknown, SettingError, t,
  type AnyModule, type Interface, type StreamEvent,
} from "../src/index.ts";
import { checkLogContent, keptFields, writtenRecord } from "../src/content.ts";
import { cases, described, native, tsFunction, type Rec } from "./cases.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

/** A router that answers nothing: a call made with it fails after its `started` (enough to see what it was called with). */
const silent = {
  resolve: (m: string) => ({ provider: "openai", model: m }),
  complete: async () => { throw new Error("no model here"); },
};

// ------------------------------------------------------------------ programs

/** A module declared with an interface, the way a TypeScript user writes it: each field `{ shape, desc?, optional, opaque? }`. */
function tsModule(iface: Rec, run: (inputs: Rec) => unknown): AnyModule {
  const field = (f: Rec, input: boolean) => ({
    shape: f.opaque ? t.opaque() : f.shape, ...(f.desc ? { desc: f.desc } : {}), ...(input ? { optional: Boolean(f.optional) } : {}),
  });
  const outputs = Object.fromEntries(iface.outputs.map((f: Rec) => [f.name, field(f, false)]));
  return module("program", {
    description: iface.description,
    input: Object.fromEntries(iface.inputs.map((f: Rec) => [f.name, field(f, true)])),
    ...(iface.outputs.length === 1 && iface.outputs[0].name === "result" ? { output: outputs["result"] } : { outputs }),
  } as never, run as never) as AnyModule;
}

/** The interface without the words a TypeScript declaration does not carry (`type`: the host's name for the type, for people). */
const withoutTypes = (iface: Rec): Interface => ({
  ...iface, inputs: iface.inputs.map(({ type: _t, ...f }: Rec) => f), outputs: iface.outputs.map(({ type: _t, ...f }: Rec) => f),
} as Interface);

class Stop extends Error {}

/** What a module's code gets for these inputs, or its refusal. */
async function inputsCheck(m: AnyModule, got: { inputs?: Rec }, inputs: Rec): Promise<Rec> {
  try {
    await m(Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, native(v)])));
  } catch (err) {
    if (err instanceof InterfaceError) return { refuses: err.code, field: err.field };
    if (!(err instanceof Stop)) throw err;
  }
  return { inputs: Object.fromEntries(Object.entries(got.inputs!).map(([k, v]) => [k, described(v)])) };
}

for (const [name, c] of cases("programs")) {
  test(`programs case ${name}`, async () => {
    if (c.program === "ai") {
      if (c.expect.refuses) {
        assert.throws(() => tsFunction(c.definition), (err: unknown) =>
          err instanceof InterfaceError && err.code === c.expect.refuses && err.field === c.expect.field);
        return;
      }
      const fn = tsFunction(c.definition);
      assert.deepEqual(fn.interface, c.expect.interface);
      assert.equal(fn.interfaceId, c.expect.signature);
      assert.equal(fn.signatureId, c.expect.signature_id);
      for (const b of c.binds) {
        const given = Object.fromEntries(Object.entries(b.inputs as Rec).map(([k, v]) => [k, native(v)]));
        if (b.expect.refuses) {
          // refused before any request (the silent router would say "no model here"), with the input named
          await assert.rejects(fn.using({ router: silent as never, retries: 0, apiRetries: 0 })(given as never),
            (err: unknown) => err instanceof InterfaceError && err.code === b.expect.refuses && err.field === b.expect.field);
          continue;
        }
        const s = fn.using({ router: silent as never, retries: 0, apiRetries: 0 }).stream(given as never);
        const events: StreamEvent[] = [];
        for await (const e of s.events()) events.push(e);
        await assert.rejects(s.result, /no model here/);
        assert.deepEqual({ inputs: (events[0] as Rec).inputs }, b.expect);
      }
      return;
    }
    if (c.program === "module") {
      const got: { inputs?: Rec } = {};
      const m = tsModule(c.interface, (inputs) => { got.inputs = inputs; throw new Stop(); });
      assert.deepEqual(m.interface, withoutTypes(c.interface));
      assert.equal(m.interfaceId, c.expect.signature);
      for (const check of c.checks) {
        if ("inputs" in check) {
          delete got.inputs;
          assert.deepEqual(await inputsCheck(m, got, check.inputs), check.expect, JSON.stringify(check.inputs));
          continue;
        }
        const r = tsModule(c.interface, () => native(check.returned) as never);
        const sample = Object.fromEntries(c.interface.inputs.filter((f: Rec) => !f.optional)
          .map((f: Rec) => [f.name, f.opaque ? new Map() : f.shape.type === "object"
            ? (f.shape.properties ? { id: 1, name: "a", children: [] } : {}) : "x"]));
        let result: Rec;
        try {
          const value = await r(sample);
          const outputs = c.interface.outputs.length === 1 ? { [c.interface.outputs[0].name]: value } : value as Rec;
          result = { outputs: Object.fromEntries(Object.entries(outputs).map(([k, v]) => [k, described(v)])) };
        } catch (err) {
          if (!(err instanceof InterfaceError)) throw err;
          result = { refuses: err.code, field: err.field };
        }
        assert.deepEqual(result, check.expect, JSON.stringify(check.returned));
      }
      return;
    }
    if (c.program === "definitions") {
      for (const x of c.interfaces) {
        let got: Rec;
        try {
          got = { signature: (await import("../src/index.ts")).interfaceSignature(checkInterface(x.interface, { ai: x.ai ?? false })) };
        } catch (err) {
          if (!(err instanceof InterfaceError)) throw err;
          got = { refuses: err.code, field: err.field };
        }
        assert.deepEqual(got, x.expect, JSON.stringify(x.interface));
        if (!x.ai && !got.refuses) assert.equal(tsModule(x.interface, () => "").interfaceId, x.expect.signature);   // defining it gives the same
      }
      return;
    }
    if (c.program === "message") {
      const { toJson } = await import("../src/values.ts");
      for (const check of c.checks) {
        const m = module("program", {
          description: c.interface.description,
          input: Object.fromEntries(c.interface.inputs.map((f: Rec) => [f.name, { shape: f.shape }])),
          output: c.interface.outputs[0].shape,
          ...(check.log_content !== undefined ? { logContent: check.log_content } : {}),
        } as never, (() => "ok") as never) as AnyModule;
        const given = Object.fromEntries(Object.entries(check.inputs as Rec).map(([k, v]) => [k, native(v)]));
        let message = "";
        await assert.rejects(m(given), (err: unknown) => {
          assert.ok(err instanceof InterfaceError && err.code === check.expect.refuses && err.field === check.expect.field);
          message = (err as Error).message;
          return true;
        });
        const value = check.inputs[check.expect.field];
        if (check.expect.quotes !== null) {
          // a stand-in is quoted by the description this language writes for the native value
          const quote = typeof value === "object" && value !== null && "$repr" in value
            ? String((toJson(given[check.expect.field])[0] as Rec)["$repr"]) : check.expect.quotes;
          assert.ok(message.includes(quote), `${message} quotes ${quote}`);
        } else {
          for (const part of [JSON.stringify(value), String(value)]) assert.ok(!message.includes(part), `${message} holds ${part}`);
        }
      }
      return;
    }
    assert.equal(c.program, "same-data");
    const programs = c.interfaces.map((iface: Rec) => {
      const got: { inputs?: Rec } = {};
      return [tsModule(iface, (inputs) => { got.inputs = inputs; throw new Stop(); }), got] as const;
    });
    assert.deepEqual(programs.map(([m]: readonly [AnyModule, Rec]) => m.interfaceId), c.expect.signatures);
    for (const check of c.checks) {
      const results = [];
      for (const [m, got] of programs) results.push(await inputsCheck(m, got, check.inputs));
      assert.deepEqual(results, check.expect, JSON.stringify(check.inputs));
    }
  });
}

test("an interface a module cannot say is refused when it is defined", () => {
  assert.throws(() => module("m", { input: { code: t.string({ pattern: "^B-[0-9]+$" }) }, output: t.string() }, () => ""),
    (err: unknown) => err instanceof InterfaceError && err.code === "interface-malformed" && err.field === "code");
  assert.throws(() => module("m", { input: { message: t.string() }, outputs: { message: t.string() } }, () => ""),
    (err: unknown) => err instanceof InterfaceError && err.field === "message");
  assert.throws(() => ai("f", { input: { frame: t.opaque() } }), (err: unknown) => err instanceof InterfaceError && err.field === "frame");
});

// ------------------------------------------------------------------ content

for (const [name, c] of cases("content")) {
  test(`content case ${name}`, () => {
    const layers = c.layers as { where: string; log_content: unknown }[];
    const refuse = () => {
      for (const l of layers) checkLogContent(l.log_content, l.where, l.where === "own" ? c.fields : undefined);
    };
    if (c.expect.refuses) {
      assert.throws(refuse, (err: unknown) => err instanceof SettingError && err.code === c.expect.refuses && err.field === c.expect.field);
      return;
    }
    refuse();
    const keep = keptFields(c.fields, layers.map((l) => l.log_content as never), c.environment);
    assert.deepEqual(writtenRecord(c.record, c.fields, keep), c.expect.record);
  });
}

// ------------------------------------------------------------------ saw

for (const [name, c] of cases("saw")) {
  if (c.kind === "shown") {
    test(`saw case ${name}`, () => assert.deepEqual(shownTurn(c.turn, c.entry), c.expect));
    continue;
  }
  test(`saw case ${name}`, () => {
    for (const q of c.queries) {
      let got: Rec;
      try {
        got = { saw: sawOf(c.records, q.call) };
      } catch (err) {
        if (!(err instanceof SawUnknown)) throw err;
        got = { unknown: err.code, call: err.call };
      }
      assert.deepEqual(got, q.expect, q.call);
      assert.deepEqual(keepsSaw(c.records, q.call), q.keeps, q.call);
    }
  });
}
