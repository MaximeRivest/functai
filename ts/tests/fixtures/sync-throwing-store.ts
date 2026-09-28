/**
 * A required journal whose store throws before returning a promise, on every
 * append. Run in its own process by failure-paths.test.ts, under a time
 * limit: before the repair this spun the event loop for ever (no timer ever
 * fired). It prints one line of JSON: what the call did, and whether a timer
 * set before it fired.
 */

import { configure, module, t, type EventStore } from "../../src/index.ts";

configure({ logCalls: false });
console.warn = () => undefined;
const store: EventStore = {
  append() {
    throw new Error("synchronous failure");
  },
  async read() {
    return { events: [] };
  },
};
let ran = false;
const f = module("f", { input: {}, output: t.string(), journal: { store, mode: "required", retries: 1, backoff: 5, timeout: 2000 } }, () => {
  ran = true;
  return "ok";
});
let fired = false;
setTimeout(() => { fired = true; }, 10);
const t0 = Date.now();
const out = await f({}).then(
  () => "returned",
  (e: { name?: string; code?: string; journal?: string; outcome?: { failed?: { code?: string } } }) =>
    [e.name, e.code, e.journal ?? "", e.outcome?.failed?.code ?? ""].join(" "),
);
await new Promise((r) => setTimeout(r, 20));
console.log(JSON.stringify({ out, fired, ran, ms: Date.now() - t0 }));
