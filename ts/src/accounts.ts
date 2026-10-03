/**
 * Sign in once: subscriptions (Claude, ChatGPT, GitHub Copilot, xAI, Kimi
 * Code), OpenRouter, or an API key for any provider.
 *
 * ```ts
 * await login("claude");                     // sign in; saved for every later session
 * configure({ lm: "claude:claude-sonnet-4-5" });
 * await logins();                            // everything you can use, and how
 * await logout("claude");
 * ```
 *
 * Logins are lm15's managed logins, saved in lm15's one credentials file
 * (`~/.config/lm15/credentials.json`, or `$LM15_CREDENTIALS_PATH`), which
 * lm15 in every language, and functai in every language, share: a login made
 * in Python is used here, and the other way round. A call uses, first match
 * wins: the `router` setting; the saved login for the model's provider; the
 * environment's keys and the Claude Code and Codex CLI logins on this machine.
 * A saved login that expired or was signed out is an error that says how to
 * sign in again: it never falls back to a metered key behind your back.
 */

import { Auth, LMRouter, TerminalUI } from "@lm15/lm15";
import { builtin, env, platform } from "./host.ts";

/** Friendly names, for `login("…")` and model strings (`"claude:claude-sonnet-4-5"`). */
const ALIASES: Record<string, string> = {
  claude: "claude-code", chatgpt: "openai-codex", codex: "openai-codex", copilot: "github-copilot", github: "github-copilot",
  kimi: "kimi-code", grok: "xai",
};

/** The accounts people sign in to (not API keys), in the order a picker shows them. */
const ACCOUNTS: readonly [string, string][] = [
  ["claude-code", "Claude (Pro/Max subscription)"], ["openai-codex", "ChatGPT (Plus/Pro subscription)"], ["github-copilot", "GitHub Copilot"],
  ["xai", "xAI / Grok (subscription)"], ["kimi-code", "Kimi Code (subscription)"],
  ["openrouter", "OpenRouter (approve in the browser; spends OpenRouter credits)"],
];

/** A model to try after signing in. */
const EXAMPLE_MODEL: Record<string, string> = {
  "claude-code": "claude:claude-sonnet-4-5", "openai-codex": "chatgpt:gpt-5.5", "github-copilot": "copilot:gpt-4.1", xai: "grok-4",
  openrouter: "openrouter:openai/gpt-4.1-mini", openai: "gpt-4.1-mini", anthropic: "claude-haiku-4-5", gemini: "gemini-2.5-flash",
  groq: "groq:openai/gpt-oss-120b",
};

const ENV_KEYS: readonly [string, string][] = [
  ["openai", "OPENAI_API_KEY"], ["anthropic", "ANTHROPIC_API_KEY"], ["gemini", "GEMINI_API_KEY"], ["gemini", "GOOGLE_API_KEY"],
  ["groq", "GROQ_API_KEY"], ["openrouter", "OPENROUTER_API_KEY"], ["mistral", "MISTRAL_API_KEY"], ["deepseek", "DEEPSEEK_API_KEY"],
  ["xai", "XAI_API_KEY"], ["together", "TOGETHER_API_KEY"], ["fireworks", "FIREWORKS_API_KEY"], ["cerebras", "CEREBRAS_API_KEY"],
];

/** `"claude"` → `"claude-code"`; lm15's names pass through. */
export function canonical(provider: string): string {
  return ALIASES[provider.toLowerCase()] ?? provider;
}

const label = (provider: string) => ACCOUNTS.find(([p]) => p === provider)?.[1] ?? provider;

let shared: Auth | null | undefined;

/** lm15's credentials store on this machine, or null where there is none (a browser, a file system without a home). */
export function localAuth(): Auth | null {
  if (shared !== undefined) return shared;
  try {
    shared = builtin("node:fs") ? Auth.local() : null;
  } catch {
    shared = null;
  }
  return shared;
}

/** Something functai can use now: a login, a saved key, or a key in the environment, and a model to try with it. */
export interface Login {
  readonly provider: string;
  readonly label: string;
  /** Where it comes from: `saved login`, `saved key`, `Claude Code CLI`, `environment ($OPENAI_API_KEY)`, … */
  readonly source: string;
  /** `ready`, `renewal due`, `needs login`, `found`, … */
  readonly status: string;
  readonly expires: string | null;
  readonly model: string | null;
}

/** Everything functai can use right now: saved logins and keys, and API keys in the environment. Reads files and the environment only. */
export async function logins(): Promise<Login[]> {
  const out: Login[] = [];
  const seen = new Set<string>();
  const a = localAuth();
  if (a) {
    for (const c of await a.connections()) {
      const st = await a.status(c.provider);
      let source = c.kind === "api_key" ? "saved key" : "saved login";
      if (c.methodId === "env") source = "saved: use env key";
      else if (c.methodId.startsWith("external:")) source = "saved: use CLI login";
      out.push({ provider: c.provider, label: label(c.provider), source, status: st.usability.replaceAll("_", " "),
        expires: st.expiresAt === "never" || st.expiresAt === "unknown" ? null : st.expiresAt, model: EXAMPLE_MODEL[c.provider] ?? null });
      seen.add(c.provider);
    }
  }
  const e = env();
  const keys = new Set<string>();
  for (const [provider, key] of ENV_KEYS) {
    if (seen.has(provider) || keys.has(provider) || !e[key]) continue;
    keys.add(provider);
    out.push({ provider, label: provider, source: `environment ($${key})`, status: "found", expires: null, model: EXAMPLE_MODEL[provider] ?? null });
  }
  return out;
}

/**
 * Sign in to a provider once; every later session, in any language, uses it.
 * For subscriptions a browser opens when this machine has a screen
 * (`openBrowser`), else a link or a device code is printed; a Claude Code or
 * Codex CLI already signed in on this machine is used as is. For API
 * providers, give the `key` to save. Already signed in: says so and does
 * nothing (`again: true` signs in again).
 */
export async function login(provider: string, opts: { key?: string; method?: string; again?: boolean; openBrowser?: boolean } = {}): Promise<Login> {
  const a = localAuth();
  if (!a) throw new Error("login needs lm15's credentials file (Node); in a browser, give the router its keys");
  const asked = provider;
  provider = canonical(provider);
  try {
    a.descriptor(provider);
  } catch {
    const known = [...new Set([...Object.keys(ALIASES), ...a.providers().filter((d) => d.methods.length).map((d) => d.id)])].sort();
    throw new TypeError(`unknown provider ${JSON.stringify(provider)}; one of: ${known.join(", ")}`);
  }
  const say = (text: string) => console.error(text);
  const current = await a.status(provider);
  const replace = current.connection?.id;
  const found = async () => (await logins()).find((x) => x.provider === provider)!;
  if (replace && !opts.again && opts.key === undefined && opts.method === undefined && current.ready) {
    const record = await found();
    say(`Already signed in to ${record.label} (${record.source}). Use login(${JSON.stringify(asked)}, { again: true }) to sign in again.`);
    return record;
  }
  const hint = EXAMPLE_MODEL[provider] ? ` Try: configure({ lm: ${JSON.stringify(EXAMPLE_MODEL[provider])} })` : "";
  if (opts.key !== undefined) {
    await a.setApiKey(provider, opts.key, replace ? { replace } : {});
    say(`Saved the ${provider} API key.${hint}`);
    return found();
  }
  const methods = a.methods(provider);
  let method = opts.method;
  if (!method) {
    const external = methods.find((m) => m.id.startsWith("external:"));
    const cli = external && (await a.status(provider)).connection === null ? external : undefined;
    method = cli?.id;
  }
  if (method?.startsWith("external:")) {
    await a.configure(provider, { method, ...(replace ? { replace } : {}) });
    say(`Using your CLI login for ${label(provider)} (read in place, renewed by lm15).${hint}`);
    return found();
  }
  const tty = (globalThis as { process?: { stdin?: { isTTY?: boolean }; stderr?: { isTTY?: boolean } } }).process;
  if (!tty?.stdin?.isTTY || !tty.stderr?.isTTY) {
    throw new Error(`signing in to ${label(provider)} asks questions in a terminal; run it in one (or save an API key: login(${JSON.stringify(asked)}, { key }))`);
  }
  const ui = new TerminalUI({ openBrowser: opts.openBrowser ?? Boolean(env()["DISPLAY"] || env()["WAYLAND_DISPLAY"] || platform() === "darwin") });
  const conn = await a.login(provider, { ui, ...(method ? { method } : {}), ...(replace ? { replace } : {}) });
  say(`Signed in: ${conn.label || label(provider)}.${hint}`);
  return found();
}

/**
 * Forget the saved login or key for a provider, on this machine (the account
 * itself is untouched). Keys in the environment still work afterwards, except
 * where lm15 blocks them on purpose (a signed-out xAI subscription does not
 * fall back to `XAI_API_KEY`).
 */
export async function logout(provider: string): Promise<void> {
  const a = localAuth();
  if (!a) throw new Error("logout needs lm15's credentials file (Node)");
  provider = canonical(provider);
  if (!(await a.status(provider)).connection) {
    console.error(`Not signed in to ${provider}.`);
    return;
  }
  await a.logout(provider);
  console.error(`Signed out of ${label(provider)}.`);
}

const routers = new Map<string, LMRouter>();

/**
 * The router a model's calls go through when none is set: one that uses the
 * saved login for the model's provider when there is one, else the
 * environment's keys and CLI logins (lm15's own rules).
 */
export function routerFor(model: string): LMRouter {
  const a = localAuth();
  if (!a) return shared$(("env"), () => new LMRouter());
  const namer = shared$("auth", () => new LMRouter({ auth: a }));
  let provider: string;
  try {
    provider = namer.resolve(model).provider;
  } catch {
    return shared$("env", () => new LMRouter());
  }
  let saved = false;
  try {
    saved = a.statusSync(provider)?.connection != null;
  } catch {
    saved = false;
  }
  return saved ? namer : shared$("env", () => new LMRouter());
}

function shared$(key: string, make: () => LMRouter): LMRouter {
  let r = routers.get(key);
  if (!r) routers.set(key, r = make());
  return r;
}
