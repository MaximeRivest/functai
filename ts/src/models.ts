/**
 * Which model, how to reach it, and what it can do (contract/functions.md,
 * "Capabilities"). lm15 routes a model string to a provider; the facts
 * about the model come from the contract's table, never from guessing.
 */

import { LMRouter } from "@lm15/lm15";
import { MODELS } from "./generated/contract.ts";

export type Capabilities = Record<string, boolean>;

/** The fixed facts a version is computed under (provider "probe"). */
export const PROBE: Capabilities = Object.fromEntries(
  Object.entries(MODELS.probe).filter(([k]) => k !== "about"),
) as Capabilities;

const set = (xs: readonly string[]) => new Set<string>(xs);
const NATIVE = set(MODELS.native.providers);
const NO_STOP = set(MODELS.native.no_stop_sequences);
const PREFILL = set(MODELS.native.assistant_prefill);
const CHAT = set(MODELS.chat_completions.providers);
const TOOL_HOSTS = set(MODELS.native_tool_hosts.providers);
export const JUDGMENT_ONLY = set(MODELS.judgment_only.providers);
const SPEAKS_AS: Record<string, string> = Object.fromEntries(
  Object.entries(MODELS.speaks_as).filter(([k]) => k !== "about"),
) as Record<string, string>;
const PREFIXES = MODELS.native.reasoning_prefixes as Record<string, readonly string[]>;

/** The facts FunctAI declares for `model` served by `provider`. */
export function capabilities(provider: string, model: string): Capabilities {
  if (JUDGMENT_ONLY.has(provider)) return { native_structured_output: true };
  if (provider in SPEAKS_AS) return capabilities(SPEAKS_AS[provider]!, model);
  const caps: Capabilities = { instruct: true };
  if (NATIVE.has(provider)) {
    caps["native_function_calling"] = true;
    caps["native_structured_output"] = true;
    caps["stop_sequences"] = !NO_STOP.has(provider);
    caps["native_reasoning"] = (PREFIXES[provider] ?? []).some((p) => model.startsWith(p));
    caps["assistant_prefill"] = PREFILL.has(provider) && !caps["native_reasoning"];
  } else if (CHAT.has(provider)) {
    Object.assign(caps, { native_function_calling: true, native_structured_output: true, stop_sequences: false, native_reasoning: false });
  } else {
    caps["native_function_calling"] = TOOL_HOSTS.has(provider);
    caps["stop_sequences"] = false;
  }
  return caps;
}

/** The facts for a call: the table, Anthropic's temperature rule, then the function's own overrides. */
export function callCapabilities(provider: string, model: string, settings: { temperature?: number | null; capabilities?: Capabilities | null }): Capabilities {
  const caps = capabilities(provider, model);
  const anthropic = provider === "anthropic" || SPEAKS_AS[provider] === "anthropic";
  const t = settings.temperature;
  if (anthropic && caps["native_reasoning"] && t !== undefined && t !== null && t !== 1) {
    caps["native_reasoning"] = false;
    caps["assistant_prefill"] = true;
  }
  return { ...caps, ...(settings.capabilities ?? {}) };
}

/** Settings a model refuses are left out of its requests, with one warning per
 * provider, instead of failing every call (contract/functions.md, "Sampling a
 * model does not take"). */
const FIXED_SAMPLING = (MODELS as unknown as { fixed_sampling: Record<string, readonly string[] | string> }).fixed_sampling;
const SAMPLING = ["temperature", "topP"] as const;
const refusedWarned = new Set<string>();

export function refusedSettings(provider: string, model: string): string[] {
  if (provider === "openai-codex") return [...SAMPLING, "maxTokens"];          // no knobs, no output cap
  const prefixes = FIXED_SAMPLING[SPEAKS_AS[provider] ?? provider];
  return Array.isArray(prefixes) && prefixes.some((p) => model.startsWith(p)) ? [...SAMPLING] : [];
}

export function adjustSettings<S extends { temperature?: number | null; topP?: number | null; maxTokens?: number | null }>(
  s: S, provider: string, model: string,
): S {
  const drop = refusedSettings(provider, model).filter((k) => {
    const v = (s as Record<string, unknown>)[k];
    return v !== undefined && v !== null && (k === "maxTokens" || provider === "openai-codex" || v !== 1);
  });
  if (!drop.length) return s;
  const key = `${provider}:${drop.join(",")}`;
  if (!refusedWarned.has(key)) {
    refusedWarned.add(key);
    const names = drop.map((k) => (k === "topP" ? "top_p" : k === "maxTokens" ? "max_tokens" : k));
    console.warn(`functai: ${provider}:${model} does not take ${names.join(", ")}; left out of its requests`);
  }
  const out: Record<string, unknown> = { ...s };
  for (const k of drop) out[k] = null;
  return out as S;
}

/** Friendly account prefixes, as in Python: `claude:` is `claude-code:`, … */
const PREFIX_ALIASES: Record<string, string> = { claude: "claude-code", chatgpt: "openai-codex", copilot: "github-copilot", kimi: "kimi-code" };

export function modelString(lm: string): string {
  const i = lm.indexOf(":");
  if (i > 0) {
    const head = lm.slice(0, i).toLowerCase();
    if (head in PREFIX_ALIASES) return `${PREFIX_ALIASES[head]}:${lm.slice(i + 1)}`;
  }
  return lm;
}

/** A model this machine can use when none is configured: the first provider with a key in the environment. */
export function defaultModel(env: Record<string, string | undefined>): string | null {
  const picks: [string, string][] = [
    ["OPENAI_API_KEY", "gpt-4.1-mini"], ["ANTHROPIC_API_KEY", "claude-haiku-4-5"],
    ["GEMINI_API_KEY", "gemini:gemini-2.5-flash"], ["GOOGLE_API_KEY", "gemini:gemini-2.5-flash"],
    ["GROQ_API_KEY", "groq:openai/gpt-oss-120b"], ["OPENROUTER_API_KEY", "openrouter:openai/gpt-4.1-mini"],
  ];
  for (const [key, model] of picks) if (env[key]) return model;
  return null;
}

let shared: LMRouter | null = null;

/** The router calls go through unless a function or `configure` gives one. */
export function defaultRouter(): LMRouter {
  shared ??= new LMRouter();
  return shared;
}
