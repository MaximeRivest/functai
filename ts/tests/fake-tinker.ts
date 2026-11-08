/** tools/fake_tinker.py, started for a test (its own process, Python's environment): null when it cannot start. */
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { CONTRACT } from "../tools/generate.ts";

const ROOT = join(CONTRACT, "..");
const PY = process.env["FUNCTAI_PYTHON"] ?? join(ROOT, "python", ".venv", "bin", "python");

export interface Fake {
  url: string;
  config(options: Record<string, unknown>): Promise<unknown>;
  state(): Promise<Record<string, any>>;
  stop(): void;
}

/** Why the fake cannot start here, or null. */
export function fakeMissing(): string | null {
  if (!existsSync(PY)) return `no Python environment at ${PY} (cd python && uv sync --all-extras): the fake Tinker runs in it`;
  const r = spawnSync(PY, ["-c", "import tinker.proto.tinker_public_pb2"], { encoding: "utf8" });
  return r.status === 0 ? null : "Python's environment has no tinker SDK (the fake reads its protobuf definitions)";
}

export async function startFake(apiKey = "tml-test"): Promise<Fake> {
  const child = spawn(PY, [join(ROOT, "tools", "fake_tinker.py"), "--api-key", apiKey], { stdio: ["ignore", "pipe", "inherit"] });
  const url: string = await new Promise((resolve, reject) => {
    let buf = "";
    child.stdout!.on("data", (d: Buffer) => {
      buf += d.toString();
      const line = buf.split("\n")[0];
      if (buf.includes("\n") && line) resolve(line.trim());
    });
    child.on("exit", (code) => reject(new Error(`the fake Tinker exited (${code})`)));
  });
  const ask = async (method: string, path: string, body?: unknown) => {
    const r = await fetch(url + path, { method, headers: { "content-type": "application/json", "x-api-key": apiKey }, body: body === undefined ? undefined : JSON.stringify(body) });
    return r.json();
  };
  return {
    url, config: (o) => ask("POST", "/_fake/config", o), state: () => ask("GET", "/_fake/state") as Promise<Record<string, any>>,
    stop: () => child.kill(),
  };
}
