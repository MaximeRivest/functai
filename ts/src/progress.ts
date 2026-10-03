/**
 * A progress line for long runs (`map`, `evaluate`): rows done, errors,
 * tokens, time left, written on stderr and updated in place.
 */

type Out = { write(s: string): unknown; isTTY?: boolean };

const stderr = (): Out | null => (globalThis as { process?: { stderr?: Out } }).process?.stderr ?? null;

/** `progress` unset: on when stderr is a terminal. */
export function showProgress(setting: boolean | null | undefined): boolean {
  if (setting !== undefined && setting !== null) return setting;
  return Boolean(stderr()?.isTTY);
}

function duration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** A line on stderr, updated as rows finish (at most five times a second, and at the end). */
export class Progress {
  private done = 0;
  private errors = 0;
  private tokens = 0;
  private readonly t0 = Date.now();
  private shown = 0;
  private readonly total: number;
  private readonly name: string;
  private readonly out: Out | null;

  constructor(total: number, name: string, out: Out | null = stderr()) {
    this.total = total;
    this.name = name;
    this.out = out;
  }

  /** A row finished: whether it failed, and the tokens it used. */
  step(failed: boolean, tokens: number): void {
    this.done++;
    if (failed) this.errors++;
    this.tokens += tokens;
    const now = Date.now();
    if (now - this.shown >= 200 || this.done === this.total) {
      this.shown = now;
      try {
        this.out?.write("\r" + this.line() + (this.done === this.total ? "\n" : ""));
      } catch {
        // a progress line never fails a run
      }
    }
  }

  line(): string {
    const elapsed = (Date.now() - this.t0) / 1000;
    const left = this.done ? (elapsed / this.done) * (this.total - this.done) : null;
    const parts = [`${this.name}: ${this.done}/${this.total} rows`];
    if (this.errors) parts.push(`${this.errors} error${this.errors === 1 ? "" : "s"}`);
    parts.push(`${this.tokens.toLocaleString("en-US")} tokens`);
    parts.push(this.done === this.total || left === null ? `${duration(elapsed)} so far` : `about ${duration(left)} left`);
    return parts.join(" · ");
  }
}
