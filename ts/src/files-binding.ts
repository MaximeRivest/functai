/**
 * Files as an input (contract/functions.md, "Pictures, sounds, documents and
 * files"): an input whose type is `Artifacts` takes Attachments' `att()`
 * results. This module is the rule, with no dependency: what such an input
 * is, how a value binds, and the refusals. Attachments itself (reading the
 * wire form back, its `parts()`) is plugged in by `functai/files`, which
 * imports `attachments-dev`; until it is, a function reading files refuses
 * `files-unavailable` and sends nothing. The main entry never imports
 * Attachments, so a program that reads no files never needs it, and a
 * bundler never looks for it.
 */

import { FilesUnavailable } from "./errors.ts";
import { jsonForm } from "./values.ts";

type Rec = Record<string, unknown>;

/** The type name of an input of files: in a signature, a saved node and a served interface. */
export const ARTIFACTS = "Artifacts";

/** Its shape: Attachments' wire form, a list of artifacts. */
export const FILES_SHAPE: Readonly<Rec> = Object.freeze({ type: "array", items: Object.freeze({ type: "object" }) });

/** A shape built by `artifacts()`, and what carries one (`t.optional`, `t.list`): refused unless it is the whole input. */
export const FILES = Symbol.for("functai.files");

/** What FunctAI needs of Attachments: its own reading of the wire form, and its parts. */
export interface FilesReader {
  /** Attachments' `Artifacts.fromWire`: throws on data it refuses (its contract's `cases/wire.json`). */
  fromWire(data: unknown): { readonly errors: readonly Rec[]; parts(): unknown[]; toWire(): unknown[] };
}

let reader: FilesReader | null = null;

/** Plug Attachments in (`functai/files` does, once). */
export function useFiles(r: FilesReader): void {
  reader = r;
}

/** Attachments, or `files-unavailable` naming what needs it. */
export function filesReader(where: string): FilesReader {
  if (reader) return reader;
  throw new FilesUnavailable(`${where} reads files (an Artifacts input), and Attachments is not loaded here: `
    + `npm install attachments-dev, then import "functai/files" once (import { artifacts } from "functai/files" does it)`);
}

/** A shape of files, marked as one (what `artifacts()` gives; a remote program's input of files). */
export function filesShape(description?: string): Rec {
  const shape: Rec = { type: "array", items: { type: "object" }, ...(description !== undefined ? { description } : {}) };
  Object.defineProperty(shape, FILES, { value: true });
  return shape;
}

/** Whether a shape carries `artifacts()` (itself, or wrapped). */
export function carriesFiles(shape: unknown): boolean {
  return typeof shape === "object" && shape !== null && (shape as Record<symbol, unknown>)[FILES] === true;
}

/** A value an input of files refuses: the call is refused `interface-input`, naming the input, before any request. */
export class FilesRefused extends Error {}

/** What a list of artifacts that failed says (the first five: source, code, message). */
function failures(errors: readonly Rec[]): string {
  const shown = errors.slice(0, 5).map((e) => `${String(e["source"])}: ${String(e["code"])} — ${String(e["message"])}`);
  return shown.join("; ") + (errors.length > 5 ? `; and ${errors.length - 5} more` : "");
}

/**
 * An input of files bound (functions.md): `att()`'s result, or its wire
 * form, as the wire form (the input's JSON: what the record, the reply
 * cache's key, a conversation and a served call hold). Refused, with
 * `FilesRefused`: what Attachments' `fromWire` refuses, and a file that
 * failed (`meta.error`).
 */
export function bindFiles(value: unknown, where: string): unknown {
  const r = filesReader(where);
  let wire: unknown = value;
  if (typeof value === "object" && value !== null && typeof (value as { toWire?: unknown }).toWire === "function") {
    wire = (value as { toWire(): unknown }).toWire();         // att()'s result, from any copy of attachments-dev
  }
  const data = jsonForm(wire);
  if (!Array.isArray(data)) {
    throw new FilesRefused(`takes att(...)'s result or its wire form (a list of artifacts), not ${data === undefined ? typeof value : JSON.stringify(data)?.slice(0, 40)}`);
  }
  let read: ReturnType<FilesReader["fromWire"]>;
  try {
    read = r.fromWire(data);
  } catch (err) {
    throw new FilesRefused(`is not Attachments' wire form: ${(err as Error).message}`);
  }
  if (read.errors.length) {
    throw new FilesRefused(`${read.errors.length} of its files could not be read, so the model would get an empty or partial `
      + `document: ${failures(read.errors)}. Do what each says (att(...).errors lists them), or leave that file out`);
  }
  return data;
}

/** What the model sees for an input of files: Attachments' parts, or one text part `(no content)`. */
export function filesParts(value: unknown, where: string): unknown[] {
  const parts = filesReader(where).fromWire(value).parts();
  return parts.length ? parts : [{ type: "text", text: "(no content)" }];
}
