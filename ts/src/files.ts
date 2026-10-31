/**
 * Files as inputs: any file, folder or web page an AI function reads, through
 * [Attachments](https://attachments.dev) (`attachments-dev` on npm).
 *
 * ```ts
 * import { ai, t } from "functai";
 * import { artifacts } from "functai/files";
 * import { att } from "attachments-dev";
 *
 * const summarize = ai("summarize", {
 *   description: "The report in three sentences.",
 *   input: { report: artifacts() },
 *   output: t.string(),
 * });
 * await summarize({ report: await att("report.pdf[images: true]") });   // each page's text, then its picture
 * ```
 *
 * `att()` turns a PDF, a Word or Excel file, slides, a web page, a folder,
 * a repository or a picture into artifacts: text, pictures, metadata. As an
 * input (contract/functions.md, "Pictures, sounds, documents and files"):
 *
 * - **The model sees** Attachments' own `parts()` at the input's place: each
 *   file's name, its text, and each page's pictures after that page's text,
 *   as pictures (never as base64 text). No files (`att([])`): `(no content)`.
 * - **Its data** is Attachments' wire form (`toWire()`), in the call's
 *   record, the reply cache's key, a conversation and a served call; a
 *   served function takes it from any language, and Python's `Artifacts`
 *   input is the same input.
 * - **A file that failed** (`att()` never throws: a missing file is an
 *   artifact with an error) refuses the call before any request
 *   (`InterfaceError`, `interface-input`), naming the file and why.
 *
 * Importing this module plugs Attachments in. A function saved in another
 * language that reads files loads only after it has been imported once
 * (`import "functai/files"`); before, it refuses `files-unavailable`.
 */

import { Artifacts, type WireArtifact } from "attachments-dev";
import { filesShape, useFiles } from "./files-binding.ts";
import type { Shape } from "./shapes.ts";

/** What an `artifacts()` input takes: `att()`'s result, or its wire form (`toWire()`, a served call's JSON). */
export type FilesInput = Artifacts | readonly WireArtifact[];

useFiles({ fromWire: (data: unknown) => Artifacts.fromWire(data) as never });

/**
 * An input of files: `att()`'s result (`Artifacts`). Its words, if any, are
 * the field's: `artifacts({ description: "the quarterly report" })`. Files are
 * a whole input: not optional and not in a list (several files are one
 * `att([...])`; none is `att([])`).
 */
export function artifacts(extra: { readonly description?: string } = {}): Shape<FilesInput> {
  if (typeof extra !== "object" || extra === null || Array.isArray(extra) || Object.keys(extra).some((k) => k !== "description")) {
    throw new TypeError("artifacts(extra): extra takes only a description: artifacts({ description: \"the report\" })");
  }
  if (extra.description !== undefined && typeof extra.description !== "string") throw new TypeError("artifacts({ description }): the description is text");
  return filesShape(extra.description) as unknown as Shape<FilesInput>;
}
