/**
 * Pictures, sounds, videos and documents as inputs (contract/functions.md,
 * "Pictures, sounds, documents and files"). A field's shape is lmcc's media
 * shape, `{"media": kind}`; its value is lm15's part: the one lm15's own
 * `image({ path })` makes, or the same as data (`{ media_type, data }`,
 * `{ media_type, url }`, `{ path }`). Either way the model is sent the part
 * itself, never its bytes as text.
 */

import { Part, type AudioPart, type BinaryPart, type DocumentPart, type ImagePart, type VideoPart } from "@lm15/lm15";
import { entriesOf, setOwn } from "./values.ts";

type Rec = Record<string, unknown>;

/** lmcc's media kinds (kernel §7b). */
export const MEDIA_KINDS = ["image", "audio", "video", "document", "binary"] as const;
export type MediaKind = (typeof MEDIA_KINDS)[number];

/**
 * A part of this kind as data, as lm15 serializes one (and as Python logs an
 * `ImagePart`): its media type, and one of `data` (base64), `url`, `file_id`
 * or `path` (a local file, read when the call is bound).
 */
export interface MediaData<K extends MediaKind = MediaKind> {
  readonly type?: K;
  readonly media_type?: string;
  readonly data?: string;
  readonly url?: string;
  readonly file_id?: string;
  readonly path?: string;
}

/** What a `t.image()` input takes: lm15's `image(...)`, or a picture as data. */
export type ImageInput = ImagePart | (MediaData<"image"> & { readonly detail?: "auto" | "low" | "high" });
/** What a `t.audio()` input takes: lm15's `audio(...)`, or a recording as data. */
export type AudioInput = AudioPart | MediaData<"audio">;
/** What a `t.video()` input takes: lm15's `video(...)`, or a video as data. */
export type VideoInput = VideoPart | MediaData<"video">;
/** What a `t.document()` input takes: lm15's `document(...)` (a PDF given whole), or one as data. */
export type DocumentInput = DocumentPart | MediaData<"document">;
/** What a `t.binary()` input takes: lm15's `binary(...)`, or bytes of any type as data. */
export type BinaryInput = BinaryPart | MediaData<"binary">;

/**
 * calls.md, "Versions": a media shape's sample is a part addressed by a web
 * address (nothing fetches it), so a function's version holds its
 * instruction and layout.
 */
const SAMPLE: Readonly<Record<MediaKind, readonly [string, string]>> = {
  image: ["image/png", "png"], audio: ["audio/wav", "wav"], video: ["video/mp4", "mp4"],
  document: ["application/pdf", "pdf"], binary: ["application/octet-stream", "bin"],
};

export function mediaKind(shape: unknown): MediaKind | null {
  if (typeof shape !== "object" || shape === null || Array.isArray(shape)) return null;
  const kind = (shape as Rec)["media"];
  return typeof kind === "string" && (MEDIA_KINDS as readonly string[]).includes(kind) ? kind as MediaKind : null;
}

/** The sample value of a media shape, or undefined when the shape is not one. */
export function mediaSample(shape: unknown): Rec | undefined {
  const kind = mediaKind(shape);
  if (!kind) return undefined;
  const [mediaType, ext] = SAMPLE[kind];
  return { media_type: mediaType, url: `https://example.com/sample.${ext}` };
}

const SUBSHAPES = ["items", "additionalProperties", "anyOf", "oneOf", "allOf", "prefixItems", "properties", "patternProperties", "$defs"];

/** Whether a shape holds a media shape anywhere (itself, a list's items, an option, a member). */
export function holdsMedia(shape: unknown, seen = new Set<unknown>()): boolean {
  if (typeof shape !== "object" || shape === null || seen.has(shape)) return false;
  seen.add(shape);
  if (Array.isArray(shape)) return shape.some((s) => holdsMedia(s, seen));
  if (mediaKind(shape)) return true;
  return SUBSHAPES.some((k) => {
    const v = (shape as Rec)[k];
    if (k === "properties" || k === "patternProperties" || k === "$defs") {
      return typeof v === "object" && v !== null && Object.values(v).some((x) => holdsMedia(x, seen));
    }
    return holdsMedia(v, seen);
  });
}

/** An lm15-ts part held in memory: a media kind with lm15-ts's camelCase members (`mediaType`, `fileId`). */
function isLm15Media(v: unknown): v is ImagePart {
  if (typeof v !== "object" || v === null || Array.isArray(v)) return false;
  const r = v as Rec;
  return typeof r["type"] === "string" && (MEDIA_KINDS as readonly string[]).includes(r["type"] as string)
    && (Object.hasOwn(r, "mediaType") || Object.hasOwn(r, "fileId")) && !Object.hasOwn(r, "media_type");
}

/**
 * The value with every lm15-ts media part in it as lm15's part data
 * (`Part.toJSON`: `media_type`, `file_id`), as Python logs an `ImagePart`:
 * the form a call's record, the reply cache's key and every other language
 * read. Lists and plain objects are walked; anything else is left as it is.
 */
export function mediaAsData(v: unknown): unknown {
  if (isLm15Media(v)) return Part.toJSON(v);
  if (Array.isArray(v)) {
    let changed = false;
    const out = v.map((x) => {
      const y = mediaAsData(x);
      if (y !== x) changed = true;
      return y;
    });
    return changed ? out : v;
  }
  if (typeof v === "object" && v !== null && Object.getPrototypeOf(v) === Object.prototype) {
    const entries = entriesOf(v as Rec).map(([k, x]) => [k, x, mediaAsData(x)] as const);
    if (entries.every(([, x, y]) => x === y)) return v;
    const out: Rec = {};                       // members in the value's order (values.ts): integer-like names too
    for (const [k, , y] of entries) setOwn(out, k, y);
    return out;
  }
  return v;
}
