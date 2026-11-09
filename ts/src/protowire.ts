/**
 * The protobuf messages Tinker uses for training and sampling results
 * (`tinker_public.proto`, recovered from the official Python SDK 0.39), written
 * and read by hand: varints, length-delimited fields, little-endian arrays.
 * Only what FunctAI sends and reads: `ForwardBackwardRequest` (written),
 * `ForwardBackwardOutput` and `SampleResponse` (read).
 */

// ------------------------------------------------------------------ writing

class Writer {
  #parts: Uint8Array[] = [];
  #size = 0;

  #push(b: Uint8Array): void {
    this.#parts.push(b);
    this.#size += b.length;
  }

  varint(n: number | bigint): void {
    let v = BigInt.asUintN(64, BigInt(n));
    const out: number[] = [];
    while (v >= 0x80n) {
      out.push(Number(v & 0x7fn) | 0x80);
      v >>= 7n;
    }
    out.push(Number(v));
    this.#push(Uint8Array.from(out));
  }

  tag(field: number, wire: number): void {
    this.varint((field << 3) | wire);
  }

  bytes(field: number, b: Uint8Array): void {
    this.tag(field, 2);
    this.varint(b.length);
    this.#push(b);
  }

  string(field: number, s: string): void {
    this.bytes(field, new TextEncoder().encode(s));
  }

  message(field: number, w: Writer): void {
    this.bytes(field, w.finish());
  }

  int(field: number, n: number): void {
    this.tag(field, 0);
    this.varint(n);
  }

  bool(field: number, b: boolean): void {
    if (b) this.int(field, 1);
  }

  double(field: number, x: number): void {
    this.tag(field, 1);
    const b = new Uint8Array(8);
    new DataView(b.buffer).setFloat64(0, x, true);
    this.#push(b);
  }

  packedInts(field: number, xs: readonly number[]): void {
    const w = new Writer();
    for (const x of xs) w.varint(x);
    this.bytes(field, w.finish());
  }

  finish(): Uint8Array {
    const out = new Uint8Array(this.#size);
    let at = 0;
    for (const p of this.#parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }
}

/** int32 little-endian bytes. */
export function int32s(xs: readonly number[]): Uint8Array {
  const b = new Uint8Array(xs.length * 4);
  const v = new DataView(b.buffer);
  xs.forEach((x, i) => v.setInt32(i * 4, x, true));
  return b;
}

function float32s(xs: readonly number[]): Uint8Array {
  const b = new Uint8Array(xs.length * 4);
  const v = new DataView(b.buffer);
  xs.forEach((x, i) => v.setFloat32(i * 4, x, true));
  return b;
}

function int64s(xs: readonly number[]): Uint8Array {
  const b = new Uint8Array(xs.length * 8);
  const v = new DataView(b.buffer);
  xs.forEach((x, i) => v.setBigInt64(i * 8, BigInt(x), true));
  return b;
}

/** A model input chunk as the wire takes it. */
export type WireChunk = { readonly tokens: readonly number[] } | { readonly image: Uint8Array; readonly format: "png" | "jpeg"; readonly expectedTokens: number };

/** One datum: its chunks and its loss inputs (`target_tokens` int64; the others float32; rows of K flattened with shape [N, K]). */
export interface WireDatum {
  readonly chunks: readonly WireChunk[];
  readonly inputs: Readonly<Record<string, readonly number[] | readonly (readonly number[])[]>>;
}

const INT64_INPUTS = new Set(["target_tokens"]);

function tensor(name: string, values: readonly number[] | readonly (readonly number[])[]): Writer {
  const w = new Writer();
  const rows = values.length > 0 && Array.isArray(values[0]);
  const flat = rows ? (values as readonly (readonly number[])[]).flat() : (values as readonly number[]);
  const shape = rows ? [values.length, (values[0] as readonly number[]).length] : [flat.length];
  const int = INT64_INPUTS.has(name);
  w.bytes(1, int ? int64s(flat) : float32s(flat));        // dense
  w.int(3, int ? 2 : 1);                                   // DTYPE_INT64 : DTYPE_FLOAT32
  w.packedInts(4, shape);
  return w;
}

/** `ForwardBackwardRequest` (fields 1 model_id, 2 seq_id, 3 data, 4 loss_fn, 5/7 loss_fn_config, 6 forward_only). */
export function forwardBackwardRequest(r: { modelId: string; seqId: number; lossFn: string; data: readonly WireDatum[]; config?: Readonly<Record<string, number | string>>; forwardOnly?: boolean }): Uint8Array {
  const w = new Writer();
  w.string(1, r.modelId);
  w.int(2, r.seqId);
  for (const d of r.data) {
    const dw = new Writer();
    for (const c of d.chunks) {
      const cw = new Writer();
      if ("tokens" in c) {
        const tw = new Writer();
        tw.bytes(1, int32s(c.tokens));
        cw.message(1, tw);
      } else {
        const iw = new Writer();
        iw.bytes(1, c.image);
        iw.string(2, c.format);
        iw.int(3, c.expectedTokens);
        cw.message(2, iw);
      }
      dw.message(1, cw);
    }
    for (const [name, values] of Object.entries(d.inputs)) {
      const ew = new Writer();
      ew.string(1, name);
      ew.message(2, tensor(name, values));
      dw.message(2, ew);
    }
    w.message(3, dw);
  }
  w.string(4, r.lossFn);
  for (const [k, v] of Object.entries(r.config ?? {})) {
    if (typeof v === "number") {
      const e = new Writer();
      e.string(1, k);
      e.double(2, v);
      w.message(5, e);
    }
    const e2 = new Writer();
    e2.string(1, k);
    const val = new Writer();
    if (typeof v === "number") val.double(1, v);
    else val.string(2, v);
    e2.message(2, val);
    w.message(7, e2);
  }
  w.bool(6, Boolean(r.forwardOnly));
  return w.finish();
}

// ------------------------------------------------------------------ reading

class Reader {
  readonly b: Uint8Array;
  at = 0;
  constructor(b: Uint8Array) {
    this.b = b;
  }

  get done(): boolean {
    return this.at >= this.b.length;
  }

  varint(): bigint {
    let v = 0n;
    let shift = 0n;
    for (;;) {
      if (this.at >= this.b.length) throw new Error("protobuf: a varint runs past the end");
      const x = this.b[this.at++]!;
      v |= BigInt(x & 0x7f) << shift;
      if (!(x & 0x80)) return v;
      shift += 7n;
    }
  }

  tag(): [number, number] {
    const t = Number(this.varint());
    return [t >>> 3, t & 7];
  }

  bytes(): Uint8Array {
    const n = Number(this.varint());
    const out = this.b.subarray(this.at, this.at + n);
    this.at += n;
    return out;
  }

  double(): number {
    const v = new DataView(this.b.buffer, this.b.byteOffset + this.at, 8).getFloat64(0, true);
    this.at += 8;
    return v;
  }

  skip(wire: number): void {
    if (wire === 0) this.varint();
    else if (wire === 1) this.at += 8;
    else if (wire === 2) this.bytes();
    else if (wire === 5) this.at += 4;
    else throw new Error(`protobuf: wire type ${wire} is not read here`);
  }
}

function fields(b: Uint8Array, each: (field: number, wire: number, r: Reader) => void): void {
  const r = new Reader(b);
  while (!r.done) {
    const [f, wire] = r.tag();
    const before = r.at;
    each(f, wire, r);
    if (r.at === before) r.skip(wire);
  }
}

function view(b: Uint8Array): DataView {
  return new DataView(b.buffer, b.byteOffset, b.byteLength);
}

function f32(b: Uint8Array): number[] {
  const v = view(b);
  return Array.from({ length: b.length / 4 }, (_, i) => v.getFloat32(i * 4, true));
}

function i32(b: Uint8Array): number[] {
  const v = view(b);
  return Array.from({ length: b.length / 4 }, (_, i) => v.getInt32(i * 4, true));
}

function i64(b: Uint8Array): number[] {
  const v = view(b);
  return Array.from({ length: b.length / 8 }, (_, i) => Number(v.getBigInt64(i * 8, true)));
}

function bf16(b: Uint8Array): number[] {
  const v = view(b);
  const out = new Float32Array(b.length / 2);
  const u = new Uint32Array(out.buffer);
  for (let i = 0; i < out.length; i++) u[i] = v.getUint16(i * 2, true) << 16;
  return [...out];
}

function decodeArray(dtype: number, b: Uint8Array): number[] {
  if (dtype === 1) return f32(b);
  if (dtype === 2) return i64(b);
  if (dtype === 3) return i32(b);
  if (dtype === 4) return bf16(b);
  throw new Error(`protobuf: tensor dtype ${dtype} is not read here`);
}

/** A `Tensor`: its values (row-major) and shape. A sparse one (CSR: Tinker answers a sparse
 * `target_prompt_logprobs` request with one) is made dense, NaN where no value is listed. */
export function readTensor(b: Uint8Array): { data: number[]; shape: number[] } {
  let dense: Uint8Array | null = null;
  const sparse: { csr: { values: Uint8Array; crow: number[]; col: number[] } | null } = { csr: null };
  let dtype = 1;
  const shape: number[] = [];
  fields(b, (f, wire, r) => {
    if (f === 1) dense = r.bytes();
    else if (f === 2) {
      const c: { values: Uint8Array; crow: number[]; col: number[] } = { values: new Uint8Array(), crow: [], col: [] };
      fields(r.bytes(), (g, _w, s) => {
        if (g === 1) c.values = s.bytes();
        else if (g === 2) c.crow = i64(s.bytes());
        else if (g === 3) c.col = i64(s.bytes());
      });
      sparse.csr = c;
    } else if (f === 3) dtype = Number(r.varint());
    else if (f === 4) {
      if (wire === 2) {
        const inner = new Reader(r.bytes());
        while (!inner.done) shape.push(Number(inner.varint()));
      } else shape.push(Number(r.varint()));
    }
  });
  const csr = sparse.csr;
  if (csr) return { data: densify(decodeArray(dtype, csr.values), csr, shape), shape };
  const data = dense ? decodeArray(dtype, dense) : [];
  return { data, shape: shape.length ? shape : [data.length] };
}

/** A CSR matrix's values in a dense row-major array of `shape` ([rows, columns]), NaN elsewhere. */
export function densify(values: readonly number[], csr: { crow: readonly number[]; col: readonly number[] }, shape: readonly number[]): number[] {
  const [rows, cols] = [shape[0] ?? 0, shape[1] ?? 1];
  if (shape.length !== 2 || csr.crow.length !== rows + 1 || csr.col.length !== values.length || csr.crow[rows] !== values.length)
    throw new Error(`protobuf: a sparse tensor that does not fit its shape [${shape}] (${csr.crow.length} row pointers, ${values.length} values)`);
  const out = new Array<number>(rows * cols).fill(NaN);
  for (let i = 0; i < rows; i++)
    for (let j = csr.crow[i]!; j < csr.crow[i + 1]!; j++) {
      const c = csr.col[j]!;
      if (c < 0 || c >= cols) throw new Error(`protobuf: a sparse tensor's column ${c} is outside its ${cols} columns`);
      out[i * cols + c] = values[j]!;
    }
  return out;
}

/** A sampled sequence. */
export interface WireSequence {
  readonly stopReason: "stop" | "length";
  readonly tokens: number[];
  readonly logprobs: number[] | null;
}

/** `SampleResponse`: sequences, prompt log-probabilities (NaN where none), target log-probabilities. */
export function readSampleResponse(b: Uint8Array): { sequences: WireSequence[]; promptLogprobs: number[] | null; targetPromptLogprobs: { data: number[]; shape: number[] } | null; promptCacheHitTokens: number } {
  const sequences: WireSequence[] = [];
  let promptLogprobs: number[] | null = null;
  let target: { data: number[]; shape: number[] } | null = null;
  let hit = 0;
  fields(b, (f, _w, r) => {
    if (f === 1) {
      let stop = 0;
      let tokens: number[] = [];
      let logprobs: number[] | null = null;
      fields(r.bytes(), (g, _w2, s) => {
        if (g === 1) stop = Number(s.varint());
        else if (g === 2) tokens = i32(s.bytes());
        else if (g === 3) logprobs = f32(s.bytes());
      });
      sequences.push({ stopReason: stop === 0 ? "stop" : "length", tokens, logprobs });
    } else if (f === 2) promptLogprobs = f32(r.bytes());
    else if (f === 4) hit = Number(r.varint());
    else if (f === 5) target = readTensor(r.bytes());
  });
  return { sequences, promptLogprobs, targetPromptLogprobs: target, promptCacheHitTokens: hit };
}

/** `ForwardBackwardOutput`: each datum's output fields (flat values with their shape), and the metrics. */
export function readForwardBackwardOutput(b: Uint8Array): { outputs: Record<string, { data: number[]; shape: number[] }>[]; metrics: Record<string, number>; type: string } {
  const outputs: Record<string, { data: number[]; shape: number[] }>[] = [];
  const metrics: Record<string, number> = {};
  let type = "";
  fields(b, (f, _w, r) => {
    if (f === 1) type = new TextDecoder().decode(r.bytes());
    else if (f === 2) {
      const record: Record<string, { dtype: number; data: Uint8Array; offsets: number[]; trailing: number[] }> = {};
      let numDatums = 0;
      fields(r.bytes(), (g, _w2, s) => {
        if (g === 2) {
          let key = "";
          let bt: { dtype: number; data: Uint8Array; offsets: number[]; trailing: number[] } | null = null;
          fields(s.bytes(), (h, _w3, t) => {
            if (h === 1) key = new TextDecoder().decode(t.bytes());
            else if (h === 2) {
              const one = { dtype: 1, data: new Uint8Array(0) as Uint8Array, offsets: [] as number[], trailing: [] as number[] };
              fields(t.bytes(), (k, wire, u) => {
                if (k === 1) one.data = u.bytes();
                else if (k === 2) one.offsets = i64(u.bytes());
                else if (k === 3) one.dtype = Number(u.varint());
                else if (k === 4) {
                  if (wire === 2) {
                    const inner = new Reader(u.bytes());
                    while (!inner.done) one.trailing.push(Number(inner.varint()));
                  } else one.trailing.push(Number(u.varint()));
                }
              });
              bt = one;
            }
          });
          if (bt) record[key] = bt;
        } else if (g === 3) numDatums = Number(s.varint());
      });
      const names = Object.keys(record);
      const n = numDatums || (names.length ? record[names[0]!]!.offsets.length - 1 : 0);
      const start = outputs.length;
      for (let i = 0; i < n; i++) outputs.push({});
      for (const name of names) {
        const bt = record[name]!;
        const trailing = bt.trailing.reduce((a, x) => a * x, 1);
        for (let i = 0; i < n; i++) {
          const part = bt.data.subarray(bt.offsets[i]!, bt.offsets[i + 1]!);
          const data = decodeArray(bt.dtype, part);
          outputs[start + i]![name] = { data, shape: [trailing ? data.length / trailing : 0, ...bt.trailing] };
        }
      }
    } else if (f === 3) {
      let key = "";
      let value = 0;
      fields(r.bytes(), (g, _w2, s) => {
        if (g === 1) key = new TextDecoder().decode(s.bytes());
        else if (g === 2) value = s.double();
      });
      metrics[key] = value;
    }
  });
  return { outputs, metrics, type: type.split(":").at(-1) ?? type };
}
