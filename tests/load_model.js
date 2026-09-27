// Shared model loading for the Deno GPU tests and benchmarks.
//
// openGGUF(path): header + tokenizer + a weights() loader, reading with node:fs (about 2x the
// throughput of Deno.FsFile seek/read) through the converted-weights cache (tests/weight_cache.js;
// WEIGHT_CACHE=0 disables it). Fresh weights per call, exactly as each test loaded them before.
//
// sharedQ38(device): the one-process runner's model. Loads the whole 27B once (64 layers, embed,
// head, MTP block), uploads every matrix and norm to the GPU once, and hands out views of that
// set: weights({lo, hi, hasEmbed, hasHead, mtp}) returns the same structure qwen35Weights would,
// with each entry already carrying its GPU buffer (entry.gpu). Qwen35Engine.create uses entry.gpu
// as-is (the same hook the browser's streamed upload uses), so any number of engines, whole or
// partial, share one copy of the weights. The engine never writes to weight buffers.
import fs from "node:fs";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF, gpuUploadEntry } from "../engine/gguf.js";
import { makeTokenizer } from "../engine/engine.js";
import { attachWeightCache } from "./weight_cache.js";
import { prefillMathFeatures } from "../engine/qwen35.js";

export const Q38_PATH = new URL("../models/q38/model.gguf", import.meta.url).pathname;
export const MOE_PATH = new URL("../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf", import.meta.url).pathname;

// layers before the MTP ("nextn") block, when the file has one
export function trunkLayers(G) {
  const n = G.meta["qwen35.block_count"];
  return n - (G.meta["qwen35.nextn_predict_layers"] || (G.tensors[`blk.${n - 1}.nextn.eh_proj.weight`] ? 1 : 0));
}

export function openGGUF(path, { skipTokenizer = false, cache = true, headerBytes = 64 << 20 } = {}) {
  const fd = fs.openSync(path, "r");
  const readAt = (off, len) => {
    const out = new Uint8Array(len);
    let o = 0;
    while (o < len) { const n = fs.readSync(fd, out, o, Math.min(len - o, 1 << 30), off + o); if (n <= 0) break; o += n; }
    return out;
  };
  const G = parseGGUFHeader(readAt(0, headerBytes).buffer, { skipTokenizer });
  const wcache = cache ? attachWeightCache(G, path) : null;
  const bytesOf = (info) => readAt(info.byteOffset, info.byteLength);
  let tok = null;
  return {
    path, G, meta: G.meta, readAt, bytesOf, cache: wcache,
    trunkLayers: trunkLayers(G),
    tokenizer: () => (tok ||= makeTokenizer(tokenizerFromGGUF(G.meta))),
    weights: (range, onProgress, onEntry) => qwen35Weights(G, bytesOf, range, onProgress, onEntry),
    close: () => fs.closeSync(fd),
  };
}

export async function gpuDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  // PREFILL_MATH=sgmatrix asks for the tensor-core features where the adapter has them (Chrome only; none in Deno)
  const device = await adapter.requestDevice({ requiredFeatures: prefillMathFeatures(adapter), requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  return { adapter, device };
}

// Count (and print the first few) uncaptured GPU errors; tests read errors.count.
export function watchGpuErrors(device, print = 3) {
  const errors = { count: 0 };
  device.addEventListener?.("uncapturederror", (e) => { errors.count++; if (errors.count <= print) console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });
  return errors;
}

// Everything a q38 test needs. Standalone: fresh weights per weights() call (cached conversion).
export async function q38Context({ skipTokenizer = false } = {}) {
  const { adapter, device } = await gpuDevice();
  const model = openGGUF(Q38_PATH, { skipTokenizer });
  return { adapter, device, model, errors: watchGpuErrors(device), shared: false };
}

// Upload every quantized matrix and f32 tensor of a weight set once; engines then reuse entry.gpu.
// The embedding keeps its CPU copy (per-token row lookups) and is never uploaded here.
export function preuploadWeights(device, w) {
  let bytes = 0;
  const f32 = (e) => {
    const src = new Uint8Array(e.data.buffer, e.data.byteOffset, e.data.byteLength);
    const buf = device.createBuffer({ size: Math.ceil(src.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(src);
    buf.unmap();
    return { kind: "f32", buf };
  };
  const up = (e) => {
    if (!e || e.gpu) return;
    if (e.kind === "q4" || e.kind === "q8") { bytes += e.qs.byteLength + e.scales.byteLength; gpuUploadEntry(device, e, false); }
    else if (e.kind === "f32") { bytes += e.data.byteLength; e.gpu = f32(e); }   // keep e.data: some layers read it directly
  };
  const layer = (L) => { for (const v of Object.values(L)) if (v && typeof v === "object" && "kind" in v) up(v); };
  for (const L of w.layers) layer(L);
  up(w.finalNorm); up(w.head);
  if (w.mtp) { layer(w.mtp.layer); up(w.mtp.ehProj); up(w.mtp.enorm); up(w.mtp.hnorm); up(w.mtp.sharedHeadNorm); }
  return bytes;
}

// The one-process runner's context: the 27B loaded and uploaded once, shared by every engine.
export async function sharedQ38Context() {
  const { adapter, device } = await gpuDevice();
  const errors = watchGpuErrors(device);
  const model = openGGUF(Q38_PATH);
  const L = model.trunkLayers;
  let t0 = performance.now();
  const full = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
  const tRead = performance.now() - t0;
  t0 = performance.now();
  const up = preuploadWeights(device, full);
  await device.queue.onSubmittedWorkDone();
  const tUp = performance.now() - t0;
  console.log(`shared 27B: ${L} layers + head + mtp read/convert ${(tRead / 1000).toFixed(1)} s, GPU upload ${(up / 2 ** 30).toFixed(2)} GB in ${(tUp / 1000).toFixed(1)} s${model.cache ? "; " + model.cache.summary() : ""}`);
  // same structure as qwen35Weights(G, ..., range), minus the load
  model.weights = async ({ lo, hi, hasEmbed = false, hasHead = false, mtp = false }) => {
    if (lo < 0 || hi > L || lo >= hi) throw new Error(`shared weights hold layers 0..${L - 1}, asked ${lo}..${hi - 1}`);
    const out = { layers: full.layers.slice(lo, hi) };
    if (hasEmbed || hasHead) out.embed = full.embed;
    if (hasHead) { out.finalNorm = full.finalNorm; out.head = full.head; }
    if (mtp && hasHead && full.mtp) out.mtp = full.mtp;
    return out;
  };
  return { adapter, device, model, errors, shared: true };
}
