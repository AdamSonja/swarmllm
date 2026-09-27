// attnPrefillTile A/B on a real model: the same long prompt prefilled with attn_flash(_t2) and with the
// tiled prefill attention (engine/wgsl/attn_tile.js), on one engine (engine.attnPrefillTile toggled at
// runtime). Compares the last prompt token's logits (relDiff = max |diff| / max |logit|, the
// tests/test_batch_q38.js measure and tolerance), the argmax, the greedy continuation, and the
// prefill time. Decode never uses the tiled kernel, so only the prefill numerics differ.
//   MODEL=27b|moe LENS=700,3000 CTX=4096 TOKENS=16 ATTN_PREFILL_TK=0|4|8|16
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_attn_tile.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, Q38_PATH, MOE_PATH } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), CTX = +env("CTX", 4096), N = +env("TOKENS", 16);
const LENS = env("LENS", "700,3000").split(",").map(Number).filter((n) => n + N + 2 <= CTX);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "27b" ? Q38_PATH : MOE_PATH);
const G = model.G, arch = G.meta["general.architecture"], nBlk = G.meta[arch + ".block_count"];
const hasMtp = Object.keys(G.tensors).some((k) => k.startsWith(`blk.${nBlk - 1}.`));
const L = trunkLayers(G), tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true,
  maxSeq: CTX, batchCols: 16, coopRowsB: 1, attnPrefillTile: true });
if (!eng.attnPrefillTile) { console.log("ATTN TILE FAIL: the tiled kernel is not available (see the warning above)"); Deno.exit(1); }
console.log(`${MODEL}: maxSeq ${CTX}, faSplit ${eng.faSplit} x ${eng.faSplits}, tile ${JSON.stringify(eng.attnPTCfg)}`);

// a realistic coding prompt: this repo's own source
let ids = [];
for (const f of ["../engine/qwen35.js", "../engine/gguf.js", "../engine/wgsl/base.js", "../room.js"]) {
  ids.push(...tok.encode(`\n// file: ${f}\n` + await Deno.readTextFile(new URL(f, import.meta.url))));
  if (ids.length >= Math.max(...LENS)) break;
}
const run = async (tile, prompt) => {
  eng.attnPrefillTile = tile;
  eng.reset(); if (eng.mtp) eng.mtpFill = true;
  const t0 = performance.now();
  await eng.prefillTokens(prompt.slice(0, -1));
  let lg = await eng.forwardToken(prompt[prompt.length - 1]);
  const s = (performance.now() - t0) / 1000;
  const first = lg.slice(), gen = [argmax(lg)];
  for (let i = 1; i < N; i++) { lg = await eng.forwardToken(gen[i - 1]); gen.push(argmax(lg)); }
  return { first, gen, s };
};
let fail = 0;
for (const len of LENS) {
  const prompt = ids.slice(0, len);
  const a = await run(false, prompt), b = await run(true, prompt);
  let md = 0, sc = 1e-6;
  for (let i = 0; i < a.first.length; i++) { md = Math.max(md, Math.abs(a.first[i] - b.first[i])); sc = Math.max(sc, Math.abs(a.first[i])); }
  const rel = md / sc;
  let same = 0; while (same < N && a.gen[same] === b.gen[same]) same++;
  const ok = argmax(a.first) === argmax(b.first) && rel < 2e-3 && Number.isFinite(rel);
  if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"} ${len} tokens: relDiff ${rel.toExponential(2)}, argmax ${argmax(a.first)} / ${argmax(b.first)}, greedy ${same}/${N} identical; ` +
    `prefill ${(len / a.s).toFixed(1)} -> ${(len / b.s).toFixed(1)} tok/s (attn_flash -> tile, one run each, includes the first decode step)`);
}
if (errors.count) fail++;
console.log(fail ? "ATTN TILE FAIL" : "ATTN TILE PASS");
Deno.exit(fail ? 1 : 0);
