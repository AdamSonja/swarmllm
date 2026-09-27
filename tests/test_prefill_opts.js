// Every prefill option together vs the pre-2026-09 prefill path (all off), same engine, same tokens:
// tiled prefill attention (attnPrefillTile), wide prefill GEMM (prefillUbatch) and, on the MoE, the
// expert-grouped tiled kernels (moeGroupPrefill; inside the wide chunks when both are on). Runtime
// switches: engine.attnPrefillTile, engine.prefillWide, engine.moeGroup.
//   * logits of the token after the prompt: relDiff vs the all-off path and vs one-token-at-a-time
//     (SEQ_ALL=1: at every length, else the shortest), argmax;
//   * greedy continuation of GEN tokens vs the all-off path (reported);
//   * speculative decoding after an all-on prefill: identical to plain decoding after the same prefill;
//   * prefill tok/s both ways (single runs, second of two when REPS=2).
//   MODEL=27b|moe  LENS=150,700,2100  GEN=24  PREFILL_UBATCH=256  MOEGROUP=256  (ATTN_PREFILL_TILE is forced on)
//   cd tests && MODEL=moe deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_prefill_opts.js
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q38_PATH, wideOpts } from "./load_model.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "27b"), GEN = +env("GEN", 24), K = +env("K", 3), REPS = +env("REPS", 1);
if (!Deno.env.get("PREFILL_UBATCH")) Deno.env.set("PREFILL_UBATCH", "256");
const LENS = env("LENS", "150,700,2100").split(",").map(Number);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const G = model.G, L = trunkLayers(G), nBlk = G.meta["qwen35.block_count"];
const hasMtp = L < nBlk;
const tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const maxSeq = Math.ceil((Math.max(...LENS) + GEN + 64) / 256) * 256;
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, batchCols: 16, coopRowsB: 1,
  attnPrefillTile: true, ...wideOpts(), ...(MODEL === "moe" ? { moeGroupPrefill: +env("MOEGROUP", 256), moeGroupUC: +env("MOEGROUP_UC", 8) } : {}) });
const set = (on) => { eng.attnPrefillTile = on && !!eng.attnPTCfg; eng.prefillWide = on && eng.ubatch > 0; eng.moeGroup = on && eng.moeGrpU > 0; };
console.log(`${MODEL}: ${L} layers, mtp ${!!eng.mtp}, attnPrefillTile ${!!eng.attnPTCfg}, ubatch ${eng.ubatch}, moeGroupPrefill ${eng.moeGrpU || "off"}${eng.moeGrpU ? ` UC ${eng.moeGrpUC} tiled ${eng.moeGrpTiled}` : ""}`);

let src = [];
for (const f of ["../engine/qwen35.js", "../engine/gguf.js", "../harness/agent.js"]) src.push(...tok.encode(await Deno.readTextFile(new URL(f, import.meta.url))));
const V = tok.vocab;
const prompt = (n) => {
  const tail = [V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n")];
  const head = [V["<|im_start|>"], ...tok.encode("user\nSummarize what this code does:\n")];
  return [...head, ...src.slice(0, n - head.length - tail.length), ...tail];
};
const rel = (a, b) => { let md = 0, sc = 1e-6; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); sc = Math.max(sc, Math.abs(b[i])); } return md / sc; };

async function run(ids, on, spec = false) {
  let out;
  for (let r = 0; r < (spec ? 1 : REPS); r++) {
    eng.reset(); set(on); if (eng.mtp) eng.mtpFill = spec;
    const t0 = performance.now();
    await eng.prefillTokens(ids.slice(0, -1));
    let lg = Float32Array.from(await eng.forwardToken(ids.at(-1)));
    const s = (performance.now() - t0) / 1000;
    const first = lg;
    let next = argmax(lg); const gen = [next];
    if (spec && eng.mtp) { while (gen.length < GEN) { for (const t of await eng.specStep(next, argmax, K)) gen.push(t); next = gen.at(-1); } gen.length = GEN; }
    else for (let i = 1; i < GEN; i++) { lg = await eng.forwardToken(next); next = argmax(lg); gen.push(next); }
    out = { lg: first, gen, tokps: ids.length / s };
  }
  return out;
}

let fail = 0, maxRel = 0;
for (const n of LENS) {
  const ids = prompt(n);
  const d = await run(ids, false), w = await run(ids, true);
  const r = rel(w.lg, d.lg); maxRel = Math.max(maxRel, r);
  const sameGen = d.gen.every((t, i) => t === w.gen[i]);
  let line = `${n} tokens: prefill all-off ${d.tokps.toFixed(1)} tok/s, all-on ${w.tokps.toFixed(1)} tok/s (${(w.tokps / d.tokps).toFixed(2)}x) · logits relDiff on vs off ${r.toExponential(2)} · argmax ${argmax(d.lg)} / ${argmax(w.lg)} · greedy ${GEN} ${sameGen ? "identical" : "DIFFERS"}`;
  if (argmax(d.lg) !== argmax(w.lg)) fail++;
  if (!sameGen) line += `\n  off: ${JSON.stringify(tok.decode(d.gen))}\n  on:  ${JSON.stringify(tok.decode(w.gen))}`;
  if (eng.mtp) {
    const s = await run(ids, true, true), same = s.gen.every((t, i) => t === w.gen[i]);
    line += ` · spec after all-on prefill ${same ? "identical to plain" : "DIFFERS from plain"}`;
    if (!same) fail++;
  }
  console.log(line);
  if (n === Math.min(...LENS) || env("SEQ_ALL", "") === "1") {
    eng.reset(); set(false); if (eng.mtp) eng.mtpFill = false;
    let lg = null; for (const id of ids) lg = await eng.forwardToken(id);
    const seq = Float32Array.from(lg);
    console.log(`  vs one-token-at-a-time: all-off prefill ${rel(d.lg, seq).toExponential(2)}, all-on prefill ${rel(w.lg, seq).toExponential(2)}`);
  }
}
set(true);
console.log(`max logits relDiff all-on vs all-off ${maxRel.toExponential(2)} (2e-3 is the dense prefill tolerance), GPU errors ${errors.count}`);
console.log(fail || errors.count ? "PREFILL OPTS FAIL (argmax / spec / GPU errors)" : "PREFILL OPTS PASS (argmax, spec == plain; relDiff reported)");
Deno.exit(fail || errors.count ? 1 : 0);
