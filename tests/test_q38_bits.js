// 27B bit-exactness fingerprint: hashes of every logit of a batched prefill (16 columns, as the room),
// plain greedy decode and speculative steps. Run on two branches: equal hashes = bit-identical outputs.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_q38_bits.js
// ATTN_PREFILL_TILE=0 (read by load_model.js) turns the tiled prefill attention off, the one accepted deviation.
// The prompt is a frozen fixture (golden/q38_bits_prompt.txt, the first 150 lines of engine/gguf.js as of
// kopt/combined). It used to be read live from engine/gguf.js, so a comment edit there changed the prompt and
// the hashes. Reference (GB10): ATTN_PREFILL_TILE=0 -> BITS plain 85b12667 hidden eba0b8d5; default -> 8a532ef5 / 52f2ae10.
import { Qwen35Engine } from "../engine/qwen35.js";
import { argmax } from "../engine/engine.js";
import { openGGUF, Q38_PATH, gpuDevice } from "./load_model.js";

const model = openGGUF(Q38_PATH);
const { device } = await gpuDevice();
const L = model.trunkLayers, tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
const eng = await Qwen35Engine.create({ device, meta: model.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 2048, batchCols: 16, coopRowsB: 1 });
const h = (arrs) => { let x = 0x811c9dc5; for (const a of arrs) { const u = new Uint32Array(a.buffer, a.byteOffset, a.length); for (let i = 0; i < u.length; i++) x = Math.imul(x ^ u[i], 0x01000193) >>> 0; } return x.toString(16); };
const ids = tok.encode(await Deno.readTextFile(new URL("./golden/q38_bits_prompt.txt", import.meta.url))).slice(0, 300);
if (ids.length < 300) throw new Error(`prompt fixture too short: ${ids.length} tokens`);
// plain: batched prefill + 12 greedy tokens
eng.reset(); if (eng.mtp) eng.mtpFill = true;
await eng.prefillTokens(ids.slice(0, -1));
let lg = await eng.forwardToken(ids.at(-1));
const plainL = [lg], plain = [argmax(lg)];
for (let i = 0; i < 12; i++) { lg = await eng.forwardToken(plain.at(-1)); plainL.push(lg); plain.push(argmax(lg)); }
console.log(`plain: ${JSON.stringify(tok.decode(plain))} logits hash ${h(plainL)}`);
// speculative from the same prompt
eng.reset(); eng.mtpFill = true;
await eng.prefillTokens(ids.slice(0, -1));
lg = await eng.forwardToken(ids.at(-1));
const sp = [argmax(lg)];
while (sp.length < 13) { for (const t of await eng.specStep(sp.at(-1), argmax, 3)) sp.push(t); }
console.log(`spec : ${JSON.stringify(tok.decode(sp.slice(0, 13)))} ${sp.slice(0, 13).every((t, i) => t === plain[i]) ? "== plain" : "DIFFERS from plain"} tokens ${sp.slice(0, 13).join(",")}`);
// the trunk hidden after all of it (spec state included)
const x = await eng._readback(eng.x, eng.stageX, eng.dims.dim);
console.log(`BITS plain ${h(plainL)} hidden ${h([x])}`);
Deno.exit(0);
