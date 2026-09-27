# exp/gpu-sample: sample on the GPU (argmax and top-k for every head)

Branch `exp/gpu-sample` from `exp/base`. Status: code complete and checked on the CPU (a CPU model of the
kernels, the kernels themselves on lavapipe, the whole engine on lavapipe with the synthetic dense and MoE
models). **Not yet run on the GB10**: the GPU suites, the Chrome profile and the room profile below still
have to be run, and the results table filled in. Both options are **off by default** until then.

## Why

Every head copies the full 248,320-float logits vector to the CPU, only to pick one id (greedy) or draw
from the top 40 (the temperature presets). On the GB10 in Chrome, the map latency grows with the bytes
(`docs/research/exp-moe-network-profile.md`): 0.93 ms for 4 B, 3.25 ms for 1 MB. That is 1 MB per plain
token and 4 to 8 MB per speculative verify. The profile's item 1 estimates about 2 ms per plain token and
3.5 ms per speculative step, solo and on the room host. Separately, the draft chain's argmax is one
workgroup scanning the whole vocabulary at 490 µs per draft on the full head.

## What changed

| where | change |
|---|---|
| `engine/wgsl/qwen35.js` | `topk_a` / `topk_b`: a two-stage argmax / top-k over the columns of a logits matrix, for k ≤ 64. Stage a runs on a grid of (ceil(n/4096), columns). Each 256-thread workgroup holds a 4096-logit slice in registers (16 per thread). It runs k selection rounds, and round r takes the best element that comes after round r-1's pick in the order (value desc, index asc), so no exclusion list is needed. Stage b runs one workgroup per column and does k rounds over the nw·k candidates. Output per column: k × [idx, bits(value)] then [nonFiniteCount, 0]. At k = 1 that is the old argmax's 16-byte `[idx, bits, …]` layout, so `emb_gather` and the host read it unchanged. NaN and -Inf are never picked. Every non-finite value is counted (bit test on the exponent, which a fast-math compiler cannot fold away). A column with no finite value gives index 0, which is what the host's greedy returns. The same column-stride uniform serves the 1-column `this.logits` and the NC-column `B.logits`. |
| `engine/topk.js` (new) | `topkK`, `readCands` (result → `{ ids, vals, bad }`), and a thread-by-thread CPU model of both stages (`topkTwoStage`) plus a plain-sort reference (`topkNaive`). |
| `engine/qwen35.js` | Options `gpuSample` (default false) and `argmaxWide` (default false). Buffers: `tkPart` (the stage-a partials), `topBuf`, `stageTop`, `stageTopN` (+128 B tail for the drafts). New: `gpuDescFor(sample)`, `headFromHiddenIds(x, desc)`, `forwardTokenIds(id, desc)`, `headBatchIds(hs, n, desc)`, and `_verifyFusedIds` (`_verifyFused` with a `desc`). With a descriptor, the fused step's map shrinks from n·vocab·4 + 128 B to n·(8k+8) + 128 B. `verifyN(…, desc)`, `specStep` and `specStepDrafts` take the descriptor from `sample.gpu`. The acceptance loop is unchanged: `sample(lgs[k])`, where the sampler reads candidates. `_dArgmax` replaces the 4 single-workgroup argmax dispatches (`_mtpRun`, `_encodeDraftChain`, `_mtpRefill`, `_preDraft0`) when `argmaxWide` is on. |
| `room/sampling.js` | `pickSampler` returns a function with `.gpu = { kind: "greedy" }` or `{ kind: "topk", k, temp }`. The function reads either logits or `{ ids, vals }`. `aiSampleTop(cands, temp)` does the same exp / softmax / `Math.random` walk as `aiSample` over the sorted pairs. A masking wrapper (the tool-name constraint in `harness/model-common.js`) is a new function without `.gpu`, so it keeps getting full logits. |
| `room.js` | `?gpusample=1` turns on `gpuSample`, and `argmaxWide` with it (`?argmaxwide=0/1` overrides). `roomGenerate` takes `desc = engine.gpuDescFor(sample)`. `aiPipeToken` and the prefill tail then call `headFromHiddenIds` and throw the same NaN error when `bad > 0`. The spec path passes the sampler through `specStep`, and the verify goes through `verifyN(…, desc)` on a chain. `ai.pending` (Continue) stores `sample(cands)`, the chosen id. |
| tests | `tests/unit/topk_test.js` (CPU). `tests/e2e/topk_kernel.mjs` (the kernels vs the CPU model, bit for bit, on any adapter). `tests/e2e/gpusample_synth.mjs` (the whole engine on the synthetic models). `GPU_SAMPLE=1` / `ARGMAX_WIDE=1` in `tests/test_moe.js` and `tests/test_mtp.js`, which also run the head check on 256 random hiddens (`tests/gpusample_check.js`). `?gpusample=1` in `tests/bench/prof.html` / `prof_chrome.mjs`, `GPU_SAMPLE=1` in `tests/prof/prof_moe_decode.js`, and `--query gpusample=1` in `tests/e2e/room_prof.mjs`. |

## Checked so far (CPU only)

- `deno test --allow-read tests/unit/topk_test.js`: 10 tests pass. The two-stage model equals a full sort at vocab 248,320 / 151,936 / 65,536 and k = 1, 20, 40, 64. It also equals the sort for n = 1 … 65,537 not a multiple of 4096, for ties (few distinct values, ties across slices), for NaN / ±Inf (never picked / picked, counted), for all -Inf, for fewer than k finite values, for strided columns with NaN padding, and for the 65,536-row draft head. k = 1 equals `greedy()` and `engine/sampling.js argmax`. `aiSampleTop(top-k)` equals `aiSample(logits)` for the same `Math.random()` at t 0.4 / 0.8 / 1.5. All 176 unit tests pass.
- `tests/e2e/topk_kernel.mjs` on lavapipe (llvmpipe, CPU Vulkan): 251 cases pass. The WGSL compiles, and the kernels equal the CPU model bit for bit, including the old single-workgroup argmax at k = 1.
- `tests/e2e/gpusample_synth.mjs` on lavapipe, synthetic dense and MoE: all checks pass.
  - `headFromHiddenIds` equals `argmax(headFromHidden)` on 256 hiddens, and top-40 equals the sorted logits.
  - `forwardTokenIds` equals plain decoding.
  - `specStep` with a `.gpu` greedy sampler equals plain on four paths: fused, runTrunk (verifyN → headBatchIds), separate submits, and `specStepDrafts`.
  - `argmaxWide` on vs off gives the same draft ids, with the full head and with a 128-row draft head.
  - No validation errors.

## To run on the GB10 (GPU queue rule first)

```
cd /home/nehanth/wt/exp-gpu-sample
# kernels vs CPU model on the real GPU (seconds)
deno run --unstable-webgpu --allow-read --allow-env tests/e2e/topk_kernel.mjs
# engine on the synthetic models (about a minute)
deno run --unstable-webgpu --allow-read --allow-env tests/e2e/gpusample_synth.mjs
# unchanged defaults still pass
tests/run.sh quick && tests/run.sh q38 && (cd tests && deno run --unstable-webgpu --allow-read --allow-env test_moe.js)
# with GPU sampling on: MoE matches llama.cpp, spec == plain, head check on 256 hiddens
(cd tests && GPU_SAMPLE=1 deno run --unstable-webgpu --allow-read --allow-env test_moe.js)
(cd tests && GPU_SAMPLE=1 deno run --unstable-webgpu --allow-read --allow-env test_mtp.js)      # 27B
(cd tests && ARGMAX_WIDE=1 deno run --unstable-webgpu --allow-read --allow-env test_moe.js)     # wide draft argmax alone
# room: split == solo with GPU sampling (real GPU: SwiftShader has no adapter on the GB10)
E2E_GPU=real node tests/e2e/room_synth.mjs --compare --devices 2 --query gpusample=1
E2E_GPU=real node tests/e2e/room_synth.mjs --compare --devices 3 --query gpusample=1
# measure (Chrome): baseline and GPU sampling, same build
node tests/prof/prof_chrome.mjs models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf /tmp/gs_off.json
node tests/prof/prof_chrome.mjs models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf /tmp/gs_on.json gpusample=1
node tests/e2e/room_prof.mjs --model qwen3.6-35b-moe --devices 2 --maxnew 48 --out /tmp/room2_on.json --query gpusample=1
node tests/e2e/room_prof.mjs --model qwen3.6-35b-moe --devices 3 --maxnew 48 --out /tmp/room3_on.json --query gpusample=1
```

If all of these pass, turn `gpuSample` and `argmaxWide` on by default (engine defaults, `room.js`
`GPU_SAMPLE`, `prof.html`).

## Results (fill in after the GPU run)

Baselines are from `exp-moe-network-profile.md` (Chrome, GB10, 35B-A3B, two-sum prompt, greedy).

| | exp/base | gpusample=1 | expected |
|---|---|---|---|
| solo plain, ms/token (tok/s) | 22.1 (45.2) | | ~20 (~49): the 1 MB map drops to a 16 B one |
| solo spec K=3, ms/step (tok/s) | 48.6 (75.5) | | ~44 (~83): the 4 MB map drops to 64 B, and the full-head draft argmax drops from 3 × 490 µs to about 3 × 40 µs (argmaxWide) |
| logits readback in the spec step's timeline, GPU idle between submits | 6.4 ms/step | | ~2 to 3 ms |
| room 2 devices plain / spec tok/s | see profile | | head outside the lap −2.4 ms per plain token; verify head map −1.5 to 2 ms, and no CPU argmax (−2.5 ms) per spec step |
| room 3 devices plain / spec tok/s | see profile | | as for 2 devices |
| kernel cost: topk_a + topk_b, k = 1 / 40, per column | 490 µs (old argmax, full vocab) | | tens of µs (61 workgroups; k rounds of ~10 barriers each) |

## Risks

- Top-k tie order: the kernels order exact ties by index. `aiSample`'s candidate table keeps no index order on ties, so a temperature draw can differ from the CPU path only when two logits are exactly equal at the cut. Greedy is exact: same rule as `greedy()`.
- Spec path NaN check: the room never checked the spec verify's logits for NaN, and still does not. `bad` is available on every candidates object if wanted.
- An all-(-Inf / NaN) column gives id 0 (like `greedy()`); the old argmax kernel gave 0xffffffff. That only happens on a broken head.
- API surface: `verifyN`, `_verifyFused`, `specStep` and `specStepDrafts` gained an optional descriptor. Callers without `.gpu` on their sampler (every test, the harness engine model, masked samplers) take the unchanged logits path. `_refillDrafts` still reads the hiddens from `B.x`, which is unchanged.
- WGSL features: workgroup `atomic<u32>` (first use in this codebase) for the non-finite count. Naga (Deno) and lavapipe accept it. Tint (Chrome) is untested until the room and Chrome runs.
- No overlap with `kopt/head-rows` (that changes the head GEMV; this changes what follows it) or the prefill branches.
