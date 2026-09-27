# Prefill candidate C: f16 operands and Chrome subgroup-matrix (tensor-core) GEMM

Branch `prefill/f16-subgroup` (from `prefill/base` 4c9a991). Everything here is **off by default**. With the
default `prefillMath: "f32"`, the engine builds the same shader module byte for byte, the same
pipelines, bind groups and dispatches, and the same state signature as before.

Status: implemented and checked on the CPU only. No GPU was used on this branch, so there are **no
measured speedups yet**. The commands under "GPU validation" produce the numbers.

## What the profile says about C

- **f16 as an ALU type is worthless here.** `vec4<f16>` FMA measured 8.6 TFLOPS against 16.7 for f32 in
  Deno/wgpu. The "f16" mode therefore keeps f32 FMAs and only rounds the operands to f16. Its job is to
  measure the numerics, not to be faster, and it is slightly slower (pack/unpack per dequantized vec4).
- **Tensor cores are the only part of C that can pay off, and only in Chrome.** Deno/wgpu does not expose
  subgroup-matrix. At the 16-column pass width, a 27B GEMM pass has to stream all ~15 GB of weights once.
  At ~270 GB/s that is ~3.7 ms per token, against 10.8 ms today. Tensor cores remove the arithmetic limit
  (the 27B GEMMs run at 3.0 to 5.8 TFLOPS against a 16.7 f32 peak), and the wall becomes weight bandwidth.
  Wider ubatches (candidate B) raise that ceiling.
- **The MoE gets nothing from C on this branch.** No MoE shape is in `GEMM_S`, so every MoE projection
  is a GEMV (candidate G1). Once G1 adds the MoE shapes, both modes cover them automatically, because
  they key on the engine's GEMM plan. Expert kernels stay GEMVs either way (candidate D).

## Modes (`prefillMath` engine option)

| mode | kernels | where | numerics |
|---|---|---|---|
| `f32` (default) | `gemm_q4/q8_{dIn}_s{S}` (unchanged) | everywhere | pinned, as before |
| `f16` | `gemm_*_r16` + `gemm_xpose_r16` (engine/wgsl/gemm.js `R16`) | any WebGPU (core WGSL) | weights `f16(f32(q)·d)`, activations `f16(x)`, f32 products and sums |
| `sgmatrix` | `gemm_sgm_q4/q8_{dIn}_s{S}` (engine/wgsl/gemm_sgm.js) | Chrome with `shader-f16` + `chromium-experimental-subgroup-matrix` | same operands as `f16`, f32 accumulation on tensor cores |

- **Selecting a mode.** Engine option `prefillMath`. Deno tests and tools read `PREFILL_MATH=f16|sgmatrix`
  (needs `--allow-env`, which every runner passes). In Chrome, `bench.html?prefillmath=...`, or
  `PREFILL_MATH=... node tests/bench/chrome_bench.mjs`. `engine.prefillMath` is also a runtime switch
  among the modes that were built, so a page can A/B against `"f32"` in one engine.
- **Only full-width prefill passes are affected.** The variants sit behind `op.gemm`, which `_dop`
  uses only when `nCols === NC` (16). Decode, verify and the prompt tail stay on the GEMV ladder, so
  spec == plain holds by construction, exactly as for the f32 GEMM (`maxDrafts = NC − 2`). The MTP
  draft-cache fill uses the same GEMM path and follows the mode.
- **Drop-in grid.** The tensor-core kernel uses the f32 GEMM's bindings (qs, sc, xT, partials, shape),
  split-K factors (`GEMM_S`, untouched), partials layout `p[split][col][dOut]` and fixed-order reduce.
  Its grid is `dOut / TM · S` (TM = 128 by default, the same as `GEMM_TILE`).
- **Fallback when unavailable.** `sgmatrix` falls back to the f32 GEMM, per engine and per op, and
  `engine.sgmWhy` records why. The causes are: a missing feature, no f16→f32 `subgroupMatrixConfigs`
  entry that tiles 128×16×32, workgroup memory over the limit, a compile error in its own shader
  module, a pipeline error, or a failed self-test. The self-test runs one Q4 and one Q8 kernel on
  random data against a float64 CPU reference with f16 operands, and requires rel L2 < 2e-3.
  Because the tensor-core module is separate, its compile failure cannot affect the main module.
- **State signature.** States made with a non-f32 mode carry `pm` in `stateSignature()`, so they are
  never restored into a session that prefills differently. f32 states keep the old signature.
- **Devices.** `prefillMathFeatures(adapter, mode)` (engine/qwen35.js) lists the features to request.
  `tests/load_model.js` and `tests/bench/bench.html` use it. room.js is not wired. Pinned numerics
  are a room protocol matter, so turning this on in rooms needs a GOVERNANCE decision.

## The tensor-core kernel (engine/wgsl/gemm_sgm.js)

- **Workgroups.** One workgroup is one subgroup, with `@workgroup_size` equal to the adapter's
  `subgroupMaxSize`. A workgroup is therefore never a partial subgroup. If the hardware runs smaller
  subgroups, each repeats the same MMAs and stores the same values: wasteful, but correct.
- **Per K stage** (KB Q4/Q8 blocks):
  1. Dequantize a TM × 32·KB weight tile to f16 in workgroup memory. The row stride is padded by PAD
     halves, so it stays a multiple of 16 bytes and the stores spread over the banks.
  2. Convert the 32·KB × 16 activation tile, in xT's own `[k][col]` order, to f16.
  3. Run (TM/M) × (32·KB/Kc) × (16/Nm) `subgroupMatrixMultiplyAccumulate` calls into f32 result
     matrices, which stay in registers for the whole K range.
- **Output.** Results are stored column-major (stride dOut) straight into the split-K partials.
- **Tuning.** Engine option `prefillSgm: {TM, KB, PAD}`, or `?sgm=JSON`. The default `{128, 1, 8}` needs
  8.7 KB of workgroup memory.
- **MMA shape.** Picked from `adapterInfo.subgroupMatrixConfigs`, largest first: 16×16×16 on
  NVIDIA/AMD Vulkan, 8×16×16 on Intel, 8×8×8 on Apple.
- **Builtin spelling.** Current Dawn writes `subgroupMatrixLoad<T, row_major>(p, off, stride)` and
  `subgroupMatrixStore<col_major>(…)`, which is how ONNX Runtime's WebGPU kernels use it. The first
  Chrome releases passed the majorness as a bool argument. The generator emits either spelling, and
  the engine tries the current one first.
- **CPU checks.** The index formulas are written once and shared by the WGSL generator and a
  JavaScript twin (`sgmGemmJS`). tests/unit/gemm_sgm_test.js runs the twin against float64 with f16
  operands for:
  - Q4_0 and Q8_0;
  - MMA 16×16×16, 8×8×8, 16×8×16 and wave64;
  - split-K 1 to 8;
  - KB 1 and 2, TM 64 to 256, PAD 0 to 16.

  Every partial is written exactly once and nothing outside them (max rel error < 1e-12). Mutating
  the load offsets, the xT stage offset or the nibble order makes the test fail.

## Exactness

- **Default path (`f32`).** It is unchanged:
  - `gemmWGSL(...)` without `R16` returns the identical string (checked against 4c9a991's generator).
  - No new pipelines are built, `_dop` picks the original op, and `requiredFeatures` is empty unless
    `PREFILL_MATH=sgmatrix`.

  Every golden (run.sh quick / q38, test_moe.js) therefore stays valid without a rerun, but please
  rerun them anyway (below).
- **`f16` / `sgmatrix`.** Both change prefill logits, so they stay off by default. The CPU estimate
  of f16 operand rounding on one GEMM output, against f32 operands, is **rel L2 2.0e-4 to 2.8e-4**
  (Q4 and Q8, with and without 50× outlier channels). That is ~50× the f32 GEMM-vs-GEMV difference
  (5e-6 gate), but under the 2e-3 prefill tolerance that `test_batch*.js` uses. Whether the 27B
  goldens (greedy MATCH vs llama.cpp) survive can only be measured. Measure it with `PREFILL_MATH=f16`
  in Deno: `f16` and `sgmatrix` have the same operands and differ only in f32 summation order. Only
  if every golden passes and the relDiff stays in tolerance could `sgmatrix` be considered for a
  per-device default. Even then it needs a room protocol decision.

## Expected gain (estimates, not measured)

- **27B at 512 tokens.** GEMMs are ~81% of prefill (10.8 of ~13.6 ms per token). If tensor cores bring
  them near the 16-column weight-bandwidth floor (~3.7 ms, realistically ~4.5 to 5.5 ms per token),
  prefill goes from **73 to ~115-135 tok/s** (1.6 to 1.85×). At 16k, attention dominates:
  **39 to ~48 tok/s** unless E lands.
- **Uncertainty.** This is the largest single lever the report found for the 27B, and the least
  certain. Whether Dawn's cooperative-matrix lowering and one-subgroup workgroups get anywhere near
  the peak is unknown, so run `tests/bench/sgm_gemm.mjs` first.
- **MoE.** 0% on this branch. After G1, the dense projections are ~0.98 ms per token, perhaps
  ~0.4 with sgm: about +10%.
- **`f16` mode.** No speed gain, and likely a few % slower. It is a numerics probe.

## GPU validation (run when the GPU is free)

```sh
cd /home/nehanth/wt/f16-subgroup
# 1. default path unchanged
tests/run.sh quick && tests/run.sh q38
(cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write test_moe.js)
# 2. f16 operand kernels vs GEMV and vs an f16-operand CPU reference (expect ~1e-4..1e-3 and < 2e-5)
(cd tests && PREFILL_MATH=f16 deno run --unstable-webgpu --allow-read --allow-env --allow-write test_gemm.js)
# 3. does f16-operand prefill keep the goldens? (16-column passes: test_ctx, and test_batch_q38 / test_mtp with BCOLS=16)
PREFILL_MATH=f16 tests/run.sh q38
(cd tests && PREFILL_MATH=f16 BCOLS=16 ROWSB=1 deno run --unstable-webgpu --allow-read --allow-env --allow-write test_batch_q38.js)
(cd tests && PREFILL_MATH=f16 BCOLS=16 ROWSB=1 deno run --unstable-webgpu --allow-read --allow-env --allow-write test_mtp.js)
# 4. tensor cores in Chrome: kernels alone (TFLOPS + relDiff vs f32 GEMM), then the whole 27B prefill A/B
node tests/bench/sgm_gemm.mjs 'reps=20&q8=1'
node tests/bench/sgm_gemm.mjs 'reps=20&q8=1&sgm={"TM":64,"KB":2,"PAD":8}'
PREFILL_MATH=sgmatrix PREFILL=512 node tests/bench/chrome_bench.mjs models/q38/model.gguf 40 batchcols=16
PREFILL_MATH=f16 PREFILL=512 node tests/bench/chrome_bench.mjs models/q38/model.gguf 40 batchcols=16
# 5. per-kernel profile with f16 operands (Deno; sgmatrix does not exist there)
(cd tests && MODEL=27b PREFILL_MATH=f16 deno run --unstable-webgpu --allow-read --allow-env --allow-write prof_prefill.js)
```

- **What to look for in step 4.** chrome_bench's page logs the adapter's `subgroupMatrixConfigs`, the
  mode the engine actually runs, and `sgmWhy` if it fell back. It then prints prefill tok/s for f32
  and the chosen mode in the same tab, the next-token logits' relDiff, and whether greedy argmax agrees.
- **If Chrome lacks the feature.** Playwright's bundled Chromium here (build 1148, about Chrome 131)
  may be too old for `chromium-experimental-subgroup-matrix`. In that case the page says so and
  everything runs f32. Use a current Chrome (for example with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`
  or `channel: "chrome"`).

## Risks

- **Nothing has been compiled on a real GPU.** The WGSL of both new kernel families has only been
  generated and checked structurally.
  - `f16`: a compile error would break engine creation in that mode only. The default module does
    not contain the `f16` kernels.
  - `sgmatrix`: a compile error falls back to f32 on its own.
- **The subgroup-matrix extension is experimental,** and its spelling has changed once. Its
  uniformity rules and workgroup-size rules may reject one-subgroup workgroups on some drivers
  (fallback). Performance with a 32-thread workgroup that dequantizes 128 rows per stage is unknown.
  An ORT-style design, with 4 subgroups per workgroup and a prepacked A, may be needed.
- **f16 range.** Activations above 65504 become inf, and the result is inf/NaN. None is expected at
  the GEMM inputs (normed x, SiLU·up, gated, attnOut), and the numbers above include 50× outliers.
  An extreme-activation model would still need a range check.
- **The self-test covers one Q4 shape and one Q8 shape, not every one.** Every shape comes from the
  same generator, and the CPU twin covers the index math over all tilings.
- **Rooms.** A peer that prefills with `f16`/`sgmatrix` produces different hidden states. Keep these
  modes single-device until GOVERNANCE decides.
