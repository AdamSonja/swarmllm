# Bench log

Every kernel/engine change gets a row. All 27B numbers are Qwen 3.8 27B Q4_0,
greedy, bit-identical output verified by the test suite (`tests/test_mtp.js`,
`tests/test_batch*.js`). GB10 = DGX Spark via Deno/wgpu (Vulkan, no subgroups,
shader-f16 available). Mac = user's MacBook, Chrome, staging site.

| Date | Change | Commit | GB10 decode plain | GB10 decode spec K=3 | GB10 spec K=7 | GB10 prefill (batched) | Mac decode | Mac prefill | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Aug 29 | Launch state | main | 3.6 | — | — | 3.0 | 2.5 | ~3 | one thread per row, f32 scales |
| Aug 31 | Coop kernels + batched prefill + fusion + f16 scales | 71b7b85..9f8b852 | 9.0 | — | — | 15.2 (4-col) | 6.7 | 14–16 | |
| Sep 1 | Multi-column batched ops (2x batched passes) | f1a87a9 | 9.1 | — | — | 39.4 | | | |
| Sep 1 | MTP self-speculation | e8d5642, 15c389f | 9.07 | 15.86 (85% acc) | — | 39.4 | 10–10.8 | 12 s/prompt | cross-network 3.5–4 tok/s |
| Sep 1 | Adaptive deep speculation (K=3/5/7 by lap RTT) | 3461d10 | 8.72–9.14 | 16.07 (85%) | 13.09 (71%, 6.0 tok/lap) | 39.4 | | | K=7 only chosen when lap >260 ms |
| Sep 1 | GPU argmax for the draft chain (8-byte readback instead of 1 MB/draft) | — | 9.06 / 8.85 | 16.13 (85%) | 12.32 (71%) | | | | neutral on GB10 (unified memory); helps discrete GPUs |
| Sep 1 | 8-wide batched columns (BCOLS=8) — prefill | — | 8.7–9.0 | | | 27.3 (4w) vs 26.1 (8w×2r) vs 27.9 (8w×4r), 86-tok prompt | | | no gain: 8-col pass ≈ 4-col pass ⇒ prefill is bound by the serial DeltaNet recurrence, not GEMV |
| Sep 1 | **Native reference: llama.cpp CUDA (build 749f688), same GGUF** | — | **7.99** (tg32) | — | — | **377** (pp86) | | | WebGPU beats native on decode (9.0 plain); prefill 14x behind ⇒ parallel recurrence is the prize |
| Sep 1 | Deep spec with single 8-col verify (BCOLS=8) | e07c0e0 | 8.68 | | K=7: 14.86 (71%); K=5: 16.65 (97%, partly contaminated) | | | | K=7 never pays solo; K=5 is the candidate |
| Sep 1 | unpack4xU8/I8 dequant in all coop kernels (probe-gated fallback) | — | 8.65 | | | 26.0 | | | bit-exact; neutral on GB10 (driver already optimized the shifts) — kept for Metal/Android where ALU is scarcer |
| Sep 1 | Bandwidth probe: achievable streaming read on GB10 via WebGPU = 184 GB/s | — | | | | | | | decode GEMVs = 15 GB / 82 ms = 183 GB/s ⇒ AT roofline; remaining decode cost is ~30 ms of small-dispatch overhead |
| Sep 2 | Accumulate matvecs (y += W·x): residual adds folded in, −192 dispatches/token | — | 8.93 | 15.64 (85%) | | 25.8 | | | bit-exact; neutral on GB10 ⇒ small dispatches are ~free on Vulkan; kept for Metal |
| Sep 2 | Gates+L2 fused (dn_pre), room at 8 cols × 2 rows, 4-col twin kernels, 2-D dispatch for tall matvecs | — | 8.8–9.2 | 15.49 (85%, room cfg) | K=5: 16.02 (97%, room cfg); K=5 @ 8×4 rows: 17.00 | 42.8 (18-tok test) | | | bug fixed: LM head at 2 rows/WG = 124k workgroups > 65535 limit, dispatch silently dropped |
| Sep 2 | GEMM prototype for prefill (bench_gemm_deno.js): 16-col tiled Q4 GEMM, vec4 shared tiles, 2×4 thread tile | — | | | | 17408×5120 × 16 cols: gemm 1.83 ms vs coop_b 4×4-col 1.93 ms (parity) | | | correct (2.5e-7); only 27 GB/s / 1.5 TFLOPS ⇒ latency-bound (2 barriers × 160 k-blocks, loads exposed). Next: register prefetch of block b+1, 64-k steps, 2+ WGs/SM |
| Sep 2 | GEMM v3 (padded shared tile, 2 blocks/barrier, register prefetch, RT=2×4 cols) | — | | | | 17408×5120×16: 1.54–1.61 ms vs 1.93 (1.21–1.25×); 5120×5120: parity (occupancy: 80 WGs) | | | bank-conflict padding was the only lever that moved it; 4×4 tiles slower (occupancy). Still ~30 GB/s / 1.8 TFLOPS: next try split-K for small dOut, 256-thread WGs, check naga bounds-check cost |
| Sep 4 | **Prefill GEMM** (roadmap 02): row-stationary Q4_0 GEMM at 16 batch columns, split-K pinned per shape, `_dop` ladder GEMM→b8→b4 | — | 9.0–9.2 (unchanged) | 15.4–15.9 (unchanged, K=5 at 16 cols) | | pass-level 34.6 → **56.7** (1.64×); end-to-end `bench.js` 30.2 → **43.7** (1.45×) | | | ffn_down + ssm_out are Q8_0 ⇒ stay on the GEMV; Q8 variant is the next lever |
| Sep 4 | Width-dependent rows per workgroup for the b8/b4 twins | — | | 12.8 → 15.4 at 16 cols | | | | | fixed the decode regression the 16-column width introduced; twins byte-identical to native-width kernels (`tests/test_twins.js`) |

## Standard prompts

Room numbers are only comparable when the prompt is the same. Use these, verbatim, and name the prompt in the row. Token counts are for Qwen 3.8 27B's tokenizer, text only; the chat template adds about 9 tokens.

| Name | Tokens | Text |
|---|---|---|
| `hello` | 1 | `hello` |
| `meaning` | 6 | `what is the meaning of life` |
| `japan` | 160 | `I am planning a two week trip through Japan in late October with my partner. We land in Tokyo, want three days there, then a day trip to Nikko, then the bullet train to Kyoto for four days with a side trip to Nara, then two nights in Osaka, and we fly home from Osaka. We like food markets, old temples, hiking, and small neighborhood bars, and we want to avoid the most crowded tourist spots where we can. Our budget is moderate, around two hundred dollars a day for the two of us not counting hotels. Please give me a day by day itinerary with one main activity each morning and afternoon, a neighborhood to eat dinner in each night, and tell me which days I should buy a rail pass for and whether it is worth it at all.` |

`hello` and `meaning` measure fixed per-request overhead. `japan` is long enough that prefill runs through the 16-column GEMM path for most of its length; use it for any prefill claim. Decode numbers are for the 400-token answer cap; the first answer in a room is slower because the speculation depth starts conservative, so report the second answer or later.

## Room benchmarks by PR (real devices)

One row per merged PR that changes speed, measured in the room on real devices, before and after. "Before" is production (`main`) on the same day; "after" is the PR's preview URL. Prefill is the seconds the status line reports; decode is the tok/s in the answer footer.

| Date | PR | Devices | Prompt | Prefill before → after | Decode before → after | Notes |
|---|---|---|---|---|---|---|
| Sep 4 | #29 prefill GEMM | MacBook (Chrome, Metal) 62 layers + embed/head, iPhone 2 layers | `japan` | 13.8 s → **8.5 s** (1.62×) | 7.3 → 7.7 tok/s (unchanged, within noise) | first answer in each room; both runs hit the 512-token context overflow (#31) after ~300 generated tokens, which does not affect the prefill number |

## Hidden-state transport (data channel)

Measured 2026-09-04 on the GB10: two headless Chromium 131 tabs on one machine, loopback shaped with netem to a 100 ms round trip (50 ms each way), one-way delay of one message, p50 over 10 samples, 1 s apart, RTCDataChannel ordered+reliable unless noted. Harness: two RTCPeerConnections over host candidates, sender stamps `performance.timeOrigin + now` in a 16-byte header.

| message | what it is | plain, one send | sliced ≤4.6 KB sends | striped over 5 associations |
|---|---|---|---|---|
| 1 KB | token id, ping | 51 ms | 51 | 52 |
| 5 KB | | 153 | 51 | 51 |
| 10 KB | one token's hidden state (5120 × f16) | 152 | 51 | 52 |
| 20 KB | | 254 | 52 (p90 152) | 52 |
| 30 KB | K=3 verify block | 255 | 52 (p90 155) | 51 |
| 50 KB | K=5 verify block | 355 | 152 | 52 (p90 153) |
| 70 KB | K=7 verify block | 458 | 152 | 52 (p90 153) |
| 164 KB | 16-token prefill chunk | 562 (p90 664) | 153 | — |

Reading: 51 ms is the physical one-way time. On a plain channel every ~4 packets beyond the first burst cost one more round trip (dcSCTP `max_burst` 4, initial cwnd 10 MTU): a single token paid 1.5 round trips per hop, a K=5 verify block 3.5, a K=7 block 4.5, a prefill chunk 5.5. Slicing every send under four packets removes the burst penalty and fixes everything up to ~30 KB; blocks above the initial window still pay one round trip on one association, and striping over five associations removes that too.

With 1 % packet loss on the same link (12 samples): plain 10 KB p50 153 / p90 356 ms, plain 164 KB p50 1373 / p90 1979 ms; sliced 10 KB p50 152 / p90 253, sliced 70 KB p50 359 / p90 771. Loss recovery costs a round trip per event on a reliable channel, so forward error correction on an unordered channel is the next lever (issue #34, step 3).

Room change: `room/transport.js`, a negotiated data channel per peer link that PeerJS never sees, slicing at 4,600 bytes and round-robin over `?wire=stripeN` associations (default `stripe4`); `?wire=off` restores PeerJS messages. Exact by construction: bytes only. Unit test `tests/unit/transport_test.js` (byte-exact reassembly under reordering and duplicates).

End-to-end on the GB10 (two headless Chromium tabs, real PeerJS signaling and WebRTC on loopback, Qwen3 0.6B Q8 split 18+10 layers): `?wire=off` 41.4 / 42.9 tok/s, `?wire=stripe4` 53.5 / 52.0 tok/s, 4 channels open per link, 127 frames sent = 127 received each way, no console errors. The loopback gain is the PeerJS serializer and its 16 KB chunking leaving the path; the round-trip gain needs a real network and is the table above.

27B in the emulator (GB10, `npm run e2e -- --phone --model qwen3.8-27b`, host 53 layers + embed/head, worker 9, phone-shaped tab 2, `japan` prompt, 400-token answers, weights served from local disk, loopback network): `--wire off` prefill 4.3 / 4.1 s, decode 10.3 / 9.9 tok/s; `--wire stripe4` prefill 4.5 / 4.2 s, decode 10.4 / 10.7 tok/s. Equal within noise on loopback, as expected; 459 frames per link each way. Both runs trip bug #31 (prompt + 400 tokens > 512 context): 1,741 GPU validation lines in the room log, which the emulator now reports.

Topology change (host link + on-demand chain links instead of a full mesh) and `--devices N` in the emulator, GB10, 27B, `japan` prompt, local signaling, loopback: 3 devices online in 3.0 min, prefill 4.5 s, decode 10.4 tok/s; 16 devices (8 phone-shaped, 64 layers dealt 18+embed / 4-5 per worker / 2 per phone) online in 2.3 min, prefill 8.3 / 7.2 s, decode 3.8 / 4.7 tok/s, every device holding one host link and two chain links, no errors. The decode drop on a zero-latency network is per-hop processing (unpack, upload, readback, pack), about 15 ms per hop, now a measured target. 64 tabs in one Chromium fail at `vkCreateDevice` (one GPU process, driver device cap); not a room limit.

## 2026-09-26: Qwen3.6-35B-A3B MoE on real hardware (GB10), expert kernels rebuilt

File: bartowski `Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (shared experts Q5_0 → Q8 on load, routers BF16 → f32 exact).
Reference: llama.cpp b10840 CUDA on the same file: 85.5 tok/s decode, 2520 tok/s pp512 (27B: 13.8 decode).

Correctness: greedy output matches llama.cpp on three chat prompts (tests/test_moe.js); speculative decoding
identical to plain. Plain "The capital of France is" is a near tie after " Paris" ("." 19.029 vs "," 18.968 here;
llama.cpp CUDA picks ","), so it is not used as a golden.

Per-token GPU time (timestamp queries, tests/prof_ts.js), before → after rebuilding the MoE kernels on the
cooperative-GEMV layout (4 threads per block, vec4 x reuse across rows, unpack4x dequant) and a parallel router:

| kernel      | before (µs × 40) | after |
|-------------|------------------|-------|
| moe_gu_q4   | 288              | 71    |
| moe_dn_q4   | 169              | 43    |
| moe_router  | 165              | 32    |
| GPU total   | 41.7 ms          | 22.8 ms |

Decode tok/s, plain / speculative (K=3):

| | Deno (GB10) | Chrome (GB10) |
|---|---|---|
| MoE before | 17.8 / 9–20 | n/a |
| MoE after  | 27.0 / 15–27 | 32–40 / 52–59 |
| 27B (unchanged) | 9.7 / 15.9 | 9.4–10.8 / 18.7–22.6 |

Deno adds ~11.7 ms per GPU sync (an empty submit + 4-byte readback: 11.7 ms in Deno, 0.76 ms in Chrome), so
Chrome numbers (tests/bench/chrome_bench.mjs) are the ones to quote. CPU encode is 4.5 ms per MoE token (732 dispatches).

## 2026-09-26: combined optimizations (branch opt/combined), GB10

Merged onto work/tabby-gpu in this order: opt/spec-one-submit (one-submit speculative step, draft chain on),
opt/fuse-projections (merged projection GEMVs, bit-identical), opt/moe-fuse (fused router + shared expert +
combine), opt/moe-kernel-tuning (configurable unfused expert GEMV layout; left opt-in, legacy by default),
opt/drafts (batched draft-cache refill, pre-run first draft, draftvocab auto-fallback). With a fused verify the
refill reads the trunk hiddens in place from the batch buffer (no CPU copy); acceptance counts are identical
on every path (fused / separate verify, chain on / off).

Correctness: `tests/run.sh quick` and `tests/run.sh q38` pass with the same outputs, hidden values and acceptance
counts as the baseline (27B bits unchanged); `tests/test_moe.js` matches llama.cpp on all three prompts with
spec == plain, also with DRAFTCHAIN=0, SPECFUSE=0 and MOE_FUSE=0; e2e drafts_synth and draftchain_synth pass
on the real GPU (E2E_GPU=real). moe_synth's "batched prefill == one token" check still fails with 202 logits,
as it does on work/tabby-gpu (not caused by these merges, also with gemm off; open).

Chrome decode tok/s (tests/bench/chrome_bench.mjs, 40 tokens, K=3), plain / speculative. Baseline: today's
work/tabby-gpu numbers (one re-run in this session: MoE 33.77 / 59.17 and 38.86 / 49.79, inside the range).

| | two-sum | hash-map |
|---|---|---|
| MoE baseline | 33.4–33.8 / 57.4–59.1 | 37.5–38.9 / 50.5–51.2 |
| MoE combined (bench defaults, run 1, 2) | 40.48 / 74.33, 41.63 / 74.68 | 45.84 / 66.54, 45.19 / 64.73 |
| MoE combined, draftvocab=65536 (room default) | 44.73 / 82.82, 33.61* / 84.22 | 43.25 / 65.00, 44.77 / 64.82 |
| 27B baseline | 10.1–10.2 / 22.0–22.1 | 10.6–10.7 / 18.1–18.7 |
| 27B combined (bench defaults, run 1, 2) | 10.23 / 24.05, 10.06 / 24.13 | 10.69 / 20.39, 10.69 / 20.27 |
| 27B combined, draftvocab=65536 | 10.17 / 27.06, 10.08 / 26.78 | 10.81 / 21.07, 10.72 / 20.82 |

MoE: plain +20–24 % (two-sum) and +16–22 % (hash-map); speculative +26–30 % and +26–32 %. 27B: plain unchanged,
speculative +9 % and +8–13 %. Acceptance is unchanged (MoE 28/33, 28/39; 27B 28/33, 26/39). The draftvocab=65536
rows were not measured on the baseline, so they only compare against the combined defaults.
(*) one run with a stall in the first timed decode (see below).

MoE GPU time per token (tests/prof_ts.js, timestamp queries): 22.74 ms, 732 dispatches (baseline, same session)
→ 20.46–20.63 ms, 502 dispatches. Top kernels: matvec_q8_coop 3.08 ms (15), moe_gus_q4_q8 3.05 (40, 76 µs),
matvec_q4_coop 2.97 (36), moe_dnc_q4_q8 1.87 (35, 54 µs), dn_pre 1.32, dn_delta_gn 1.28, moe_route 1.25 (40),
matvec_coop 1.11 (70), rmsnorm 1.08 (81). Deno wall is sync-bound (~35 ms/token steady on both), but with
fuseProj + moeFuse together Deno shows a one-time 1.4–1.8 s stall a dozen tokens into decode (a V8 scavenge of
~1.5 s under external-memory pressure, `--trace-gc`), so prof_ts's 20-token wall average reads 92–128 ms. Neither
branch alone shows it; the cause is open.

## 2026-09-26: load time for tests and benches (CPU side; GPU not yet measured)

Loader CPU cost, with the GPU upload stubbed (`tests/bench/load_profile.js`): the 27B took 43.8 s (6.5 s reading, 37.2 s converting, 20.1 s of that the Q4_0 repack) and the MoE 46.0 s (8.4 s reading, 37.6 s converting). The repack is now 3x faster (u16 copies). With the new converted-weights cache (`tests/weight_cache.js`), a warm load is 4 to 6 s for the 27B and about 9 s for the MoE. The Chrome bench's static server read ranges at 0.27 GB/s; 8 MB reads bring that to 2 GB/s, and pre-converted tensors (`bench.html?wcache=1`) take the MoE tab load path from 74.7 s to 17 s on the CPU side. `tests/run_q38_once.js` runs the 27B suite over one upload. Details, and the commands still to run for GPU validation, are in [testing-fast.md](testing-fast.md).

## 2026-09-27: tiled prefill attention (attnPrefillTile, candidate E), on by default

`attn_flash_tile` (engine/wgsl/attn_tile.js) replaces attn_flash / attn_flash_t2 on full-width prefill passes: one
workgroup per (split, KV head, 64 query rows) shares each K/V tile across the whole pass. GB10, Deno,
`bench_ctx.js` CTX=16640 TOKENS=8, prefill tok/s over the segment ending at each fill, off → on:

| | 512 | 4096 | 16384 |
|---|---|---|---|
| 27B | 65.9 → 69.8 | 57.4 → 74.3 | 30.2 → 66.1 |
| MoE | 151.3 → 159.7 | 133.3 → 166.0 | 80.8 → 117.2 |

Whole 16000-token prompt: 27B 39.7 → 69.1, MoE 99.5 → 153.9. Attention kernel time at 4096 (`prof_prefill.js`):
27B 14.5 → 2.2 s, MoE 4.8 → 1.0 s. Decode never uses the kernel. Every golden passes with it on; logit relDiff vs
attn_flash and the MoE router caveat are in docs/research/prefill-profile-2026-09.md (candidate E).
`ATTN_PREFILL_TILE=0` / `?attnptile=0` restores attn_flash.
