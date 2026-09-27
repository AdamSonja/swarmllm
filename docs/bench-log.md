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
| Sep 2 | GEMM prototype for prefill (benchmarks/bench_gemm.js): 16-col tiled Q4 GEMM, vec4 shared tiles, 2×4 thread tile | — | | | | 17408×5120 × 16 cols: gemm 1.83 ms vs coop_b 4×4-col 1.93 ms (parity) | | | correct (2.5e-7); only 27 GB/s / 1.5 TFLOPS ⇒ latency-bound (2 barriers × 160 k-blocks, loads exposed). Next: register prefetch of block b+1, 64-k steps, 2+ WGs/SM |
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

## Per-hop telemetry on a Mac and an iPhone (PR #51, @aaryanmanchanda)

Measured 2026-09-13 by @aaryanmanchanda on his own devices, from PR #51 (closed because per-hop telemetry landed on the main work branch in a different shape; the numbers are his). MacBook Air M1 8 GB (Brave) as host, iPhone 13 Safari as worker with a 0.5 GB pledge, same Wi-Fi, local HTTPS. Qwen3 0.6B Q8, default wire (`?wire=stripe4`), `japan` prompt, second answer in the room. Generation: 400 tokens at 12.4 tok/s on 2 devices.

| | p50 | p90 |
|---|---|---|
| iPhone compute per hop | 30 ms | 35 ms |
| Derived transport (lap minus compute minus host pack and compute) | 10 ms | 64.2 ms |

His run also found two bugs that the main branch fixed too: the striped wire's fixed slice header dropped per-hop data, and the lap timer started after the host's own work, so derived transport read 0 on every lap.

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

## 2026-09-27: Emulated latency, 1 to 3 devices (branch bench/latency), GB10

What a room feels like when the devices are not on the same machine. Harness: `tests/e2e/room_latency.mjs`.
Each device is its own headless Chromium 131 (own profile, own GPU process, weight cache on disk), real PeerJS
signaling and real WebRTC over loopback, `?wire=stripe4`, all devices on the one GB10 GPU. Latency is added in
the page, not by netem (that needs root, which this machine does not give us): every activation frame a device
sends is held for the one-way delay before it goes on the wire, the same thing `?netlag=ms` does. The harness
serves room.js with two dev-only changes (the delay is read live so one loaded room can sweep, and a switch for
plain decoding); room.js itself is unchanged. Split is even by pledge (MoE 20+20 and 14+13+13 layers, 27B
32+32 and 22+21+21; the host also holds embed and head). `japan` prompt (172 tokens with the template), exact
(greedy) sampling, new chat before every answer so every answer prefills the whole prompt, 128-token answers,
two answers per cell, mean shown (the two were within 0.4 tok/s except one MoE 2-device spec pair, 34.6 / 30.7).

Decode tok/s, plain / speculative, and time to first token:

| Model | Devices | 0 ms | 5 ms | 20 ms | 50 ms | Time to first token |
|---|---|---|---|---|---|---|
| 35B MoE | 1 | 32.5 / 44.2 | | | | 1.04 s |
| 35B MoE | 2 | 27.9 / 32.7 | 21.1 / 27.9 | 12.9 / 20.6 | 7.2 / 13.5 | 1.10 s at 0 ms, 1.13 s at 50 ms |
| 35B MoE | 3 | 24.8 / 33.3 | 17.7 / 26.9 | 9.9 / 17.5 | 5.2 / 10.8 | 1.14 s at 0 ms, 1.14 s at 50 ms |
| 27B | 1 | 9.2 / 14.6 | | | | 2.57 s |
| 27B | 2 | 8.6 / 11.9 | 7.8 / 10.9 | 6.3 / 9.3 | 4.6 / 7.6 | 2.52 s at 0 ms, 2.53 s at 50 ms |
| 27B | 3 | 8.4 / 11.0 | 7.3 / 10.2 | 5.5 / 8.6 | 3.7 / 6.6 | 2.53 s at 0 ms, 2.55 s at 50 ms |

Latency is one way, per hop. A token goes host → worker(s) → host, so a room of N devices pays N hops per lap.

Draft acceptance: MoE 59 % on one device, 38–44 % on 2, 44–51 % on 3; 27B 47–48 % on one device, 35–38 % on
2 and 3. The room picks the draft depth by lap time (K=3 alone, deeper on a chain), so a chain drafts more and
accepts a smaller share of it; the tokens per lap still go up.

Reading:
- Plain decode is exactly compute plus hops times latency. MoE on 2 devices: 36 ms per token at 0 ms, 139 ms at
  50 ms (+103 = 2 × 50); on 3 devices 40 → 194 ms (+154 = 3 × 50). 27B on 2 devices 116 → 220 ms, on 3 119 →
  270 ms. Nothing else in the pipeline grows with latency.
- Speculative decoding is what keeps a room usable on a real network: at 20 ms it is 1.6x plain for the MoE on
  2 devices and 1.8x on 3; at 50 ms 1.9x and 2.1x. On 0 ms it only buys 1.2–1.4x, because every device here
  shares one GPU and a verify block costs real GPU time.
- The MoE on 2 devices at 5 ms (a good home Wi-Fi) is 21 plain / 28 speculative, and still 13 / 21 at 20 ms
  (same city). The 27B at 20 ms is 6.3 / 9.3, close to what it does alone on this machine plain.
- Time to first token hardly moves with latency (+0.03 s at 50 ms): prefill sends the prompt in 16-token frames
  with several in flight, so the network is paid about once per prompt, not once per frame. It is dominated by
  the prefill compute itself (172 tokens in ~1.0 s on the MoE, ~2.5 s on the 27B).
- Splitting costs a little even at 0 ms (MoE 32.5 → 27.9 → 24.8 plain): per hop the hidden state is read back,
  packed, sent, unpacked and uploaded, about 3.5–4 ms per hop here.

References on the same machine:
- Chrome, one device, `tests/bench/chrome_bench.mjs` (2026-09-26, combined build, 40 tokens, K=3, coding
  prompts): MoE 40–46 plain / 65–84 speculative, 27B 10.1–10.8 / 20–27. The room numbers above are lower on one
  device mainly because the `japan` itinerary drafts worse than the coding prompts (acceptance 47–59 % against
  70–85 %) and because a room answer includes the room's own per-token work (sampling, chat, telemetry).
- llama.cpp CUDA, same GGUFs: MoE 85.5 tok/s decode and 2520 tok/s pp512 (b10840), so about 0.07 s to prefill
  this prompt; 27B 13.8 decode (b10840), 377 tok/s pp86 (749f688), about 0.5 s for this prompt. One device,
  no network.

Caveats: loopback with an added delay is not a real link. No bandwidth limit, no loss, no jitter, and none of
the congestion-window round trips that big frames pay on a fresh link (see "Hidden-state transport" above; with
`stripe4` a single token and a K≤7 verify block fit the first window, so decode should be close, but prefill
frames of 164 KB would pay more on a real link than here). Control messages are not delayed. All devices share
one GPU, so compute on different devices does not overlap the way it would on separate machines, which makes the
0 ms rows pessimistic for a real room and the high-latency rows about right. Delay is applied with setTimeout in
the sending tab, so it is at least the stated value (timer slack of about 1 ms).

Notes from getting it to run: with every device as a tab of one Playwright context (the default
`browser.newContext()`), the context is off-the-record, the Cache API weight store lives in RAM, and loading the
35B MoE grew one browser process past 13 GB until Chromium aborted (SIGTRAP) about 6 GB in. One browser per
device with an on-disk profile fixed it. With the weight store disabled instead, the MoE load stalled at about
460 MB per device in this harness (cause not found). `tests/e2e/room.mjs` uses the one-context setup and would
likely hit the same crash on the MoE (it has no MoE entry in its local-weights map today).

## 2026-09-26: load time for tests and benches (CPU side; GPU not yet measured)

Loader CPU cost, with the GPU upload stubbed (`tests/bench/load_profile.js`): the 27B took 43.8 s (6.5 s reading, 37.2 s converting, 20.1 s of that the Q4_0 repack) and the MoE 46.0 s (8.4 s reading, 37.6 s converting). The repack is now 3x faster (u16 copies). With the new converted-weights cache (`tests/weight_cache.js`), a warm load is 4 to 6 s for the 27B and about 9 s for the MoE. The Chrome bench's static server read ranges at 0.27 GB/s; 8 MB reads bring that to 2 GB/s, and pre-converted tensors (`bench.html?wcache=1`) take the MoE tab load path from 74.7 s to 17 s on the CPU side. `tests/run_q38_once.js` runs the 27B suite over one upload. Details, and the commands still to run for GPU validation, are in [testing-fast.md](testing-fast.md).

## 2026-09-27: kernel pass on decode (27B) and the dense engine (Qwen3 1.7B), branch kopt/combined, GB10

Everything here is exact: the 27B's outputs are bit-identical to kopt/base (`tests/test_q38_bits.js`: same
logits hash 85b12667 and trunk-hidden hash eba0b8d5 over a 16-column prefill, 13 plain tokens and 3 spec
steps; `tests/run.sh q38once` passes; `tests/test_moe.js` 3x MATCH llama.cpp, spec == plain), and the dense
engine's logits are bit-identical to kopt/base at 2,600 tokens of context (`tests/test_dense_exact.js`,
hashes 445aa938 / 6aef0553 on both branches; `tests/run.sh quick` passes). Changes that were not exact are
off by default (the one-kernel dense glue).

What changed (branches, each from kopt/base = feat/engine-opt + opt/load-cache):
- `kopt/dense-attn` dense attention that is not latency-bound: `attn_scores_d` (G heads per thread share
  each K row), `attn_softmax_d` (parallel max and exp, then ONE thread adds the exponentials in position
  order with 8 loads in flight), `attn_out_d` (the per-(head, dim) chains read V / p tiles staged in shared
  memory by the whole workgroup, next tile prefetched into registers). Same operations in the same order
  as attn_scores / attn_softmax (a one-thread-per-head kernel before) / attn_out. Switch: `attnFast`.
- `kopt/dense-fuse` (on dense-attn): residual adds folded into the o / down GEMVs (`_acc`), one rmsnorm
  dispatch for all batch columns, the qk-norm + rope + K/V cache writes as three reference-shaped
  multi-column kernels (`head_norm_dmc`, `rope_dmc`, `kv_store_d`): no cache copies, one compute pass per
  layer. A single fused glue kernel was measured too (`fuseGlue`, off): the same rope expression compiles
  to differently rounded code inside a bigger kernel on NVIDIA Vulkan, so it is not exact.
- `kopt/dense-qkv` (on dense-fuse): q, k, v from one GEMV over the row-concatenated weights (`mergeQKV`).
  Dawn (Chrome) rejects two writable bindings of one buffer in a dispatch where wgpu (Deno) does not, so
  the glue kernels bind the merged buffer once; the Chrome bench now logs uncaptured GPU errors.
- `kopt/encode-ahead` both engines: while the GPU runs token N, the command buffer of position N + 1 is
  recorded (a token's commands depend only on the position and the switches); logits copy in the same
  submit. Switch: `encodeAhead`.
- `kopt/rmsnorm` rmsnorm and the dn_pre q/k L2 norms with 4 / 8 loads in flight, same in-order sums.
- `kopt/head-rows` LM head GEMV with 8 rows per workgroup (rows per workgroup never enter a row's
  arithmetic): on for the dense engine, off for the hybrid (no gain on the 27B head).
- `kopt/wide-loads` opt/wide-loads (16-byte weight loads) re-based and GPU-validated: bit-identical in the
  model (COOPWIDE=4,2 gives the same 27B hashes) but no faster (GPU 90.0 -> 91.0 ms/token), not merged.

### Qwen3 1.7B Q8 (dense engine), before (kopt/base) -> after (kopt/combined)

Chrome (tests/bench/chrome_bench.mjs, `prefill=512,4096`; plain decode = 40 tokens after a short chat
prompt; the dense model has no draft head, so no speculative number):

| | base run 1 | base run 2 | combined run 1 | combined run 2 | |
|---|---|---|---|---|---|
| decode, short chat (two-sum / hash-map) | 52.6 / 52.6 | 51.3 / 51.1 | 61.1 / 63.7 | 58.7 / 61.8 | +17% |
| prefill to 512 tokens | 83.9 | 84.2 | 261.1 | 259.4 | 3.1x |
| prefill to 4096 tokens | 22.2 | 22.3 | 176.3 | 176.6 | 7.9x |
| decode at 512 | 35.4 | 35.1 | 58.4 | 58.6 | +66% |
| decode at 4096 | 10.45 | 10.43 | 32.9 | 32.8 | 3.1x |

Deno (tests/bench_dense.js, FILLS=512,4096): prefill 82.2 / 82.4 -> 219.4 / 221.2 at 512 and 19.2 / 19.3 ->
158.9 / 159.6 at 4096; decode 22.9 / 24.0 -> 36.5 / 36.7 at 512 and 7.49 / 7.42 -> 25.3 / 24.9 at 4096.
Per token at 4k context (tests/prof_dense.js): 219 ms -> 41 ms wall; the old softmax was one thread per
head walking 4k positions three times in global memory.

### Qwen 3.8 27B Q4_0, before (kopt/base) -> after (kopt/combined)

Chrome (`batchcols=16&prefill=512,4096`, 40 tokens, K=3):

| | base run 1 | base run 2 | combined run 1 | combined run 2 | combined run 3 |
|---|---|---|---|---|---|
| plain (two-sum / hash-map) | 10.62 / 10.56 | 10.56 / 10.48 | 11.11 / 11.09 | 11.03 / 11.11 | 11.11 / 11.08 |
| spec K=3 (two-sum / hash-map) | 24.01 / 20.19 | 23.69 / 20.12 | 23.99 / 20.12 | 23.96 / 20.28 | 24.08 / 20.30 |
| prefill to 512 / 4096 | 72.7 / 66.4 | 72.5 / 66.3 | 72.7 / 66.1 | 73.0 / 66.3 | 73.0 / 66.3 |
| decode at 4096 | 9.46 | 9.42 | 9.90 | 9.84 | 9.89 |

Plain decode +5%, speculative and prefill unchanged (the verify pass and prefill use the batched kernels,
which this pass did not touch; prefill is the prefill workflow's). Deno (tests/bench_ctx.js MODEL=27b,
FILLS=512,4096): plain 8.77 / 8.80 -> 9.28 / 9.29 at 512 and 8.59 / 8.60 -> 9.08 / 9.08 at 4096; spec at 512
16.98 / 16.97 -> 16.92 / 16.93; prefill 66.2 / 65.9 -> 66.0 / 66.0. (The spec number at 4096 is not
comparable: bench_ctx builds its prompt from engine/qwen35.js, whose text changed, and acceptance went
17/42 -> 14/51.) tests/prof_ts.js: wall 106.1 -> 101.3 ms/token from encode-ahead (GPU time unchanged),
then GPU 90.9 -> 88.7 ms from rmsnorm (3.18 -> 1.64 ms, 129 dispatches) and dn_pre (2.10 -> 1.35 ms).

Where the 27B token still goes (GPU 88.7 ms): the GEMVs are ~79 ms at 200-215 GB/s (gate/up 469 us x 64,
LM head 5.8 ms at 228 GB/s); exact knobs tried per shape (tests/bench_wide.js: 16-byte loads, 1-16 rows
per workgroup) are all bit-identical and within +-5% of today's kernel, so the remaining gap to llama.cpp
(13.8 tok/s) is load efficiency that an exact kernel cannot reorder its way out of.
