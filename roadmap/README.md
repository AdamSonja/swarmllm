# Roadmap

What we are working on next, grouped by the part of Pooled it touches. Each line is one item with its priority and its GitHub issue; the pinned [Roadmap: Pooled](https://github.com/Nehanth/swarmllm/issues/83) issue has the same list as a checklist. Longer design notes live in the numbered files in this folder; an item links its file when it has one. File names never change, because issues and links point at them.

Priorities:

- **P0**: now. Broken, wrong, or the biggest thing holding rooms back.
- **P1**: next. Planned and ready to build.
- **P2**: later, or needs a design note first.

When an issue is filed, it gets the `area:` label of the section it sits in here.

## Kernels

- **P0** Prefill speed: land the prefill kernels (GEMM tiles, chunked DeltaNet, flash-attention prefill, MoE grouping) with before/after rows for the 27B and the MoE at 1K and 8K prompts. [02](02-prefill-gemm.md) ([#2](https://github.com/Nehanth/swarmllm/issues/2))
- **P0** Coop GEMV with 8 rows per workgroup returns wrong values: fix the kernel or drop ROWS=8 from the autotuner. ([#47](https://github.com/Nehanth/swarmllm/issues/47))
- **P1** Phone WebGPU limits: windowed bindings for stacked MoE experts, a per-device limits probe, and a phone smoke test. ([#65](https://github.com/Nehanth/swarmllm/issues/65))
- **P1** Close the Mac gap: re-measure at HEAD, measure WebGPU bandwidth on the Mac, and ship a browser profiling page. [24](24-close-the-mac-gap.md) ([#24](https://github.com/Nehanth/swarmllm/issues/24))
- **P1** Prefill GEMM for the MoE and the 1.7B: pin GEMM shapes for dim 2048 and the 1.7B so they stop falling back to the batched GEMV. ([#66](https://github.com/Nehanth/swarmllm/issues/66))
- **P2** MoE decode kernels: shared expert as a slot, fold the combine, fewer dispatches per token, then freeze MoE goldens. ([#67](https://github.com/Nehanth/swarmllm/issues/67))

## Inference

- **P0** Split MoE room ends answers mid-call around 200 tokens: make a 2 and 3 device MoE room match solo, starting from the open batched-prefill mismatch. [31](31-moe-split-correctness.md) ([#68](https://github.com/Nehanth/swarmllm/issues/68))
- **P1** Persist room checkpoints to OPFS so a reload resumes with the conversation reused instead of re-prefilled. [30](30-long-context-and-sessions.md) ([#69](https://github.com/Nehanth/swarmllm/issues/69))
- **P1** Tokenizer check against llama.cpp: the pre-tokenizer uses GPT-2's split, not Qwen's, which can change prompt tokens for code and newlines. ([#70](https://github.com/Nehanth/swarmllm/issues/70))
- **P2** Longer context: time 1K, 8K and 32K prompts on real hardware and decide whether int8 KV can be the default. [30](30-long-context-and-sessions.md) ([#71](https://github.com/Nehanth/swarmllm/issues/71))
- **P2** Host-side overhead: GPU top-k and sampling for the batched head, and encode-ahead. [26](26-host-side-overhead.md) ([#26](https://github.com/Nehanth/swarmllm/issues/26))
- **P2** Batched multi-session decode: run chat and Code sessions through one batched pass, bit-exact per session. ([#72](https://github.com/Nehanth/swarmllm/issues/72))
- **P2** Pipelined speculative windows across devices, so a multi-device room keeps more than one lap in flight. [09](09-lap-overlap.md) ([#9](https://github.com/Nehanth/swarmllm/issues/9))

## Harness (Code mode)

- **P1** Prefix-cache reuse across agent turns and stable compaction, so a long Code session never re-prefills more than what is new. [30](30-long-context-and-sessions.md) ([#73](https://github.com/Nehanth/swarmllm/issues/73))
- **P1** Agent robustness: the full tool-call grammar in the sampler, repeated-error detection, fuzzy edit matching, first-error-first feedback. [28](28-code-mode-agent-robustness.md) ([#74](https://github.com/Nehanth/swarmllm/issues/74))
- **P1** Turn on `run_js` in production once the preview origin is live, so the agent can check logic, not just page errors. [29](29-code-mode-preview-origin.md) ([#75](https://github.com/Nehanth/swarmllm/issues/75))
- **P1** Harness correctness fixes from the audit: engine-mode ContextFull, stale repeat-call answers after outside edits, preview chunk size, orphaned session files. ([#76](https://github.com/Nehanth/swarmllm/issues/76))

## Loading

- **P1** Pinned model revisions, weight hash checks, and `navigator.storage.persist()`, so an upstream re-upload can never mix old and new ranges. [19](19-pinned-revisions-and-cache.md) ([#19](https://github.com/Nehanth/swarmllm/issues/19))
- **P1** Cache converted weights per model revision and layer range, so a returning device skips conversion. [19](19-pinned-revisions-and-cache.md) ([#77](https://github.com/Nehanth/swarmllm/issues/77))
- **P1** Downloads: retry with backoff on 429 and 5xx, require 206 range responses, a pinned mirror, and tuned parallel streams. [18](18-download-path.md) ([#18](https://github.com/Nehanth/swarmllm/issues/18))

## Networking

- **P1** TURN relay for strict NATs, with direct or relayed shown per link. [01](01-relay-fallback.md) ([#1](https://github.com/Nehanth/swarmllm/issues/1))
- **P1** Self-hosted signaling, vendored PeerJS, and a status canary. [15](15-self-hosted-signaling-and-status.md) ([#15](https://github.com/Nehanth/swarmllm/issues/15))
- **P1** Cut the per-hop fixed cost and use the fewest-hops chain: GPU-side packing, one readback per hop, chain order by measured round trip. [27](27-placement-and-chain-order.md) ([#27](https://github.com/Nehanth/swarmllm/issues/27))
- **P1** Security sweep: host-authoritative protocol, message validation, CSP and safe headers, an honest SECURITY.md. [17](17-security-sweep.md) ([#17](https://github.com/Nehanth/swarmllm/issues/17))
- **P2** Data channel on lossy Wi-Fi: measure p50 and p90 hop time, then try unordered channels with forward error correction. [25](25-cross-network-quick-wins.md) ([#34](https://github.com/Nehanth/swarmllm/issues/34))
- **P2** Offline rooms: hotspot, QR signaling, and weights shared from a device that already has them. [05](05-offline-rooms.md) ([#5](https://github.com/Nehanth/swarmllm/issues/5))

## Room

- **P0** Escape peer-supplied device names in HTML attributes and cap their length: a crafted name can run script in every member's tab today. [17](17-security-sweep.md) ([#17](https://github.com/Nehanth/swarmllm/issues/17))
- **P1** Automatic re-deal when a device leaves or joins, resuming from the last checkpoint instead of re-prefilling. [03](03-swarm-recovery.md) ([#3](https://github.com/Nehanth/swarmllm/issues/3))
- **P1** Hand the host role to another device when the host leaves, keeping the chat thread. [27](27-placement-and-chain-order.md) ([#58](https://github.com/Nehanth/swarmllm/issues/58))
- **P1** Keep worker tabs alive under memory pressure and in the background, with a clear way back into the room. ([#44](https://github.com/Nehanth/swarmllm/issues/44))
- **P1** Actionable errors and a "copy diagnostic report" button with build, protocol, GPU limits and per-hop stats. [16](16-actionable-errors-and-versioning.md) ([#16](https://github.com/Nehanth/swarmllm/issues/16))
- **P2** Kick a device, and dedupe devices that rejoin, by a stable per-browser id. ([#56](https://github.com/Nehanth/swarmllm/issues/56))
- **P2** Public demo room: ask-only guests, per-guest quotas, a room that stays up. [20](20-public-demo-room.md) ([#20](https://github.com/Nehanth/swarmllm/issues/20))
- **P2** Split room.js into modules (pipeline, generation, request and stream contract) so the room can run headless. [22](22-extract-the-runtime.md) ([#22](https://github.com/Nehanth/swarmllm/issues/22))

## UI

- **P1** Lend this device: wake lock, warnings when backgrounded or on battery saver, what this device holds, and a clear message when the room ends. ([#78](https://github.com/Nehanth/swarmllm/issues/78))
- **P2** Show download speed and time left on the load bar and on each device card. [18](18-download-path.md) ([#57](https://github.com/Nehanth/swarmllm/issues/57))
- **P2** Delete cached weights for one model, with sizes per model. [19](19-pinned-revisions-and-cache.md) ([#59](https://github.com/Nehanth/swarmllm/issues/59))

## Infra

- **P1** Separate preview origin for Code mode on a second domain, so a hung preview cannot freeze the room. [29](29-code-mode-preview-origin.md) ([#79](https://github.com/Nehanth/swarmllm/issues/79))
- **P2** `npx pooled serve`: an OpenAI-compatible local endpoint for a room, and later a native peer for headless GPUs. [04](04-local-endpoint.md), [11](11-native-peer.md) ([#4](https://github.com/Nehanth/swarmllm/issues/4))

## Tests and benchmarks

- **P1** Real-model eval run for Code mode: the 12 eval tasks on the MoE and the 27B, then a two-device room, with success rate, steps and reused tokens. ([#80](https://github.com/Nehanth/swarmllm/issues/80))
- **P1** Real cross-network numbers: per-link latency and loss in the room emulator, then one run across real networks. [25](25-cross-network-quick-wins.md) ([#25](https://github.com/Nehanth/swarmllm/issues/25))
- **P1** GPU tests that print FAIL must exit nonzero, so `tests/run.sh` catches them. ([#81](https://github.com/Nehanth/swarmllm/issues/81))
- **P2** Reproducible benchmarks: a written protocol, a `/bench` page, and a device matrix. [21](21-reproducible-benchmarks.md) ([#21](https://github.com/Nehanth/swarmllm/issues/21))
- **P2** Benchmarks use the room's settings (one shared preset), and the kernel-family profiler is rebuilt from the engine's own pipeline list. ([#82](https://github.com/Nehanth/swarmllm/issues/82))

## Docs

- **P2** Rename leftovers in GitHub issue titles, tracking issues for every item here, and `area:` labels on every issue. [23](23-contributor-on-ramp.md) ([#23](https://github.com/Nehanth/swarmllm/issues/23))

## Done

| # | Item | Status |
|---|---|---|
| 12 | [Stop, fail fast, re-deal: the room survives launch day](12-room-survives-launch-day.md) | landed; automatic re-deal is under Room |
| 13 | [Multi-turn conversation and an honest context limit](13-multi-turn-context.md) | landed; context per model is under Inference |
| 14 | [Pre-flight check, join links, and a model ladder](14-preflight-and-join-links.md) | landed |
| 06 | [More models: Qwen 3.8 27B, Qwen 3.6 35B MoE, Qwen3 1.7B](06-more-models.md) | shipped |
| - | [Rename to Pooled at pooled.run](../docs/rename-pooled.md) | code, docs and domain done; the repo move is left |

Research notes with no scheduled work yet: [07 persistent rooms and stats](07-rooms-and-stats.md), [08 audited compute](08-verification.md), [10 expert-split MoE](10-moe-expert-split.md). Older reasoning is in [docs/archive/master-plan.md](../docs/archive/master-plan.md) and the [roadmap gap review](../docs/archive/roadmap-review.md), which keep the old name.

**Explicitly not on the roadmap** (see [GOVERNANCE.md](../GOVERNANCE.md)): accounts, tokens or ads, rooms open to strangers by default, distributed training, noise or permutation "privacy" features, frontier-scale models that cannot fit a room. Native and headless peers *are* on it; the browser path just never stops being enough on its own.
