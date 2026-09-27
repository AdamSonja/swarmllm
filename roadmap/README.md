# Roadmap

What we are working on next, grouped by the part of Pooled it touches. Each line is one item with its priority and its GitHub issue (`(#issue)` is a placeholder until the issue is filed). Longer design notes live in the numbered files in this folder; an item links its file when it has one. File names never change, because issues and links point at them.

Priorities:

- **P0**: now. Broken, wrong, or the biggest thing holding rooms back.
- **P1**: next. Planned and ready to build.
- **P2**: later, or needs a design note first.

When an issue is filed, it gets the `area:` label of the section it sits in here.

## Kernels

- **P0** Prefill speed: land the prefill kernels (GEMM tiles, chunked DeltaNet, flash-attention prefill, MoE grouping) with before/after rows for the 27B and the MoE at 1K and 8K prompts. [02](02-prefill-gemm.md) (#issue)
- **P0** Coop GEMV with 8 rows per workgroup returns wrong values: fix the kernel or drop ROWS=8 from the autotuner. (#issue)
- **P1** Phone WebGPU limits: windowed bindings for stacked MoE experts, a per-device limits probe, and a phone smoke test. (#issue)
- **P1** Close the Mac gap: re-measure at HEAD, measure WebGPU bandwidth on the Mac, and ship a browser profiling page. [24](24-close-the-mac-gap.md) (#issue)
- **P1** Prefill GEMM for the MoE and the 1.7B: pin GEMM shapes for dim 2048 and the 1.7B so they stop falling back to the batched GEMV. (#issue)
- **P2** MoE decode kernels: shared expert as a slot, fold the combine, fewer dispatches per token, then freeze MoE goldens. (#issue)

## Inference

- **P0** Split MoE room ends answers mid-call around 200 tokens: make a 2 and 3 device MoE room match solo, starting from the open batched-prefill mismatch. [31](31-moe-split-correctness.md) (#issue)
- **P1** Persist room checkpoints to OPFS so a reload resumes with the conversation reused instead of re-prefilled. [30](30-long-context-and-sessions.md) (#issue)
- **P1** Tokenizer check against llama.cpp: the pre-tokenizer uses GPT-2's split, not Qwen's, which can change prompt tokens for code and newlines. (#issue)
- **P2** Longer context: time 1K, 8K and 32K prompts on real hardware and decide whether int8 KV can be the default. [30](30-long-context-and-sessions.md) (#issue)
- **P2** Host-side overhead: GPU top-k and sampling for the batched head, and encode-ahead. [26](26-host-side-overhead.md) (#issue)
- **P2** Batched multi-session decode: run chat and Code sessions through one batched pass, bit-exact per session. (#issue)
- **P2** Pipelined speculative windows across devices, so a multi-device room keeps more than one lap in flight. [09](09-lap-overlap.md) (#issue)

## Harness (Code mode)

- **P1** Prefix-cache reuse across agent turns and stable compaction, so a long Code session never re-prefills more than what is new. [30](30-long-context-and-sessions.md) (#issue)
- **P1** Agent robustness: the full tool-call grammar in the sampler, repeated-error detection, fuzzy edit matching, first-error-first feedback. [28](28-code-mode-agent-robustness.md) (#issue)
- **P1** Turn on `run_js` in production once the preview origin is live, so the agent can check logic, not just page errors. [29](29-code-mode-preview-origin.md) (#issue)
- **P1** Harness correctness fixes from the audit: engine-mode ContextFull, stale repeat-call answers after outside edits, preview chunk size, orphaned session files. (#issue)

## Loading

- **P1** Pinned model revisions, weight hash checks, and `navigator.storage.persist()`, so an upstream re-upload can never mix old and new ranges. [19](19-pinned-revisions-and-cache.md) (#issue)
- **P1** Cache converted weights per model revision and layer range, so a returning device skips conversion. [19](19-pinned-revisions-and-cache.md) (#issue)
- **P1** Downloads: retry with backoff on 429 and 5xx, require 206 range responses, a pinned mirror, and tuned parallel streams. [18](18-download-path.md) (#issue)

## Networking

- **P1** TURN relay for strict NATs, with direct or relayed shown per link. [01](01-relay-fallback.md) (#issue)
- **P1** Self-hosted signaling, vendored PeerJS, and a status canary. [15](15-self-hosted-signaling-and-status.md) (#issue)
- **P1** Cut the per-hop fixed cost and use the fewest-hops chain: GPU-side packing, one readback per hop, chain order by measured round trip. [27](27-placement-and-chain-order.md) (#issue)
- **P1** Security sweep: host-authoritative protocol, message validation, CSP and safe headers, an honest SECURITY.md. [17](17-security-sweep.md) (#issue)
- **P2** Data channel on lossy Wi-Fi: measure p50 and p90 hop time, then try unordered channels with forward error correction. [25](25-cross-network-quick-wins.md) (#issue)
- **P2** Offline rooms: hotspot, QR signaling, and weights shared from a device that already has them. [05](05-offline-rooms.md) (#issue)

## Room

- **P0** Escape peer-supplied device names in HTML attributes and cap their length: a crafted name can run script in every member's tab today. [17](17-security-sweep.md) (#issue)
- **P1** Automatic re-deal when a device leaves or joins, resuming from the last checkpoint instead of re-prefilling. [03](03-swarm-recovery.md) (#issue)
- **P1** Hand the host role to another device when the host leaves, keeping the chat thread. [27](27-placement-and-chain-order.md) (#issue)
- **P1** Keep worker tabs alive under memory pressure and in the background, with a clear way back into the room. (#issue)
- **P1** Actionable errors and a "copy diagnostic report" button with build, protocol, GPU limits and per-hop stats. [16](16-actionable-errors-and-versioning.md) (#issue)
- **P2** Kick a device, and dedupe devices that rejoin, by a stable per-browser id. (#issue)
- **P2** Public demo room: ask-only guests, per-guest quotas, a room that stays up. [20](20-public-demo-room.md) (#issue)
- **P2** Split room.js into modules (pipeline, generation, request and stream contract) so the room can run headless. [22](22-extract-the-runtime.md) (#issue)

## UI

- **P1** Lend this device: wake lock, warnings when backgrounded or on battery saver, what this device holds, and a clear message when the room ends. (#issue)
- **P2** Show download speed and time left on the load bar and on each device card. [18](18-download-path.md) (#issue)
- **P2** Delete cached weights for one model, with sizes per model. [19](19-pinned-revisions-and-cache.md) (#issue)

## Infra

- **P1** Separate preview origin for Code mode on a second domain, so a hung preview cannot freeze the room. [29](29-code-mode-preview-origin.md) (#issue)
- **P2** `npx pooled serve`: an OpenAI-compatible local endpoint for a room, and later a native peer for headless GPUs. [04](04-local-endpoint.md), [11](11-native-peer.md) (#issue)

## Tests and benchmarks

- **P1** Real-model eval run for Code mode: the 12 eval tasks on the MoE and the 27B, then a two-device room, with success rate, steps and reused tokens. (#issue)
- **P1** Real cross-network numbers: per-link latency and loss in the room emulator, then one run across real networks. [25](25-cross-network-quick-wins.md) (#issue)
- **P1** GPU tests that print FAIL must exit nonzero, so `tests/run.sh` catches them. (#issue)
- **P2** Reproducible benchmarks: a written protocol, a `/bench` page, and a device matrix. [21](21-reproducible-benchmarks.md) (#issue)
- **P2** Benchmarks use the room's settings (one shared preset), and the kernel-family profiler is rebuilt from the engine's own pipeline list. (#issue)

## Docs

- **P2** Rename leftovers in GitHub issue titles, tracking issues for every item here, and `area:` labels on every issue. [23](23-contributor-on-ramp.md) (#issue)

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
