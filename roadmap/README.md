# Roadmap

One file per item. Each has a status, the reason it matters, a design sketch, and what "done" means, so anyone can pick one up. Order within a group is priority order. Item numbers and file names never change, because issues and links point at them.

Progress is tracked in the pinned "Roadmap: Pooled" issue. Older reasoning is in [docs/archive/master-plan.md](../docs/archive/master-plan.md) and the [roadmap gap review](../docs/archive/roadmap-review.md), which keep the old name.

## Now (by Oct 11)

The rename and the things a public launch cannot do without.

| # | Item | Status |
|---|---|---|
| — | [Rename to Pooled and move to pooled.run](../docs/rename-pooled.md) | in progress |
| 17 | [Security sweep before launch: host-authoritative protocol, honest SECURITY.md, hardened page](17-security-sweep.md) | planned · [#17](https://github.com/Nehanth/swarmllm/issues/17) |
| 16 | [Fail loudly and legibly: actionable errors, a diagnostic report, build and protocol version](16-actionable-errors-and-versioning.md) | planned (protocol version landed) · [#16](https://github.com/Nehanth/swarmllm/issues/16) |
| 15 | [Self-hosted signaling, vendored PeerJS, and a status canary](15-self-hosted-signaling-and-status.md) | planned · [#15](https://github.com/Nehanth/swarmllm/issues/15) |
| 01 | [Relay fallback for strict NATs](01-relay-fallback.md) | planned · [#1](https://github.com/Nehanth/swarmllm/issues/1) |
| 19 | [Pinned model revisions, weight integrity, and cache management](19-pinned-revisions-and-cache.md) | planned (with #59) · [#19](https://github.com/Nehanth/swarmllm/issues/19) |
| 23 | [Contributor on-ramp: labels, seeded issues, no-GPU track, hardware-verifier role](23-contributor-on-ramp.md) | in part · [#23](https://github.com/Nehanth/swarmllm/issues/23) |

## Code mode

The in-browser coding agent on the room's model.

| # | Item | Status |
|---|---|---|
| 28 | [Code mode: agent robustness](28-code-mode-agent-robustness.md) | in progress |
| 29 | [Code mode: preview origin on a second domain](29-code-mode-preview-origin.md) | planned |
| 30 | [Long context and sessions](30-long-context-and-sessions.md) | in part |

## Speed

| # | Item | Status |
|---|---|---|
| 02 | [Prefill GEMM](02-prefill-gemm.md) | Q4 and Q8 landed · prefill/* work open · [#2](https://github.com/Nehanth/swarmllm/issues/2) |
| 26 | [Host-side overhead: encode-ahead, one-submit speculation, GPU sampling, fused glue](26-host-side-overhead.md) | in part (GPU sampling open) · [#26](https://github.com/Nehanth/swarmllm/issues/26) |
| 24 | [Close the Mac gap](24-close-the-mac-gap.md) | planned · [#24](https://github.com/Nehanth/swarmllm/issues/24) |
| 25 | [Cross-network quick wins: draft cache, frame chunking, prefill acks, per-hop telemetry](25-cross-network-quick-wins.md) | landed · real cross-network numbers open · [#25](https://github.com/Nehanth/swarmllm/issues/25) |
| 09 | [Overlapping laps across the network (PipeInfer-style)](09-lap-overlap.md) | research · [#9](https://github.com/Nehanth/swarmllm/issues/9) |

## Multi-device

Rooms that keep going when devices come and go, and bigger models split well.

| # | Item | Status |
|---|---|---|
| 03 | [Spare layer copies and room recovery (takes automatic re-deal from 12)](03-swarm-recovery.md) | planned · [#3](https://github.com/Nehanth/swarmllm/issues/3) |
| 27 | [Placement, chain order and host election (takes #35; see #58)](27-placement-and-chain-order.md) | speed-aware placement landed · chain order open · [#27](https://github.com/Nehanth/swarmllm/issues/27) |
| 31 | [35B MoE split across devices: check correctness](31-moe-split-correctness.md) | research (placeholder) |
| 10 | [Expert-split mixture-of-experts across a room](10-moe-expert-split.md) | research · [#10](https://github.com/Nehanth/swarmllm/issues/10) |
| 18 | [Download path: Hugging Face backoff and mirrors, parallel range streams, phone cache](18-download-path.md) | in part (with #57) · [#18](https://github.com/Nehanth/swarmllm/issues/18) |
| 05 | [Offline rooms: hotspot + QR signaling + peer weight sharing](05-offline-rooms.md) | planned · [#5](https://github.com/Nehanth/swarmllm/issues/5) |

## Later

| # | Item | Status |
|---|---|---|
| 04 | [`npx pooled serve`: an OpenAI-compatible local endpoint](04-local-endpoint.md) | planned · [#4](https://github.com/Nehanth/swarmllm/issues/4) |
| 11 | [Native peer for headless GPUs](11-native-peer.md) | planned · [#11](https://github.com/Nehanth/swarmllm/issues/11) |
| 07 | [Persistent rooms and contribution stats](07-rooms-and-stats.md) | planned · [#7](https://github.com/Nehanth/swarmllm/issues/7) |
| 08 | [Randomly audited compute](08-verification.md) | research · [#8](https://github.com/Nehanth/swarmllm/issues/8) |
| 20 | [Public demo room: ask-only guests, question queue, quotas](20-public-demo-room.md) | planned (queue landed) · [#20](https://github.com/Nehanth/swarmllm/issues/20) |
| 21 | [Reproducible benchmarks and the paper (MLSys, Oct 30)](21-reproducible-benchmarks.md) | planned · [#21](https://github.com/Nehanth/swarmllm/issues/21) |
| 22 | [Extract the runtime: `engine/generate.js`, `room/pipeline.js`, request/stream contract, headless driver](22-extract-the-runtime.md) | planned · [#22](https://github.com/Nehanth/swarmllm/issues/22) |

## Done

| # | Item | Status |
|---|---|---|
| 12 | [Stop, fail fast, re-deal: the room survives launch day](12-room-survives-launch-day.md) | landed · automatic re-deal moved to 03 · [#12](https://github.com/Nehanth/swarmllm/issues/12) |
| 13 | [Multi-turn conversation and an honest context limit](13-multi-turn-context.md) | landed · context per model moved to 30 · [#13](https://github.com/Nehanth/swarmllm/issues/13) |
| 14 | [Pre-flight check, join links, and a model ladder](14-preflight-and-join-links.md) | landed · measured budget moved to #44 · [#14](https://github.com/Nehanth/swarmllm/issues/14) |
| 06 | [More models (shipped: Qwen 3.8 27B, Qwen 3.6 35B MoE; Qwen3.5-122B MoE in the engine)](06-more-models.md) | shipped · [#6](https://github.com/Nehanth/swarmllm/issues/6) |

**Explicitly not on the roadmap** (see [GOVERNANCE.md](../GOVERNANCE.md)): accounts, tokens or ads, rooms open to strangers by default, distributed training, noise/permutation "privacy" features, frontier-scale models that cannot fit a room. Native and headless peers *are* on it (11); the browser path just never stops being enough on its own.

## Status meanings

- **research**: needs a design note before code; open an issue with the *Research / design proposal* template.
- **planned**: design agreed in the item file; ready to build.
- **in progress** / **in part**: someone is on it, or part of it has landed (the item file says which part).
- **landed** / **shipped**: done; what is left moved to the item or issue named in the status.
