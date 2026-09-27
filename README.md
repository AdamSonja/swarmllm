<p align="center">
  <a href="https://pooled.run"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="site/logo/wordmark-dark.svg">
    <img src="site/logo/wordmark-light.svg" height="56" alt="pooled">
  </picture></a>
</p>
<p align="center"><b>Run a big open model with your friends, in the browser.</b></p>
<p align="center">
  <a href="https://pooled.run">Site</a> ·
  <a href="https://pooled.run/room">Start a room</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/bench-log.md">Benchmarks</a> ·
  <a href="roadmap/">Roadmap</a> ·
  <a href="SECURITY.md">Threat model</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-2A45E0">
  <img alt="runtime" src="https://img.shields.io/badge/runs%20on-WebGPU%20%2B%20WebRTC-16171c">
</p>

https://github.com/user-attachments/assets/4f349e4b-c699-45da-abe8-e9162689293e

<p align="center"><sub>Demo, recorded September 7, 2026: Qwen 3.8 27B across a MacBook and an iPhone in browser tabs, same Wi‑Fi, 400 tokens at 10.7 tok/s. <a href="https://github.com/Nehanth/swarmllm/releases/download/v0.2.0/swarmllm-demo-2026-09-07.mp4">Download</a>.</sub></p>

Pooled runs one open model across the devices in a room. Your laptop, a friend's desktop and a phone each hold some of the model's layers, and together they run a model none of them could run alone. There is nothing to install and no account, and no server does any of the thinking.

## How it works

Friends open one link. Each device (laptop, desktop, phone) holds some of the model's layers and runs them on its own GPU with our own WGSL kernels. For every token, a small hidden state (10 KB) passes from tab to tab over direct WebRTC connections, and the host turns the result into the next word.

```
host      embed the last token → hidden state
   ↓ 10 KB over WebRTC
laptop    layers 0–21           ─┐
desktop   layers 22–42           ├─ each device runs its layers on its own GPU
phone     layers 43–63          ─┘
   ↓ back to the host
host      final norm → LM head → sample → next token (and draft the ones after it)
```

Models in the room today:

| Model | Size | Notes |
|---|---|---|
| Qwen 3.8 27B | ~15 GB (Q4_0) | hybrid Gated DeltaNet + attention, built-in draft layer for speculative decoding, 16K context (up to 32K) |
| Qwen 3.6 35B MoE | Q4_0, needs ~22 GB across the room | 256 experts, 8 active per token, so it decodes several times faster than the 27B; 32K context (up to 64K) |
| Qwen3 1.7B | ~2 GB (Q8_0) | small and quick, for rooms of phones and light laptops |

Every device downloads only its own layers, from Hugging Face or from another device in the room that already has them, and keeps them cached for next time. If a device leaves, the room says so and deals its layers out again. Speculative decoding, batched prefill and every kernel trick are checked by golden tests, so the speculative stream is the same as plain decoding.

Details: [docs/architecture.md](docs/architecture.md), [docs/kernels.md](docs/kernels.md), [docs/protocol.md](docs/protocol.md), [docs/models.md](docs/models.md).

## Code mode

Switch the room to Code and ask for something, like "build a tetris game". The room's model runs a small agent loop: it writes the files, serves them on a virtual localhost (`:5173`) in a sandboxed preview, reads its own console errors, and fixes them. Everyone in the room sees the files and can run the same preview on their own screen. The files live in the host's browser, or in a folder on disk the host picks, and edits to a real folder wait for the host's approval.

Design: [docs/design/harness-app.md](docs/design/harness-app.md). What the agent can and cannot touch: [SECURITY.md](SECURITY.md#code-mode).

## Run it locally

```bash
git clone https://github.com/Nehanth/swarmllm && cd swarmllm
npx -y serve -l 8080 .        # then open http://localhost:8080/room
```

`serve` reads `serve.json` for the `/room` and `/r/:code` rewrites. Production uses the same two rewrites in `vercel.json`, so keep the two files in step.

## Tests

```bash
npm test                                   # unit tests, no GPU (Deno)
npm run e2e:code                           # Code mode end to end, no GPU: two tabs, a real room, a scripted model
node tests/e2e/preview_browser.mjs         # the Code mode preview sandbox, no GPU
node tests/e2e/room_synth.mjs              # the real engine and room on a tiny synthetic model, no GPU
npm run test:gpu                           # golden tests on Qwen3 0.6B (needs a WebGPU GPU and Deno 2)
npm run test:q38                           # 27B suites, including speculative == plain
```

The no-GPU tests run headless Chromium with WebGPU on SwiftShader. They need `npm install` (Playwright and the `peer` server). GPU tests read model files from `models/`; see [docs/models.md](docs/models.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Architecture

```
engine/            the WebGPU engine (ES modules; engine.js re-exports the public API)
  dense.js         DenseEngine: dense models (Qwen3, SmolLM)
  qwen35.js        Qwen35Engine: hybrid Gated DeltaNet + attention, MoE, batched paths, speculation
  wgsl/            kernels: base, coop (GEMV family), qwen35 (DeltaNet), gemm (prefill), moe
  gguf.js · tokenizer.js · sampling.js · quant.js · autotune.js · selftest.js · safetensors.js
room.js            the room: signaling, links, layer split, downloads, the generation loop
room/              transport and wire, conversation, models, plan, lookup drafts, preflight, QR, room card, Code pane
harness/           Code mode: agent loop, tools, workspace, preview server and sandbox, sessions
p2p.html           the room's markup and styles (served at /room and /r/<code>)
index.html, site/  the landing page
tests/             unit/ (no GPU) · e2e/ (SwiftShader) · GPU golden tests · run.sh
benchmarks/        tok/s harnesses and the kernel-family profiler
docs/              architecture, kernels, protocol, models, bench log, design, research, archive
roadmap/           one file per planned item
```

More in [docs/architecture.md](docs/architecture.md).

## Performance

Qwen 3.8 27B Q4_0, greedy, same output at every row. Full history with commits in [docs/bench-log.md](docs/bench-log.md).

| Device | Decode plain | Decode speculative | Prefill | Native llama.cpp, same GGUF, same machine |
|---|---|---|---|---|
| NVIDIA GB10 (Deno / Vulkan, headless) | 9.0 tok/s | 16.1 tok/s | 44 tok/s | 8.0 decode (tg32), 377 prefill (pp86), CUDA build 749f688 |
| MacBook Pro (Chrome / Metal), solo | 6.7 tok/s | 10.8 tok/s | ~20 tok/s | — |
| MacBook Pro + iPhone, same Wi‑Fi, 62 + 2 layers | — | 7.7 tok/s | 8.5 s for a 169-token prompt | — |
| Cross-internet room (host + peer) | — | 3.5–6 tok/s | — | — |

Qwen 3.6 35B MoE on the GB10 in Chrome: 32–46 tok/s plain and 50–84 tok/s speculative, depending on the prompt and the build (bench log, 2026-09-26). Prefill is the known gap to native, and the work on it is roadmap 02.

## How it compares

Other projects split or share models across machines. The differences are what has to be installed and where the model runs.

| | Model per device | Devices | Install | Network |
|---|---|---|---|---|
| **Pooled** | some of the layers | laptops, desktops and phones, any OS with a WebGPU browser | none, open a URL | same Wi‑Fi or across the internet (WebRTC) |
| exo | some of the layers | machines that run Python and MLX or tinygrad | Python package per node | one network |
| llama.cpp `rpc-server` | some of the layers | machines that run the binary | binary and an open port per node; the docs say not for untrusted networks | LAN in practice |
| Petals | some of the layers | server GPUs in a public swarm | Python client and server | internet, public swarm |
| distributed-llama | some of the layers | Linux boxes and Raspberry Pis | binary per node | LAN |
| WebLLM / MLC, transformers.js | the whole model | one browser tab | none | none needed |
| Ollama, llmman | the whole model | one machine per request | native app | routing between machines, no splitting |

The engine underneath is our own WGSL, not WebLLM, MLC or llama.cpp. The model weights and tokenizer come from Qwen, hosting from Hugging Face, and signaling from PeerJS.

Browsers: Chrome on macOS is the tested host. Safari on an iPhone joins a room and holds a few layers. Safari on a Mac reloads the tab under memory pressure when it holds most of the 27B, so do not host from it. Firefox and Linux Chromium need WebGPU turned on and we have not tested them. Devices without WebGPU join as ask-only guests.

## Privacy

A room is a shared conversation: everyone in it sees the questions and answers, on purpose. No server sees them. The devices running layers work on hidden states, which are *not* private against a determined peer, so run rooms with people you would share a document link with. See [SECURITY.md](SECURITY.md).

## Roadmap

What's next, grouped as Now, Code mode, Speed, Multi-device and Later: [roadmap/README.md](roadmap/README.md). Progress is tracked in the pinned "Roadmap: Pooled" issue.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [GOVERNANCE.md](GOVERNANCE.md). Benchmark reports from hardware we don't have are especially welcome (there's an issue template). Contributors are listed in [AUTHORS](AUTHORS).

## Formerly SwarmLLM

Pooled used to be called SwarmLLM. The name collided with the older [enapt/SwarmLLM](https://github.com/enapt/SwarmLLM), so we renamed (see discussion #53). swarmllm.ai redirects to pooled.run, and old join links keep working. Your first visit to pooled.run downloads model weights again, because browser caches are per site.

## Citation

```bibtex
@software{pooled2026,
  author = {Narendrula, Nehanth},
  title  = {Pooled: run a big open model across your devices, in the browser},
  note   = {Formerly SwarmLLM},
  year   = {2026},
  url    = {https://github.com/Nehanth/swarmllm}
}
```

## Acknowledgements

Model weights and the GGUF format come from the [Qwen](https://huggingface.co/Qwen) team and [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), whose speculative-decoding graph for Qwen 3.5/3.8 was the reference for ours. Prior work that shaped this: [Petals](https://github.com/bigscience-workshop/petals), [exo](https://github.com/exo-explore/exo), [WebLLM](https://github.com/mlc-ai/web-llm), [LlamaWeb](https://arxiv.org/abs/2605.20706), and the Gated DeltaNet and PipeInfer papers.

## License

[MIT](LICENSE).
