# Docs

## Current

- [architecture.md](architecture.md): how a token flows through the engine and the room, and how Code mode fits on top.
- [tech-stack.md](tech-stack.md): what Pooled is built from and why.
- [kernels.md](kernels.md): the WebGPU engine: kernel families and every trick with its measured effect.
- [protocol.md](protocol.md): room lifecycle and compute frames.
- [models.md](models.md): supported models and how to add one.
- [long-context-and-sessions.md](long-context-and-sessions.md): long context, sessions and checkpoints (roadmap 30).
- [bench-log.md](bench-log.md): every performance change with hardware, commit and numbers.
- [agents.md](agents.md): rules for coding agents working in this repo.
- [rename-pooled.md](rename-pooled.md): the plan for the rename from SwarmLLM to Pooled.
- Roadmap: [../roadmap/](../roadmap/README.md).

## Design

- [design/harness-app.md](design/harness-app.md): Code mode, the in-browser coding agent.
- [design/harness-light.md](design/harness-light.md): making Code mode lighter and sturdier, from small-model harnesses.

## Research

Dated notes from September 2026. They record what we knew then and are not kept up to date.

- [research/kernels-2026-09.md](research/kernels-2026-09.md) and [research/kernels-next-2026-09.md](research/kernels-next-2026-09.md): kernel and scheduling techniques ranked against this engine.
- [research/distributed-2026-09.md](research/distributed-2026-09.md): multi-device decode, hop latency, and how the older enapt/SwarmLLM compares.
- [research/moe-2026-09.md](research/moe-2026-09.md) and [research/offload-2026-09.md](research/offload-2026-09.md): mixture of experts and expert offload.
- [research/tabby-2026-09.md](research/tabby-2026-09.md) and [research/tabby-next-2026-09.md](research/tabby-next-2026-09.md): long context, sessions and tool calls ("Tabby" was the working name of that branch).

## Archive

[archive/](archive/README.md): older plans, kernel plans, early research and handoff notes. They keep the old name on purpose.
