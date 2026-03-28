# LlamaBot — High-Level Design

**Created**: 2026-03-28

**Last updated**: 2026-03-28 — Decision 3: sync routing uses `ToolBot.__call__`; async routing uses `AsyncToolBot.__call__` (`acompletion`). See [AgentBot LLD](designs/agentbot/LLD.md).

## Problem Statement

Researchers and developers need a **Pythonic**, **composable** way to call many LLM providers (via LiteLLM), experiment in notebooks, and ship small apps (CLI, FastAPI, HTMX) without ad-hoc glue code for every project.

## Goals

1. **Unified bot API** — Common patterns for completion, tools, RAG, structure, and agents with consistent message and config handling.
2. **Provider breadth** — Support models exposed through LiteLLM with sensible defaults and overrides.
3. **Composable components** — Messages, docstores, tools, memory, and observability as optional building blocks.
4. **Clear reference patterns** — Documented **reference implementations** (including PocketFlow-based agents) that users can copy or replace.

## Non-Goals

- **Being a full application framework** — LlamaBot is a library; apps bring routing, auth, and deployment.
- **Mandating one agent architecture** — Multiple orchestration styles are valid; the library ships **examples** and **reference graphs**, not a single “blessed” agent product.

## Target Users

- **Notebook experimenters** — Quick `SimpleBot` / `ToolBot` / `AgentBot` usage with marimo or Jupyter-style workflows.
- **App builders** — FastAPI + HTMX demos, CLI tools, and bots with streaming and SSE.
- **Maintainers** — Contributors extending bots, components, and docs.

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                      LlamaBot library                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Bots    │  │Components│  │  CLI /   │  │  Recorder  │ │
│  │ (Simple, │  │ messages,  │  │  helpers │  │  spans,    │ │
│  │  Tool,   │  │ docstore,  │  │          │  │  sqlite    │ │
│  │  Query,  │  │ tools,     │  │          │  │            │ │
│  │  Agent,  │  │ memory     │  │          │  │            │ │
│  │  …)      │  │            │  │          │  │            │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘ │
│         │              │                    LiteLLM         │
│         └──────────────┴──────────────────────► API          │
└─────────────────────────────────────────────────────────────┘
```

**Agent / orchestration:** `AgentBot` is a **reference implementation** of one PocketFlow graph (decide → tool → loopback). Other graphs are encouraged; see the AgentBot feature LLD.

## Key Design Decisions

### Decision 1: LiteLLM as the single provider surface

**Choice**: Route completions through LiteLLM.

**Rationale**: One integration point for many providers and options; aligns with LlamaBot’s “all models LiteLLM supports” goal.

### Decision 2: PocketFlow for agent graph orchestration

**Choice**: Use PocketFlow `Flow` / `AsyncFlow` for the default `AgentBot` topology.

**Rationale**: Explicit graph, routing labels, and loopback without a custom scheduler; users can fork or replace `DecideNode` / edges.

### Decision 3: Tool routing via `ToolBot` / `AsyncToolBot` inside `DecideNode`

**Choice**: The sync graph uses `ToolBot.__call__`; the async graph uses `AsyncToolBot.__call__` with `acompletion`, both single-turn tool-calling.

**Rationale**: Reuses the same message and schema plumbing as `ToolBot`; async entrypoint matches other `Async*` bots.

## Risks and Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Confusion between “AgentBot” and “the only agent” | Document AgentBot as **reference graph**; see `designs/agentbot` LLD and EARS. |
| Sync/async drift | Parallel `Async*` bots and async completion paths where supported; document in LLDs. |

## Related Designs

- [AgentBot reference PocketFlow graph (LLD)](designs/agentbot/LLD.md)
- [Unified chat memory (topic note)](design/unified_chat_memory.md)
- [Observability (topic note)](design/observability.md)
- [Log viewer (topic note)](design/log_viewer.md)
- [Agents build own tools (topic note)](design/agents-build-own-tools.md)
