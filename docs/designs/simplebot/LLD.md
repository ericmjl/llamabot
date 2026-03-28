# SimpleBot and AsyncSimpleBot — Low-Level Design

**Created**: 2026-03-28

**Last updated**: 2026-03-28

**HLD Link**: [../../high-level-design.md](../../high-level-design.md)

## Requirements (EARS)

- [simplebot-EARS.md](./simplebot-EARS.md) — `SimpleBot`, `AsyncSimpleBot`, shared completion helpers.

## Overview

`SimpleBot` is the **base completion stack** for most LlamaBot bots: **LiteLLM** (`completion` / `acompletion`) behind the shared message model, optional **recorder spans**, and **SQLite logging** of turns. Module-level helpers (`completion_kwargs_for_messages`, `make_response`, `make_async_response`, `stream_chunks`, `async_stream_chunks`, `stream_tokens_for_messages`, `extract_tool_calls`, `extract_content`) live in `llamabot.bot.simplebot` and are reused by `StructuredBot`, `ToolBot`, `QueryBot`, and others.

## Classes

| Class | Module | Base |
| ----- | ------ | ---- |
| `SimpleBot` | `llamabot.bot.simplebot` | — |
| `AsyncSimpleBot` | `llamabot.bot.simplebot` | `SimpleBot` |

## Shared completion pipeline

LiteLLM calls are built by `completion_kwargs_for_messages`: `model`, `messages` (role/content from `BaseMessage.model_dump`), `temperature`, `stream`, `completion_kwargs`, optional `api_key`, `mock_response`, and—when a **`tools`** attribute exists on the bot—`tools` and `tool_choice`.

- **Sync** responses: `make_response` → `litellm.completion`.
- **Async** responses: `make_async_response` → `litellm.acompletion`.
- **Streaming assembly**: `stream_chunks` (sync) and `async_stream_chunks` (async) consume LiteLLM streams when `stream_target` is not `"none"`; `stream_tokens_for_messages` yields text deltas for async streaming.

`extract_tool_calls` / `extract_content` normalize `ModelResponse` into `AIMessage` fields and support **JSON-in-content** tool calls (e.g. some Ollama-style outputs) when `message.tool_calls` is absent.

## Message composition

`compose_messages_for_human_messages` builds:

1. `SystemMessage` (from `system_prompt`).
2. Optional **memory** messages via `memory.retrieve(...)` when `memory` is set (type `AbstractDocumentStore` in the constructor; used for chat history or RAG-style retrieval depending on implementation).
3. **User** messages after `to_basemessage(...)`.

## Call semantics

- **`SimpleBot.__call__`**: Creates a **Span** (child of current span when present), records metadata, calls `make_response` + `stream_chunks`, builds `AIMessage` with content and tool calls, **sqlite_log**, and appends to `memory` when configured (last processed user message + assistant message).
- **`AsyncSimpleBot.__call__`**: Uses **token streaming** via `stream_tokens_for_messages` and a `finalize` callback to assemble the final `AIMessage`, span fields, logging, and memory. Raises `RuntimeError` if no assistant message was assembled.

## Configuration

- **`stream_target`**: `stdout`, `panel`, `api`, or `none` (invalid values raise).
- **`o1-preview` / `o1-mini`**: System prompt is coerced to `HumanMessage`, `temperature` set to `1.0`, `stream_target` forced to `none`.
- **`json_mode`**: When `True`, `completion_kwargs_for_messages` requires `pydantic_model` on the bot (used by `StructuredBot`).

## Traceability (intent → code)

| EARS ID prefix | Code |
| -------------- | ---- |
| `CORE-SIMPLE-*` | `llamabot/bot/simplebot.py` |

## Related Documents

- [High-Level Design](../../high-level-design.md)
- [StructuredBot LLD](../structuredbot/LLD.md) — Pydantic validation loop; subclasses `SimpleBot`.
- [ToolBot LLD](../toolbot/LLD.md) — tool selection; subclasses `SimpleBot`.
- [QueryBot LLD](../querybot/LLD.md) — RAG path (`compose_rag_messages`); does not use `SimpleBot.__call__` message composition.
- [simplebot-EARS](./simplebot-EARS.md)
