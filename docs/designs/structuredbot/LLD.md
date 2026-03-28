# StructuredBot and AsyncStructuredBot — Low-Level Design

**Created**: 2026-03-28

**Last updated**: 2026-03-28

**HLD Link**: [../../high-level-design.md](../../high-level-design.md)

## Requirements (EARS)

- [structuredbot-EARS.md](./structuredbot-EARS.md) — `StructuredBot`, `AsyncStructuredBot`, Pydantic validation loop.

## Overview

`StructuredBot` subclasses `SimpleBot` but does **not** use `SimpleBot.__call__` for its main path. It runs a dedicated loop that asks the model for JSON, validates against a **Pydantic** model, and retries with validation errors in the conversation until success or `num_attempts` is exhausted.

## Classes

| Class | Module | Base |
| ----- | ------ | ---- |
| `StructuredBot` | `llamabot.bot.structuredbot` | `SimpleBot` |
| `AsyncStructuredBot` | `llamabot.bot.structuredbot` | `StructuredBot` |

## Model gate

`StructuredBot.__init__` validates that the LiteLLM model supports `response_format` / `response_schema` (via `get_supported_openai_params` / `supports_response_schema`), except for **`ollama_chat`** in the model name string where the check is skipped.

## Message composition

`compose_structured_first_attempt_messages` uses **only** `[system_prompt] + to_basemessage(user)`—no `memory` retrieval path. `StructuredBot` does not expose `memory` on its constructor; it does not participate in the `SimpleBot` memory/docstore flow for structured calls.

## `__call__` loop

Iterates up to `num_attempts`: `make_response` + `stream_chunks`, `extract_content`, `json.loads`, `pydantic_model.model_validate`. On failure, extends the message list with the last assistant content and a `HumanMessage` describing the validation error (`get_validation_error_message`). On final failure, `allow_failed_validation` may return `model_construct(**last_codeblock)`.

## Async

`AsyncStructuredBot.__call__` runs the synchronous `StructuredBot.__call__` in **`asyncio.to_thread`**. `stream_async` streams one structured attempt without validation on partial chunks (see code).

## Traceability (intent → code)

| EARS ID prefix | Code |
| -------------- | ---- |
| `CORE-STRUCT-*` | `llamabot/bot/structuredbot.py` |

## Related Documents

- [High-Level Design](../../high-level-design.md)
- [SimpleBot LLD](../simplebot/LLD.md) — shared LiteLLM helpers (`make_response`, `stream_chunks`, `completion_kwargs_for_messages`).
- [structuredbot-EARS](./structuredbot-EARS.md)
