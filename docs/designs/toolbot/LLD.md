# ToolBot and AsyncToolBot — Low-Level Design

**Created**: 2026-03-28

**Last updated**: 2026-03-28

**HLD Link**: [../../high-level-design.md](../../high-level-design.md)

## Requirements (EARS)

- [toolbot-EARS.md](./toolbot-EARS.md) — `ToolBot`, `AsyncToolBot`, tool schemas and chat memory.

## Overview

`ToolBot` performs single-turn **tool selection**: the model returns **function calls**, and the bot exposes a **list of `ChatCompletionMessageToolCall`** (not a user-facing assistant string). `AsyncToolBot` uses native **`acompletion`** and **`async_stream_chunks`**—no `asyncio.to_thread` around the sync `ToolBot.__call__`.

## Classes

| Class | Module | Base |
| ----- | ------ | ---- |
| `ToolBot` | `llamabot.bot.toolbot` | `SimpleBot` |
| `AsyncToolBot` | `llamabot.bot.toolbot` | `ToolBot` |

## Tools and memory

- Constructor accepts `tools` and `chat_memory` (`ChatMemory` default). When `tools` is empty or missing `DEFAULT_TOOLS` by name, **`DEFAULT_TOOLS` is prepended** (`llamabot.components.tools`).
- Each tool’s `json_schema` is passed to LiteLLM as `tools`; `name_to_tool_map` maps function names back to callables.

## Message composition

`compose_tool_messages`:

1. Resolves `Callable` arguments by calling them (must return `str`).
2. `system_prompt` + `chat_memory.retrieve(first_user_content)` when `user_messages` exist.
3. Optional `execution_history` (last five calls) as a `SystemMessage` prefix.
4. Appends user messages.

## Call semantics

- **`ToolBot.__call__`**: `make_response` + `stream_chunks`, `extract_tool_calls`, updates `chat_memory` with first user message and `AIMessage(content=str(tool_calls))`.
- **`AsyncToolBot.__call__`**: `make_async_response` + `async_stream_chunks`, then same extraction and memory updates.

## Traceability (intent → code)

| EARS ID prefix | Code |
| -------------- | ---- |
| `CORE-TOOL-*` | `llamabot/bot/toolbot.py` |

## Related Documents

- [High-Level Design](../../high-level-design.md)
- [SimpleBot LLD](../simplebot/LLD.md) — shared LiteLLM helpers.
- [AgentBot LLD](../agentbot/LLD.md) — PocketFlow graph uses `ToolBot` / `AsyncToolBot` inside `DecideNode`.
- [toolbot-EARS](./toolbot-EARS.md)
