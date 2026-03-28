# SimpleBot and AsyncSimpleBot — EARS

**Parent LLD**: [./LLD.md](./LLD.md)

## Construction and streaming

- [x] **CORE-SIMPLE-001**: The system shall provide a class `SimpleBot` whose constructor accepts `stream_target` only from the set `stdout`, `panel`, `api`, `none`; any other value shall raise `ValueError`.

- [x] **CORE-SIMPLE-002**: Where `model_name` is `o1-preview` or `o1-mini`, the system shall set the system prompt to a `HumanMessage` with the same content, set `temperature` to `1.0`, and set `stream_target` to `none`.

- [x] **CORE-SIMPLE-003**: The system shall build LiteLLM chat messages via `completion_kwargs_for_messages` using only `role` and `content` fields from each `BaseMessage`.

- [x] **CORE-SIMPLE-004**: Where `json_mode` is true and the bot has no `pydantic_model` attribute, or `pydantic_model` is not a subclass of `pydantic.BaseModel`, `completion_kwargs_for_messages` shall raise `ValueError`.

## Message composition and memory

- [x] **CORE-SIMPLE-010**: When `memory` is set (document store), `compose_messages_for_human_messages` shall prepend the system prompt plus `memory.retrieve(...)` before the user messages converted via `to_basemessage`.

- [x] **CORE-SIMPLE-011**: `SimpleBot.__call__` shall append the last processed user message and the assistant `AIMessage` to `memory` when `memory` is configured.

## Completion and logging

- [x] **CORE-SIMPLE-020**: `SimpleBot.__call__` shall record a span (child of the current span when one exists), call `make_response` / `stream_chunks`, construct an `AIMessage` with `extract_content` and `extract_tool_calls`, invoke `sqlite_log` for the full turn, and return that `AIMessage`.

- [x] **CORE-SIMPLE-021**: `extract_tool_calls` shall return structured `tool_calls` when present; when absent and `message.content` is JSON describing tool calls (including single-object or list forms), the system shall synthesize `ChatCompletionMessageToolCall` objects.

## AsyncSimpleBot

- [x] **CORE-SIMPLE-030**: `AsyncSimpleBot.__call__` shall stream via `stream_tokens_for_messages` and shall raise `RuntimeError` if no assistant message is produced after streaming completes.

## Verification

| ID | Tests / code |
| --- | --- |
| CORE-SIMPLE-001–002 | `tests/bot/test_simplebot.py` (`test_simple_bot_init_invalid_stream_target`, o1 branches) |
| CORE-SIMPLE-003–004 | `tests/bot/test_simplebot.py` (`completion_kwargs_for_messages` / JSON mode) |
| CORE-SIMPLE-010–011 | `tests/bot/test_simplebot.py` (memory + `__call__` mocks) |
| CORE-SIMPLE-020–021 | `tests/bot/test_simplebot.py` (`extract_tool_calls`, `extract_content`, `__call__`) |
| CORE-SIMPLE-030 | `llamabot/bot/simplebot.py` (`AsyncSimpleBot.__call__`); streaming coverage in `tests/test_streaming_async.py` (`stream_async`) |

## Related Documents

- [SimpleBot LLD](./LLD.md)
- [High-Level Design](../../high-level-design.md)
