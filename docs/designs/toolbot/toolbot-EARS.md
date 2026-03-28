# ToolBot and AsyncToolBot — EARS

**Parent LLD**: [./LLD.md](./LLD.md)

## Tool registration

- [x] **CORE-TOOL-001**: Where `tools` is `None` or empty, `ToolBot.__init__` shall use `DEFAULT_TOOLS` as the full tool list.

- [x] **CORE-TOOL-002**: Where `tools` is non-empty and does not include all `DEFAULT_TOOLS` by function name, the system shall prepend `DEFAULT_TOOLS` before the user tools.

- [x] **CORE-TOOL-003**: The system shall expose LiteLLM `tools` as the list of `json_schema` attributes from the resolved tool functions and shall map `__name__` to the callable in `name_to_tool_map`.

## Message composition

- [x] **CORE-TOOL-010**: `compose_tool_messages` shall resolve callable arguments by invoking them and requiring a `str` result wrapped in `HumanMessage`.

- [x] **CORE-TOOL-011**: When `chat_memory` is set and user messages exist, the message list shall include `chat_memory.retrieve(user_messages[0].content)` after the system prompt.

- [x] **CORE-TOOL-012**: When `execution_history` is provided, the system shall append a `SystemMessage` summarizing up to the last five tool calls before the user messages.

## ToolBot sync

- [x] **CORE-TOOL-020**: `ToolBot.__call__` shall return the list from `extract_tool_calls` after `make_response` and `stream_chunks`, and shall append the first user message and an `AIMessage` whose content is the string representation of that tool-call list to `chat_memory`.

## AsyncToolBot

- [x] **CORE-TOOL-030**: `AsyncToolBot.__call__` shall obtain completions via `make_async_response` and shall assemble the final `ModelResponse` with `async_stream_chunks` before `extract_tool_calls`.

- [x] **CORE-TOOL-031**: `AsyncToolBot.__call__` shall not wrap synchronous `ToolBot.__call__` in `asyncio.to_thread`; it shall use native `acompletion`.

## Verification

| ID | Tests / code |
| --- | --- |
| CORE-TOOL-001–003 | `tests/bot/test_toolbot.py` |
| CORE-TOOL-010–012 | `tests/bot/test_toolbot.py` (`compose_tool_messages`) |
| CORE-TOOL-020 | `tests/bot/test_toolbot.py` |
| CORE-TOOL-030–031 | `tests/bot/test_async_toolbot_call.py`, `llamabot/bot/toolbot.py` (`AsyncToolBot`) |

## Related Documents

- [ToolBot LLD](./LLD.md)
- [SimpleBot LLD](../simplebot/LLD.md)
- [AgentBot reference graph EARS](../agentbot/reference-graph-EARS.md)
- [High-Level Design](../../high-level-design.md)
