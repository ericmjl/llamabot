# QueryBot and AsyncQueryBot — Low-Level Design

**Created**: 2026-03-28

**Last updated**: 2026-03-28

**HLD Link**: [../../high-level-design.md](../../high-level-design.md)

## Requirements (EARS)

- [querybot-EARS.md](./querybot-EARS.md) — RAG composition, retrieval spans, logging, async behavior.

## Overview

`QueryBot` answers a user query using **retrieval-augmented generation**: it pulls chunks from a required **`AbstractDocumentStore`** (`docstore`), optionally pulls additional chunks from a second store passed as **`memory`**, then completes with LiteLLM via the same **`completion_kwargs_for_messages`** path as `SimpleBot`. It **does not** use `SimpleBot.compose_messages_for_human_messages` or the default “single user turn + optional memory retrieve” flow from `SimpleBot.__call__`; it replaces the call path with **`compose_rag_messages`** and **`QueryBot.__call__`**.

## Inheritance and modules

| Class | Module | Base |
| ----- | ------ | ---- |
| `QueryBot` | `llamabot.bot.querybot` | `SimpleBot` |
| `AsyncQueryBot` | `llamabot.bot.querybot` | `QueryBot` |

## Data and dependencies

| Field | Type | Role |
| ----- | ---- | ---- |
| `docstore` | `AbstractDocumentStore` | Primary retrieval corpus; **required**. |
| `memory` | `Optional[AbstractDocumentStore]` | Optional second retrieval source; also receives **`append`** of assistant text after each successful call when set. |

`SimpleBot.__init__` is invoked **without** wiring `memory` from the `QueryBot` constructor; `QueryBot` assigns `self.memory` after `super().__init__`, reusing the attribute name for the optional second store.

## RAG message composition

`compose_rag_messages(query, n_results, outer_span)`:

1. Resolves `query_content` from `str`, `HumanMessage`, or `BaseMessage`.
2. Starts `messages` with `system_prompt`.
3. Under a nested span **`retrieval`**: `docstore.retrieve(query_content, n_results)`; each chunk becomes `RetrievedMessage(content=chunk)`; span records document counts.
4. If `memory` is set: under span **`memory_retrieval`**, `memory.retrieve(query_content, n_results)` with the same pattern; span records counts.
5. Appends **`HumanMessage(content=query_content)`**.
6. Sets outer span `query` / `temperature` metadata.
7. Returns `(messages, processed_messages)` where **`processed_messages = to_basemessage(messages)`** for the LiteLLM call, and **`messages`** is the list used for **`sqlite_log`** (see below).

## Completion and logging

- **`QueryBot.__call__`**: Outer **Span** (same pattern as other bots), `compose_rag_messages`, then **`make_response(self, processed_messages, stream)`** and **`stream_chunks`**, then **`extract_tool_calls`** / **`extract_content`** → **`AIMessage`**. **`sqlite_log(self, messages + [response_message])`** uses the **pre-`to_basemessage`** `messages` list (system + retrieved + human). If `memory` is set, **`memory.append(response_message.content)`** (assistant string only).

- **`AsyncQueryBot.__call__`**: Runs **`super().__call__`** in **`asyncio.to_thread`** with the same `query` and `n_results`.

- **`AsyncQueryBot.stream_async`**: Same spans and `compose_rag_messages`, then **`stream_tokens_for_messages(self, processed_messages, finalize=...)`** with a `finalize` that logs and appends to `memory` like sync.

## Traceability (intent → code)

| EARS ID | Code |
| ------- | ---- |
| `QRY-RAG-*` | `llamabot/bot/querybot.py` |

## Related Documents

- [High-Level Design](../../high-level-design.md)
- [SimpleBot LLD](../simplebot/LLD.md) — shared LiteLLM helpers (`make_response`, `stream_chunks`, `stream_tokens_for_messages`, `completion_kwargs_for_messages`).
- [QueryBot EARS](./querybot-EARS.md)
