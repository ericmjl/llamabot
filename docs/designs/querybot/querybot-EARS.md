# QueryBot and AsyncQueryBot — EARS

**Parent LLD**: [./LLD.md](./LLD.md)

## Construction

- [x] **QRY-RAG-001**: The system shall provide `QueryBot` with a required `docstore` of type `AbstractDocumentStore` and an optional `memory` of the same abstract type.

- [x] **QRY-RAG-002**: After `SimpleBot.__init__`, the system shall set `self.memory` from the `QueryBot` constructor argument (optional second document store), independent of `SimpleBot`’s default `memory` parameter.

## Retrieval and message order

- [x] **QRY-RAG-010**: `compose_rag_messages` shall call `docstore.retrieve(query_content, n_results)` and wrap each returned chunk in `RetrievedMessage`.

- [x] **QRY-RAG-011**: Where `memory` is set, the system shall call `memory.retrieve(query_content, n_results)` and wrap each chunk in `RetrievedMessage` after docstore chunks and before the final user `HumanMessage`.

- [x] **QRY-RAG-012**: The final message in the composed list before `to_basemessage` shall be `HumanMessage` with content equal to the query string.

- [x] **QRY-RAG-013**: The system shall record nested spans `retrieval` and, when applicable, `memory_retrieval`, with result counts on the spans and outer span.

## Completion and persistence

- [x] **QRY-RAG-020**: `QueryBot.__call__` shall pass `processed_messages` from `compose_rag_messages` to `make_response` / `stream_chunks`, and shall pass `messages + [response_message]` to `sqlite_log`.

- [x] **QRY-RAG-021**: Where `memory` is set, `QueryBot.__call__` shall call `memory.append` with the assistant message **content string** after logging.

## AsyncQueryBot

- [x] **QRY-RAG-030**: `AsyncQueryBot.__call__` shall delegate to `QueryBot.__call__` via `asyncio.to_thread` with the same `query` and `n_results`.

- [x] **QRY-RAG-031**: `AsyncQueryBot.stream_async` shall use `compose_rag_messages` then `stream_tokens_for_messages` with `processed_messages`, and the `finalize` callback shall log the turn and append assistant content to `memory` when `memory` is set.

## Verification

| ID | Tests / code |
| --- | --- |
| QRY-RAG-001–002, 010–013, 020–021 | `tests/bot/test_querybot.py` |
| QRY-RAG-030–031 | `tests/test_streaming_async.py` (`AsyncQueryBot.stream_async`); `QRY-RAG-030` implementation in `llamabot/bot/querybot.py` |

## Related Documents

- [QueryBot LLD](./LLD.md)
- [High-Level Design](../../high-level-design.md)
