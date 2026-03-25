# Async streaming (`stream_async`)

LlamaBot separates **synchronous** completion via [`SimpleBot.__call__`](bots/simplebot.md)
(and `stream_target`) from **asynchronous** APIs on parallel types (`AsyncSimpleBot` and other `Async*`
classes from `llamabot`) that provide `stream_async` and `await` for FastAPI, SSE, and notebooks
that use `async`/`await`.

**Example (Marimo):** [AsyncSimpleBot walkthrough](../examples/async_simplebot.py) (`uvx marimo run --sandbox docs/examples/async_simplebot.py`). Example notebooks install **`llamabot` from this repository** using PEP 723 `[tool.uv.sources]` (path relative to each `.py` file), not from PyPI.

## Canonical contract

Implementations expose:

```python
async def stream_async(...) -> AsyncGenerator[str, None]: ...
```

- **Return type**: An async iterator of **assistant text deltas** (strings). Empty strings are not yielded.
- **Independence from `stream_target`**: Async streaming uses `litellm.acompletion` with `stream=True` when the model supports token streaming (see `model_supports_token_streaming` in `llamabot.bot.simplebot`). Setting `stream_target="none"` only affects synchronous `__call__` on sync bots (and printing in `AsyncSimpleBot.__call__`); it does not disable `stream_async`.
- **Models without token streaming**: For `o1-preview` and `o1-mini`, the provider response is non-streamed; `stream_async` yields a **single** string (full content) to keep the same iterator contract.
- **Side effects**: When streaming finishes, bots run the same logging and memory updates as the synchronous path (where applicable), using the assembled response from `stream_chunk_builder`.

## Shared helpers

Module-level helpers in `llamabot.bot.simplebot`:

| Symbol | Role |
|--------|------|
| `completion_kwargs_for_messages` | Shared kwargs for `completion` / `acompletion` |
| `make_async_response` | `await acompletion(...)` |
| `stream_tokens_for_messages` | Async iterator over text deltas + optional finalize callback |
| `model_supports_token_streaming` | Whether to use `stream=True` for async streaming |

## Bot-specific entry points

Sync bots (`SimpleBot`, `QueryBot`, …) expose blocking `__call__` only. Use the matching **async** class for `stream_async` (and `await __call__` where provided):

| Async class | `stream_async` signature | Notes |
|-------------|---------------------------|--------|
| `AsyncSimpleBot` | `stream_async(*human_messages)` | Same constructor and message shape as `SimpleBot`; `await __call__(...)` returns `AIMessage` |
| `AsyncQueryBot` | `stream_async(query, n_results=20)` | Same as `QueryBot` RAG flow |
| `AsyncToolBot` | `stream_async(*messages, execution_history=None)` | Same as `ToolBot.__call__` |
| `AsyncStructuredBot` | `stream_async(*user_messages)` | Streams **one** structured attempt; no validation on partial chunks |

Message lists are built from shared `compose_*` helpers on the sync base classes (for example `SimpleBot.compose_messages_for_human_messages`, `QueryBot.compose_rag_messages`).

## SSE mapping

`sse_stream` in `llamabot.sse` wraps `stream_async` and maps chunks to SSE events:

| Event type | When |
|------------|------|
| `message` (default) | Each text delta |
| `done` | After the stream completes |
| `error` | On exception (stringified message in `data`) |

See [SSE reference](sse.md) for FastAPI usage.

## Migration note

Using `stream_target="api"` with `SimpleBot.__call__` returns a **sync** generator. For async servers and SSE, use
`AsyncSimpleBot` (or another `Async*` bot) with `stream_async` or `sse_stream`.
