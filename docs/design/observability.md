# Observability in LlamaBot

This document describes the observability features built into LlamaBot's various bot implementations. These features help track bot behavior, performance, and usage patterns.

## Span-Based Observability

All bots now automatically create **spans** for observability. Spans provide structured tracing of bot operations with timing, attributes, and hierarchical relationships. Spans are **always enabled by default** - no need to call `enable_span_recording()`.

### Span Attributes

Each bot call creates a span with relevant attributes:

- **SimpleBot**: `query`, `model`, `temperature`, `input_message_count`, `input_user_messages`, `input_assistant_messages`, `tool_calls`, `tool_calls_count`

- **QueryBot**: `query`, `n_results`, `model`, `docstore_results`, `memory_results`

- **StructuredBot**: `query`, `model`, `pydantic_model`, `num_attempts`, `validation_attempts`, `validation_success`, `validation_time`, `schema_fields`, `schema_nested_models`

- **AgentBot**: `query`, `max_iterations`, `result`, `iterations`

### Nested Spans

Bots create nested spans for detailed observability:

- `simplebot_call` / `querybot_call` / `structuredbot_call` / `agentbot_call` (root)
  - `llm_request` (LLM API call)
  - `llm_response` (response processing)
  - `retrieval` (for QueryBot - document retrieval)
  - `memory_retrieval` (for QueryBot - memory retrieval)

### Accessing Spans

View spans in several ways:

```python
from llamabot import get_spans

# Get all spans
all_spans = get_spans()

# Get spans for a specific operation
spans = get_spans(operation_name="simplebot_call")

# Display bot's spans (in marimo notebooks)
bot = SimpleBot(...)
bot("Hello")  # Make a call
bot  # Display spans automatically via _repr_html_()
```

## Legacy: run_meta (Deprecated)

**Note**: The `run_meta` dictionary is deprecated for bots. Spans now provide all the same information and more. `run_meta` is only used internally when span recording is disabled, which no longer happens for bots since spans are always enabled.

## Best Practices

1. **Use Span Visualization**: Display bot objects in marimo notebooks to automatically see span hierarchies and timing
2. **Query by Operation**: Use `get_spans(operation_name="...")` to filter spans and see only relevant operations
3. **Monitor Validation**: For `StructuredBot`, check `validation_attempts` and `validation_success` attributes in spans
4. **Track Retrieval Performance**: For `QueryBot`, check `docstore_results` and `memory_results` attributes
5. **Analyze Tool Usage**: Check `tool_calls` and `tool_calls_count` attributes to understand tool execution patterns
6. **View Hierarchies**: Spans automatically show parent-child relationships, making it easy to understand execution flow

## Logging

All bot interactions are automatically logged using `sqlite_log`, which stores:

- Complete message history
- Tool calls and their results
- Timing information
- Bot configuration

This logging can be used for:

- Debugging issues
- Analyzing bot performance
- Training data collection
- Usage pattern analysis
