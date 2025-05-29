# Observability in LlamaBot

This document describes the observability features built into LlamaBot's various bot implementations. These features help track bot behavior, performance, and usage patterns.

## Base Metrics (SimpleBot)

All bots inherit from `SimpleBot`, which provides basic observability through the `run_meta` dictionary. This dictionary is initialized at the start of each bot call and contains:

```python
{
    "start_time": datetime.now(),
    "message_counts": {
        "user": 0,
        "assistant": 0,
        "tool": 0
    },
    "tool_usage": {}
}
```

The base metrics track:
- Start time of the bot call
- Message counts by type (user, assistant, tool)
- Basic tool usage statistics

## Specialized Bot Metrics

### AgentBot

The `AgentBot` extends the base metrics with additional tracking for autonomous planning and tool execution:

```python
{
    "max_iterations": max_iterations,
    "current_iteration": 0,
    "tool_usage": {
        "tool_name": {
            "calls": 0,
            "success": 0,
            "failures": 0,
            "total_duration": 0.0
        }
    },
    "planning_metrics": {
        "plan_generated": False,
        "plan_revisions": 0,
        "plan_time": 0.0
    }
}
```

### QueryBot

The `QueryBot` adds metrics specific to document retrieval and query processing:

```python
{
    "query": query_text,
    "n_results": n_results,
    "retrieval_metrics": {
        "docstore_retrieval_time": 0,
        "memory_retrieval_time": 0,
        "docstore_results": 0,
        "memory_results": 0
    }
}
```

### StructuredBot

The `StructuredBot` includes metrics for schema validation and response processing:

```python
{
    "validation_attempts": 0,
    "validation_success": False,
    "schema_complexity": {
        "fields": 0,
        "nested_models": 0
    },
    "validation_time": 0.0
}
```

## Accessing Metrics

All metrics are stored in the `run_meta` dictionary of each bot instance. You can access them after a bot call:

```python
bot = AgentBot(...)
response = bot("What is the weather?")
print(bot.run_meta)  # Access all metrics
```

## Best Practices

1. **Monitor Tool Usage**: Track the success/failure rates and duration of tool calls to identify potential issues or optimization opportunities.
2. **Watch Planning Metrics**: For `AgentBot`, monitor planning success rates and revision counts to ensure efficient task decomposition.
3. **Track Validation**: For `StructuredBot`, keep an eye on validation attempts and success rates to catch schema issues early.
4. **Query Performance**: Monitor retrieval times and result counts in `QueryBot` to optimize document store performance.
5. **Message Flow**: Use message counts to understand the conversation flow and identify potential bottlenecks.

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
