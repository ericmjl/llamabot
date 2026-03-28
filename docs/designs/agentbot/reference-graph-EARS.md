# AgentBot reference graph — EARS

**Parent LLD**: [./LLD.md](./LLD.md)

## Reference graph topology

- [x] **AGT-GRAPH-001**: The system shall provide a class `AgentBot` that constructs a PocketFlow graph whose unique entry node is a decision node that selects at most one tool per decision step from the union of default tools and user-supplied tools.

- [x] **AGT-GRAPH-002**: When the user invokes `AgentBot` with a query string, the system shall append that query to `shared["memory"]` and run the flow until a terminal tool completes or execution stops.

- [x] **AGT-GRAPH-003**: While `max_iterations` is set and the iteration limit is exceeded, the decision node shall force routing consistent with the `respond_to_user` tool when that tool is present in the tool list.

- [x] **AGT-GRAPH-004**: Where a custom `decide_node` is supplied, the system shall use it in place of the default `DecideNode` and shall still attach tool nodes and edges for routing names that match the tool functions.

## Decision step (ToolBot / AsyncToolBot)

- [x] **AGT-GRAPH-010**: The system shall implement `DecideNode.exec` such that the routing step uses `ToolBot`, and `DecideNode.aexec` such that the routing step uses `AsyncToolBot`, each with the same tool list and model configuration as the `AgentBot` instance (subject to Ollama-specific prompt and `tool_choice` adjustments).

- [x] **AGT-GRAPH-011**: The default `AgentBot` graph shall not invoke `QueryBot` or `StructuredBot` unless that logic is introduced inside a tool implementation or a custom decision node.

## Async variant

- [x] **AGT-GRAPH-020**: Where `AsyncAgentBot` is used, the system shall run the same graph topology with `AsyncFlow` and shall perform the decision step via **`AsyncToolBot.__call__`** (LiteLLM ``acompletion``), not sync **`ToolBot.__call__`** on a worker thread.

- [x] **AGT-GRAPH-021**: The system shall not expose a separate async entrypoint (e.g. ``acall``) on **`ToolBot`**; async tool selection shall live on **`AsyncToolBot.__call__`** only.

## Related Documents

- [AgentBot LLD](./LLD.md)
- [High-Level Design](../../high-level-design.md)
