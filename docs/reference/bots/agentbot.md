# AgentBot API Reference

AgentBot is a graph-based agent that uses PocketFlow for tool orchestration. It automatically executes tools in a multi-step workflow based on user requests.

## Class Definition

```python
class AgentBot:
    """An AgentBot that uses PocketFlow for tool orchestration.

    This bot requires user-provided tools to be decorated with @tool. It creates
    a decision node that uses ToolBot to select tools and executes them through
    a PocketFlow graph.
    """
```

## Constructor

```python
def __init__(
    self,
    tools: List[Callable],
    decide_node: Optional[DecideNode] = None,
    system_prompt: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    chat_memory: Optional[ChatMemory] = None,
    **completion_kwargs,
)
```

### Parameters

- **tools** (`List[Callable]`): List of tools decorated with `@tool`. These tools will be automatically wrapped as PocketFlow nodes and connected to the decision node.

- **decide_node** (`Optional[DecideNode]`, default: `None`): Optional custom decision node. If provided, overrides `system_prompt` parameter. If not provided, a default `DecideNode` is created.

- **system_prompt** (`Optional[str]`, default: `None`): System prompt string for decision-making. Only used if `decide_node` is not provided. Ignored if `decide_node` is provided.

- **model_name** (`str`, default: `"gpt-4o-mini"`): The name of the model to use for decision-making and tool selection.

- **chat_memory** (`Optional[ChatMemory]`, default: `None`): Chat memory component for maintaining conversation context.

- **completion_kwargs**: Additional keyword arguments to pass to the completion function.

### Tool Requirements

Tools must be decorated with `@tool` before being passed to AgentBot:

```python
from llamabot.components.tools import tool

@tool
def my_tool(arg: str) -> str:
    """Tool description."""
    return arg

bot = AgentBot(tools=[my_tool])
```

For terminal tools (like `respond_to_user`), use `@tool(loopback_name=None)`:

```python
@tool(loopback_name=None)
def respond_to_user(response: str) -> str:
    """Respond to the user."""
    return response
```

## Methods

### `__call__`

```python
def __call__(
    self,
    *messages: Union[str, BaseMessage],
) -> Any
```

Execute the agent workflow with the given messages.

#### Parameters

- **messages**: One or more messages to process. Can be strings or `BaseMessage` objects.

#### Returns

- **Any**: The result from the terminal node (usually a string response or object).

#### Example

```python
import llamabot as lmb

@lmb.tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return results

@lmb.tool(loopback_name=None)
def respond_to_user(response: str) -> str:
    """Respond to the user."""
    return response

bot = lmb.AgentBot(
    tools=[search_web, respond_to_user],
    model_name="gpt-4o-mini"
)

result = bot("Research the latest developments in AI and summarize them")
```

### `visualize`

```python
def visualize(self) -> str
```

Generate a Mermaid diagram representation of the agent's flow graph.

#### Returns

- **str**: Mermaid diagram code that can be rendered.

#### Example

```python
bot = lmb.AgentBot(tools=[my_tool1, my_tool2])
mermaid_diagram = bot.visualize()
print(mermaid_diagram)
```

## Default Tools

AgentBot automatically includes these default tools:

- **`today_date()`**: Returns the current date (loops back to decide node)
- **`respond_to_user(response: str)`**: Sends a text response to the user (terminal node, no loopback)
- **`return_object_to_user(variable_name: str)`**: Returns an object from the calling context's globals (terminal node, no loopback)
- **`inspect_globals()`**: Inspects available global variables (loops back to decide node)

## Flow Graph Structure

AgentBot creates a flow graph where:

1. **Decision Node** (`DecideNode`): Analyzes the conversation and selects which tool to execute
2. **Tool Nodes**: Execute the selected tools
3. **Loopback**: Tools can loop back to the decision node (except terminal tools)
4. **Terminal Nodes**: Tools with `loopback_name=None` end the workflow

## Attributes

- **flow** (`Flow`): The PocketFlow flow graph
- **decide_node** (`DecideNode`): The decision node
- **tools** (`List[Callable]`): List of tool functions

## Usage Examples

### Basic Multi-Step Workflow

```python
import llamabot as lmb

@lmb.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    :param city: The name of the city
    :return: Weather information
    """
    return f"The weather in {city} is sunny, 72°F"

@lmb.tool(loopback_name=None)
def respond_to_user(response: str) -> str:
    """Respond to the user."""
    return response

agent = lmb.AgentBot(
    tools=[get_weather, respond_to_user],
    model_name="gpt-4o-mini"
)

result = agent("What's the weather in New York?")
```

### With Custom System Prompt

```python
import llamabot as lmb

@lmb.tool
def analyze_data(data: str) -> str:
    """Analyze data."""
    return "Analysis complete"

agent = lmb.AgentBot(
    tools=[analyze_data],
    system_prompt="You are a data analysis expert.",
    model_name="gpt-4o-mini"
)
```

### With Chat Memory

```python
import llamabot as lmb

memory = lmb.ChatMemory()

@lmb.tool
def my_tool(arg: str) -> str:
    return arg

agent = lmb.AgentBot(
    tools=[my_tool],
    chat_memory=memory,
    model_name="gpt-4o-mini"
)
```

### Visualizing the Flow Graph

```python
import llamabot as lmb

@lmb.tool
def step1(data: str) -> str:
    return "Step 1 complete"

@lmb.tool
def step2(data: str) -> str:
    return "Step 2 complete"

agent = lmb.AgentBot(tools=[step1, step2])
mermaid_diagram = agent.visualize()
print(mermaid_diagram)
```

## Differences from ToolBot

- **AgentBot**: Multi-turn planning, automatically executes tools in a graph
- **ToolBot**: Single-turn execution, returns tool calls for you to execute

## Observability

Span-based logging is enabled by default for all tools decorated with `@tool`. You can customize span attributes:

```python
@lmb.tool(exclude_args=["api_key"], operation_name="custom_name")
def my_tool(api_key: str, data: str) -> str:
    """Tool with custom span configuration."""
    return result
```

## Related Classes

- **ToolBot**: Single-turn tool execution bot
- **DecideNode**: Decision-making node component
- **Tools Module**: Tool decorator and utilities

## See Also

- [AgentBot Tutorial](../../tutorials/agentbot.md)
- [Which Bot Should I Use?](../../getting-started/which-bot.md)
- [Tools Component](../components/tools.md)
