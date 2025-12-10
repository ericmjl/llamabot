# ChatMemory API Reference

ChatMemory provides intelligent conversation memory with configurable threading and retrieval strategies.

## Class Definition

```python
class ChatMemory:
    """Unified chat memory system with configurable threading and retrieval.

    :param node_selector: Strategy for selecting parent nodes (None = LinearNodeSelector)
    :param summarizer: Optional summarization strategy (None = no summarization)
    :param context_depth: Default depth for context retrieval
    """
```

## Constructor

```python
def __init__(
    self,
    node_selector: Optional[NodeSelector] = None,
    summarizer: Optional[Summarizer] = None,
    context_depth: int = 5,
)
```

### Parameters

- **node_selector** (`Optional[NodeSelector]`, default: `None`): Strategy for selecting parent nodes when adding messages. If `None`, uses `LinearNodeSelector` (linear memory). For intelligent threading, use `LLMNodeSelector`.

- **summarizer** (`Optional[Summarizer]`, default: `None`): Optional summarization strategy. If `None`, no summarization is performed. For threaded memory, use `LLMSummarizer`.

- **context_depth** (`int`, default: `5`): Default depth for context retrieval when traversing the conversation graph.

## Class Methods

### `threaded`

```python
@classmethod
def threaded(cls, model: str = "gpt-4o-mini", **kwargs) -> "ChatMemory"
```

Create ChatMemory with LLM-based threading.

#### Parameters

- **model** (`str`, default: `"gpt-4o-mini"`): LLM model name for node selection and summarization
- **kwargs**: Additional arguments passed to ChatMemory constructor

#### Returns

- **ChatMemory**: A ChatMemory instance with LLM-based threading enabled

#### Example

```python
import llamabot as lmb

# Create threaded memory
memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")
```

## Methods

### `append`

```python
def append(self, message: BaseMessage) -> None
```

Append a message to the memory.

#### Parameters

- **message** (`BaseMessage`): The message to append (HumanMessage, AIMessage, etc.)

#### Example

```python
import llamabot as lmb
from llamabot.components.messages import HumanMessage, AIMessage

memory = lmb.ChatMemory()
memory.append(HumanMessage(content="Hello"))
memory.append(AIMessage(content="Hi there!"))
```

### `retrieve`

```python
def retrieve(
    self,
    query: str,
    n_results: int = 10,
    context_depth: Optional[int] = None,
) -> List[BaseMessage]
```

Smart retrieval that adapts based on memory configuration.

#### Parameters

- **query** (`str`): The search query
- **n_results** (`int`, default: `10`): Number of results to return
- **context_depth** (`Optional[int]`, default: `None`): Context depth (uses default if None)

#### Returns

- **List[BaseMessage]**: List of relevant messages

#### Behavior

- **Linear memory**: Returns the most recent `n_results` messages
- **Threaded memory**: Performs semantic search with context traversal up to `context_depth` levels

#### Example

```python
import llamabot as lmb

memory = lmb.ChatMemory()

# Add some messages
memory.append(HumanMessage(content="What is Python?"))
memory.append(AIMessage(content="Python is a programming language."))
memory.append(HumanMessage(content="Tell me more about it."))

# Retrieve relevant messages
relevant = memory.retrieve("programming language", n_results=5)
```

### `reset`

```python
def reset(self) -> None
```

Reset the memory, clearing all stored messages.

#### Example

```python
import llamabot as lmb

memory = lmb.ChatMemory()
memory.append(HumanMessage(content="Hello"))
memory.reset()  # Clear all messages
```

### `to_mermaid`

```python
def to_mermaid(self) -> str
```

Generate a Mermaid diagram representation of the conversation graph.

#### Returns

- **str**: Mermaid diagram code

#### Example

```python
import llamabot as lmb

memory = lmb.ChatMemory.threaded()
# ... add messages ...
mermaid_diagram = memory.to_mermaid()
print(mermaid_diagram)
```

## Attributes

- **graph** (`networkx.DiGraph`): The NetworkX graph storing conversation structure
- **node_selector** (`NodeSelector`): The node selection strategy
- **summarizer** (`Optional[Summarizer]`): The summarization strategy (if any)
- **context_depth** (`int`): Default context depth for retrieval

## Memory Types

### Linear Memory

Simple, fast memory that stores messages in order:

```python
import llamabot as lmb

memory = lmb.ChatMemory()  # Linear by default
```

**Characteristics:**

- Fast (no LLM calls)
- Stores messages sequentially
- Retrieves most recent messages
- Best for simple conversations

### Threaded Memory

Intelligent memory that connects related conversation topics:

```python
import llamabot as lmb

memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")
```

**Characteristics:**

- Uses LLM to connect related topics
- Creates conversation threads
- Semantic search for retrieval
- Best for complex, multi-topic conversations

## Usage Examples

### Basic Linear Memory

```python
import llamabot as lmb
from llamabot.components.messages import HumanMessage, AIMessage

memory = lmb.ChatMemory()

memory.append(HumanMessage(content="What is Python?"))
memory.append(AIMessage(content="Python is a programming language."))

# Retrieve recent messages
recent = memory.retrieve("", n_results=5)
```

### Threaded Memory

```python
import llamabot as lmb

memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")

memory.append(HumanMessage(content="What is Python?"))
memory.append(AIMessage(content="Python is a programming language."))
memory.append(HumanMessage(content="How do I install it?"))

# Semantic search finds related messages
relevant = memory.retrieve("programming language", n_results=5)
```

### With Bots

```python
import llamabot as lmb

# Create memory
memory = lmb.ChatMemory()

# Use with SimpleBot
bot = lmb.SimpleBot(
    system_prompt="You are helpful.",
    memory=memory,
    model_name="gpt-4o-mini"
)

# Bot automatically uses memory
response1 = bot("My name is Alice")
response2 = bot("What's my name?")  # Bot remembers!
```

### Visualizing Conversation Graph

```python
import llamabot as lmb

memory = lmb.ChatMemory.threaded()
# ... add messages ...

# Generate Mermaid diagram
diagram = memory.to_mermaid()
print(diagram)
```

## Related Classes

- **NodeSelector**: Strategy for selecting parent nodes
- **Summarizer**: Strategy for summarizing messages
- **LinearNodeSelector**: Linear memory selector
- **LLMNodeSelector**: LLM-based intelligent selector
- **LLMSummarizer**: LLM-based summarizer

## See Also

- [Which Bot Should I Use?](../../getting-started/which-bot.md)
- [SimpleBot Reference](../bots/simplebot.md)
