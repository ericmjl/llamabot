# Chat Memory

!!! note
    This module provides intelligent conversation memory with both linear and graph-based threading capabilities.

The chat memory system allows bots to maintain context across conversation turns, enabling more coherent and contextual responses. It supports two main modes:

- **Linear Memory**: Fast, simple memory that stores messages in chronological order
- **Graph Memory**: Intelligent threading that connects related conversation topics using LLM-based analysis

## Quick Start

### Basic Linear Memory

```python
import llamabot as lmb

# Create a bot with simple linear memory (fast, no LLM calls)
memory = lmb.ChatMemory()  # Default linear mode
bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    memory=memory
)

# Chat loop automatically uses memory
response1 = bot("Hello! How are you?")
response2 = bot("What did I just ask you?")  # Bot can reference previous conversation
```

### Intelligent Graph Memory

```python
import llamabot as lmb

# Create a bot with intelligent threading (uses LLM for smart connections)
memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")
bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    memory=memory
)

# Bot can now handle conversation threading intelligently
response1 = bot("Let's talk about Python programming.")
response2 = bot("What are the benefits of using Python?")  # Continues Python thread
response3 = bot("Now let's discuss machine learning.")  # Starts new thread
response4 = bot("What libraries should I use for ML in Python?")  # Connects back to Python thread
```

## Core Components

### ChatMemory

The main class that provides unified memory functionality.

```python
class ChatMemory:
    def __init__(self,
                 node_selector: Optional[NodeSelector] = None,
                 summarizer: Optional[Summarizer] = None,
                 context_depth: int = 5):
        """Initialize chat memory with configuration.

        :param node_selector: Strategy for selecting parent nodes (None = LinearNodeSelector)
        :param summarizer: Optional summarization strategy (None = no summarization)
        :param context_depth: Default depth for context retrieval
        """
```

**Factory Methods:**

- `ChatMemory()` - Creates linear memory (fast, no LLM calls)
- `ChatMemory.threaded(model="gpt-4o-mini")` - Creates graph memory with LLM-based threading

### Node Selectors

#### LinearNodeSelector
- Always selects the most recent assistant message as parent
- Creates linear conversation flow
- No LLM calls required
- Used by default in linear mode

#### LLMNodeSelector
- Uses LLM to intelligently select which assistant message to branch from
- Considers message content and conversation context
- Supports retry logic with feedback
- Used in graph mode

### Summarizers

#### LLMSummarizer
- Generates message summaries for better threading
- Optional component that can be disabled for performance
- Uses LLM to create concise summaries of message content

## Usage Patterns

### Memory in Chat Loop

The standard pattern for using memory in a bot:

```python
def __call__(self, *human_messages):
    # 1. Process incoming messages
    processed_messages = to_basemessage(human_messages)

    # 2. RETRIEVAL: Get relevant context from memory
    memory_messages = []
    if self.memory:
        memory_messages = self.memory.retrieve(
            query=f"From our conversation history, give me the most relevant information to the query, {[p.content for p in processed_messages]}",
            n_results=10,
            context_depth=5
        )

    # 3. Build message list with context
    messages = [self.system_prompt] + memory_messages + processed_messages

    # 4. Generate response
    response_message = AIMessage(content=content, tool_calls=tool_calls)

    # 5. STORAGE: Save conversation turn to memory
    if self.memory:
        self.memory.append(processed_messages[-1], response_message)

    return response_message
```

### Memory Operations

#### Storage
```python
# Add conversation turn to memory
memory.append(human_message, assistant_message)
```

#### Retrieval
```python
# Get relevant context
context_messages = memory.retrieve(
    query="What did we discuss about Python?",
    n_results=5,
    context_depth=3
)
```

#### Reset
```python
# Clear all stored messages
memory.reset()
```

### Visualization

```python
# Export conversation graph as Mermaid diagram
mermaid_diagram = memory.to_mermaid()
print(mermaid_diagram)
```

## When to Use Each Memory Type

| Use Case | Memory Type | Why |
|----------|-------------|-----|
| **Simple Q&A** | `lmb.ChatMemory()` | Fast, no LLM calls needed |
| **Multi-topic conversations** | `lmb.ChatMemory.threaded()` | Smart threading connects related topics |
| **Performance critical** | `lmb.ChatMemory()` | No additional LLM latency |
| **Complex discussions** | `lmb.ChatMemory.threaded()` | Maintains conversation context across topics |
| **Real-time chat** | `lmb.ChatMemory()` | Immediate responses |
| **Research/analysis** | `lmb.ChatMemory.threaded()` | Can reference earlier parts of conversation |

## Advanced Configuration

### Custom Memory Setup

```python
import llamabot as lmb

# Custom memory configuration (advanced users)
memory = lmb.ChatMemory(
    node_selector=lmb.LLMNodeSelector(model="gpt-4o-mini"),
    summarizer=lmb.LLMSummarizer(model="gpt-4o-mini"),  # Optional for better threading
    context_depth=10  # How far back to look for context
)

bot = lmb.SimpleBot(
    system_prompt="You are a coding assistant.",
    model_name="gpt-4o-mini",
    memory=memory
)
```

### Memory with Different Bot Types

```python
import llamabot as lmb

# Linear memory for simple conversations (fast)
linear_memory = lmb.ChatMemory()  # Default linear
simple_bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    memory=linear_memory
)

# Graph memory for complex conversations with threading (smart)
graph_memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")
query_bot = lmb.QueryBot(
    system_prompt="You are a helpful assistant.",
    memory=graph_memory
)

# Custom memory for specific needs (advanced)
custom_memory = lmb.ChatMemory(
    node_selector=lmb.LLMNodeSelector(model="gpt-4o-mini"),
    summarizer=None  # No summarization for performance
)
structured_bot = lmb.StructuredBot(
    system_prompt="You are a helpful assistant.",
    pydantic_model=SomeModel,
    memory=custom_memory
)
```

## Architecture

The chat memory system uses a modular architecture:

```
llamabot/components/chat_memory/
├── __init__.py        # Exports main classes and functions
├── memory.py          # Main ChatMemory class
├── retrieval.py       # Retrieval functions
├── storage.py         # Storage functions
├── visualization.py   # Visualization functions
└── selectors.py       # Node selection strategies
```

### Data Model

#### ConversationNode
```python
@dataclass
class ConversationNode:
    id: int  # Auto-incremented based on number of nodes in graph
    message: BaseMessage  # Single message (not conversation turn)
    summary: Optional[MessageSummary] = None
    parent_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
```

#### MessageSummary
```python
class MessageSummary(BaseModel):
    title: str = Field(..., description="Title of the message")
    summary: str = Field(..., description="Summary of the message. Two sentences max.")
```

## Threading Model

The graph memory uses a **tree structure** where all nodes are connected in a single conversation tree:

```
