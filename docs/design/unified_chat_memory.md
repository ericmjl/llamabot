# Unified Chat Memory Design

## Overview

This document outlines the design for a unified chat memory system that consolidates linear and graph-based memory into a single, configurable class. The system separates storage and retrieval concerns while providing multiple API levels for different use cases.

## Quick Start

```python
import llamabot as lmb

# Simple linear memory (fast, no LLM calls)
memory = lmb.ChatMemory()

# Intelligent graph memory with threading
memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")

# Use with any bot
bot = lmb.SimpleBot(system_prompt="You are helpful", memory=memory)
response = bot("Hello!")  # Memory automatically stores and retrieves context
```

## Core Design Principles

1. **Unified Interface**: Single class handles both linear and graph-based memory
2. **Separation of Concerns**: Storage operations (append) are separate from retrieval operations (search/context)
3. **Configuration at Instantiation**: Memory mode and behavior are set once and never change
4. **NetworkX Backend**: Direct use of NetworkX for graph operations without over-abstraction
5. **Optional Summarization**: Summarization is optional and can be disabled for performance

## Key Concepts

### Linear vs Graph Memory

| Feature | Linear Memory | Graph Memory |
|---------|---------------|--------------|
| **Speed** | Fast (no LLM calls) | Slower (LLM for threading) |
| **Intelligence** | Simple (last N messages) | Smart (semantic threading) |
| **Use Case** | Simple conversations | Complex multi-threaded chats |
| **LLM Calls** | None | 1-2 per message (optional) |

### Conversation Threading

**Linear Memory**: Messages are stored in order, retrieved as recent history
```
H1 → A1 → H2 → A2 → H3 → A3
```

**Graph Memory**: Messages are intelligently connected based on content
```
H1: "Let's talk about Python"
└── A1: "Python is great for data science"
    ├── H2: "What about machine learning?" → A2: "ML libraries include..."
    └── H3: "Tell me about databases" → A3: "SQL databases are..."
```

## Architecture

### API Levels

#### High-Level API (Opinionated)
```python
# Default linear memory
memory = ChatMemory()  # Uses LinearNodeSelector by default

# Graph memory with LLM-based threading
memory = ChatMemory.threaded(model="gpt-4o-mini")
```

#### Low-Level API (Full Configurability)
```python
memory = ChatMemory(
    node_selector=LLMNodeSelector(model="gpt-4o-mini"),
    summarizer=LLMSummarizer(model="gpt-4o-mini"),  # Optional
    context_depth=5  # Default context depth for retrieval
)
```

#### Factory Methods Implementation
```python
@classmethod
def threaded(cls, model: str = "gpt-4o-mini", **kwargs) -> "ChatMemory":
    """Create ChatMemory with LLM-based threading.

    :param model: LLM model name for node selection and summarization
    :param kwargs: Additional arguments passed to ChatMemory constructor
    """
    return cls(
        node_selector=LLMNodeSelector(model=model),
        summarizer=LLMSummarizer(model=model),  # Optional but recommended for threading
        **kwargs
    )
```

#### Under the Hood: What `.threaded()` Actually Does

When you call `ChatMemory.threaded(model="gpt-4o-mini")`, here's exactly what happens:

```python
# 1. Factory method creates LLMNodeSelector
llm_selector = LLMNodeSelector(model="gpt-4o-mini")
# This creates a selector that will use GPT-4o-mini to choose conversation threads

# 2. Factory method creates LLMSummarizer
llm_summarizer = LLMSummarizer(model="gpt-4o-mini")
# This creates a summarizer that will generate message summaries for better threading

# 3. Factory method calls the main constructor
memory = ChatMemory(
    node_selector=llm_selector,
    summarizer=llm_summarizer,
    context_depth=5  # Default value
)

# 4. Constructor initializes the memory system
def __init__(self, node_selector, summarizer, context_depth=5):
    self.graph = nx.DiGraph()  # Empty conversation graph
    self.node_selector = llm_selector  # Will use LLM for thread selection
    self.summarizer = llm_summarizer   # Will generate message summaries
    self.context_depth = context_depth # How far back to look for context
    self._next_node_id = 1             # Start numbering nodes from 1
```

**Result**: You get a `ChatMemory` instance that:
- Uses LLM-based intelligent threading instead of linear memory
- Automatically generates message summaries for better thread selection
- Maintains a conversation graph with parent-child relationships
- Can retrieve context by traversing conversation threads

**Equivalent Manual Creation:**
```python
# This is exactly what .threaded() does internally
memory = ChatMemory(
    node_selector=LLMNodeSelector(model="gpt-4o-mini"),
    summarizer=LLMSummarizer(model="gpt-4o-mini"),
    context_depth=5
)
```

**Note:** We chose the factory method pattern over alternatives like constructor with mode parameters or separate classes. The factory pattern provides clearer intent through descriptive method names while keeping the `__init__` method clean and focused on low-level configuration. This approach makes the API more readable and maintainable, especially as we add more memory modes and configuration options.

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

**Key Points:**
- Each node represents a single message (human or assistant)
- **id**: Auto-incremented integer providing natural ordering
- **parent_id**: Creates threading relationships (None for root)
- **message**: Contains role information (human/assistant) via BaseMessage
- **timestamp**: Metadata for when the message was created
- **summary**: Optional for performance
- Immutable once created

#### MessageSummary
```python
class MessageSummary(BaseModel):
    title: str = Field(..., description="Title of the message")
    summary: str = Field(..., description="Summary of the message. Two sentences max.")
```

## Usage Examples

### Basic SimpleBot with Linear Memory

```python
import llamabot as lmb

# Create a bot with simple linear memory (fast, no LLM calls)
memory = lmb.ChatMemory()  # Default linear
bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    memory=memory
)

# Chat loop automatically uses memory
response1 = bot("Hello! How are you?")
response2 = bot("What did I just ask you?")  # Bot can reference previous conversation
```

### SimpleBot with Graph Memory

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

### Custom Memory Configuration

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

### Memory Retrieval in Chat Loop

```python
import llamabot as lmb

# Bot with memory that can retrieve context
memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")
bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    memory=memory
)

# Simulate a conversation
bot("I'm working on a Python project.")
bot("I need to handle file I/O.")
bot("What's the best way to read CSV files?")
bot("Can you remind me what we discussed about file I/O?")  # Bot retrieves relevant context
```

### Memory Export and Visualization

```python
from llamabot.bot.simplebot import SimpleBot
from llamabot.components.chat_memory import ChatMemory

# Create bot with graph memory
memory = ChatMemory.threaded(model="gpt-4o-mini")
bot = SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    memory=memory
)

# Have a conversation
bot("Let's discuss Python.")
bot("What about data structures?")
bot("Now let's talk about machine learning.")
bot("What ML libraries work well with Python?")

# Export conversation graph
mermaid_diagram = memory.to_mermaid()
print(mermaid_diagram)

# Get conversation statistics
print(f"Total messages: {len(memory.graph.nodes())}")
print(f"Conversation threads: {len([n for n in memory.graph.nodes() if memory.graph.out_degree(n) == 0])}")
```

### When to Use Each Memory Type

| Use Case | Memory Type | Why |
|----------|-------------|-----|
| **Simple Q&A** | `lmb.ChatMemory()` | Fast, no LLM calls needed |
| **Multi-topic conversations** | `lmb.ChatMemory.threaded()` | Smart threading connects related topics |
| **Performance critical** | `lmb.ChatMemory()` | No additional LLM latency |
| **Complex discussions** | `lmb.ChatMemory.threaded()` | Maintains conversation context across topics |
| **Real-time chat** | `lmb.ChatMemory()` | Immediate responses |
| **Research/analysis** | `lmb.ChatMemory.threaded()` | Can reference earlier parts of conversation |

### Memory in Different Bot Types

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
structured_bot = StructuredBot(
    system_prompt="You are a helpful assistant.",
    pydantic_model=SomeModel,
    memory=custom_memory
)
```

### Memory Reset and State Management

```python
from llamabot.bot.simplebot import SimpleBot
from llamabot.components.chat_memory import ChatMemory

# Create bot with memory
memory = ChatMemory()  # Default linear
bot = SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    memory=memory
)

# Have a conversation
bot("Hello!")
bot("How are you?")

# Reset memory for new conversation
memory.reset()

# Bot no longer remembers previous conversation
response = bot("What did we just talk about?")  # Bot won't remember
```

## Core Operations

### Storage Operations

#### append(human_message: BaseMessage, assistant_message: BaseMessage)
- Adds both messages to the graph
- Creates parent-child relationship between messages
- Uses node selector to determine threading (linear vs graph mode)
- Prevents cycles and orphaned nodes



#### reset()
- Clears all stored messages
- Resets graph to empty state

### Retrieval Operations

#### retrieve(query: str, n_results: int = 10, context_depth: int = 5) -> List[BaseMessage]
- Smart retrieval that adapts based on memory configuration
- **Linear memory**: Ignores query, returns recent messages (fast)
- **Graph memory**: Uses semantic search with BM25 or similar algorithm, then traverses up thread paths
- **n_results**: Number of relevant messages to find via semantic search
- **context_depth**: Number of nodes to traverse up each thread path for context
- Returns most relevant messages with their conversation context
- Works like existing docstore implementations

**Context Depth Example:**
```
H1: "Let's talk about Python" (root)
└── A1: "Python is great for data science"
    ├── H2: "What about machine learning?"
    │   └── A2: "ML libraries include scikit-learn"
    └── H3: "Tell me about databases"
        └── A3: "SQL databases are..."

# Thread path for A2: A2 ← H2 ← A1 ← H1 (root)
memory.retrieve(query="machine learning", n_results=1, context_depth=2)
# Returns: [A2, H2, A1] (relevant message + 2 messages up thread path)
```

## Threading Model

### Conversation Structure
The graph memory uses a **tree structure** where all nodes are connected in a single conversation tree:

```
H1: "Let's talk about Python" (root - first human message)
└── A1: "Python is great for data science"
    ├── H2: "What about machine learning?"
    │   └── A2: "ML libraries include scikit-learn"
    └── H3: "Tell me about databases"
        └── A3: "SQL databases are..."

# Thread paths:
# Thread 1: H1 → A1 → H2 → A2
# Thread 2: H1 → A1 → H3 → A3
```

### Threading Rules

1. **Root Node**: The first human message becomes the root of the conversation tree
2. **Branching Rules**:
   - **Human messages** can only branch from **assistant messages**
   - **Assistant messages** can only branch from **human messages**
   - This enforces the conversation turn structure: Human → Assistant → Human → Assistant...
3. **Thread Definition**: Threads are paths from root to leaf nodes (active conversation endpoints)
   - **Leaf nodes**: Nodes with no out-edges (no children)
   - **Root node**: First human message with no parent (parent_id = None)

### Node Selection Strategies

#### LinearNodeSelector
- Always selects the leaf assistant node (node with no out-edges that is an assistant message)
- Creates linear conversation flow
- No LLM calls required
- Used in linear mode
- **Constraint**: Can only select assistant messages as parents for human messages

#### LLMNodeSelector
- Uses LLM to intelligently select which assistant message to branch from
- Considers message content and conversation context
- Supports retry logic with feedback
- Used in graph mode
- **Constraint**: Can only select assistant messages as parents for human messages
- **First message handling**: If no assistant messages exist, creates root node (parent_id = None)

#### Node Selection Logic
- **First message**: If no assistant nodes exist, message becomes root (`parent_id = None`)
- **Subsequent messages**: LLM selects best assistant message as parent
- **No fallbacks**: LLM selection should be reliable; if it fails, message becomes root

## Usage Patterns

### Memory Usage in Chat Loop

The following example shows how memory is used inside a bot's `__call__` method. This is the standard pattern that all memory types should follow:

```python
def __call__(self, *human_messages):
    # 1. Process incoming messages
    processed_messages = to_basemessage(human_messages)

    # 2. RETRIEVAL: Get relevant context from memory
    memory_messages = []
    if self.memory:
        # Memory system handles the complexity internally
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

**Key Points:**
- **Retrieval happens before response generation** to provide context
- **Storage happens after response generation** to save the conversation turn
- **Memory is self-aware** - the `retrieve()` method automatically chooses the best strategy based on memory type
- **No mode checking needed** - bot implementers don't need to know about memory internals
- **Performance optimization is automatic** - linear memory skips expensive semantic search
- **Memory is optional** - bot works without memory, just with less context
- **Unified API** - same method calls work for all memory types

**Note:** This example shows the core memory operations with logging stripped out for clarity. Real implementations should include appropriate logging and error handling.

### Export and Visualization

#### Mermaid Export
```python
# Export conversation graph
mermaid_diagram = memory.to_mermaid()

# Filter by role for cleaner visualization
assistant_nodes = [n for n in memory.graph.nodes()
                  if memory.graph.nodes[n]['node'].message.role == 'assistant']
```

## Implementation Details

### Modular Architecture

The implementation uses a modular approach to keep the main `ChatMemory` class clean and focused:

```
llamabot/components/chat_memory/
├── __init__.py        # Exports main classes and functions
├── memory.py          # Main ChatMemory class
├── retrieval.py       # Retrieval functions
├── storage.py         # Storage functions
├── visualization.py   # Visualization functions
└── selectors.py       # Node selection strategies
```

**Module Exports in `__init__.py`:**
```python
# Main classes
from .memory import ChatMemory
from .selectors import LinearNodeSelector, LLMNodeSelector
from .storage import append_linear, append_with_threading
from .retrieval import get_recent_messages, semantic_search_with_context
from .visualization import to_mermaid

__all__ = [
    "ChatMemory",
    "LinearNodeSelector",
    "LLMNodeSelector",
    "append_linear",
    "append_with_threading",
    "get_recent_messages",
    "semantic_search_with_context",
    "to_mermaid"
]
```

**Test Structure Should Mirror Components:**
```
tests/components/chat_memory/
├── test_memory.py     # Test main ChatMemory class
├── test_retrieval.py  # Test retrieval functions
├── test_storage.py    # Test storage functions
├── test_visualization.py  # Test visualization functions
└── test_selectors.py  # Test node selection strategies
```

#### Main Class (Clean and Focused)
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
        # Initialize NetworkX graph for storage
        self.graph = nx.DiGraph()

        # Set node selector (linear by default, LLM-based if provided)
        self.node_selector = node_selector or LinearNodeSelector()

        # Set optional summarizer
        self.summarizer = summarizer

        # Validate and store context depth
        if context_depth < 0:
            raise ValueError("context_depth must be non-negative")
        self.context_depth = context_depth

        # Track next node ID for auto-incrementing
        self._next_node_id = 1

    def retrieve(self, query: str, n_results: int = 10, context_depth: int = None) -> List[BaseMessage]:
        """Smart retrieval that adapts based on memory configuration."""
        context_depth = context_depth or self.context_depth

        if isinstance(self.node_selector, LinearNodeSelector):
            return get_recent_messages(self.graph, n_results)
        else:
            return semantic_search_with_context(self.graph, query, n_results, context_depth)

    def append(self, human_message: BaseMessage, assistant_message: BaseMessage):
        """Add conversation turn to memory."""
        if isinstance(self.node_selector, LinearNodeSelector):
            append_linear(self.graph, human_message, assistant_message, self._next_node_id)
            self._next_node_id += 2  # Increment for both messages
        else:
            append_with_threading(self.graph, human_message, assistant_message, self.node_selector, self._next_node_id)
            self._next_node_id += 2  # Increment for both messages
```

#### Separate Functions (Implementation Details)
```python
# retrieval.py
def get_recent_messages(graph: nx.DiGraph, n_results: int) -> List[BaseMessage]:
    """Get the most recent N messages from linear memory."""

def semantic_search_with_context(graph: nx.DiGraph, query: str, n_results: int, context_depth: int) -> List[BaseMessage]:
    """Find relevant nodes via semantic search, then traverse up their thread paths for context."""

def traverse_thread_path(graph: nx.DiGraph, node: int, depth: int) -> List[BaseMessage]:
    """Traverse up a conversation thread path from a given node."""

# storage.py
def append_linear(graph: nx.DiGraph, human_message: BaseMessage, assistant_message: BaseMessage, next_node_id: int):
    """Append messages to linear memory."""
    # Create human node
    human_node = ConversationNode(
        id=next_node_id,
        message=human_message,
        parent_id=find_leaf_assistant_node(graph) if graph.nodes() else None
    )
    graph.add_node(next_node_id, node=human_node)

    # Create assistant node
    assistant_node = ConversationNode(
        id=next_node_id + 1,
        message=assistant_message,
        parent_id=next_node_id
    )
    graph.add_node(next_node_id + 1, node=assistant_node)

    # Add edges
    if human_node.parent_id:
        graph.add_edge(human_node.parent_id, next_node_id)
    graph.add_edge(next_node_id, next_node_id + 1)

def append_with_threading(graph: nx.DiGraph, human_message: BaseMessage, assistant_message: BaseMessage, node_selector, next_node_id: int):
    """Append messages with intelligent threading following conversation turn structure."""
    # Use node selector to find best parent for human message
    parent_id = node_selector.select_parent(graph, human_message)

    # Create human node
    human_node = ConversationNode(
        id=next_node_id,
        message=human_message,
        parent_id=parent_id
    )
    graph.add_node(next_node_id, node=human_node)

    # Create assistant node
    assistant_node = ConversationNode(
        id=next_node_id + 1,
        message=assistant_message,
        parent_id=next_node_id
    )
    graph.add_node(next_node_id + 1, node=assistant_node)

    # Add edges
    if parent_id:
        graph.add_edge(parent_id, next_node_id)
    graph.add_edge(next_node_id, next_node_id + 1)

# visualization.py
def to_mermaid(graph: nx.DiGraph, **kwargs) -> str:
    """Convert graph to Mermaid diagram."""
```

**Benefits:**
- **Cleaner main class** - focuses on high-level API
- **Easier testing** - can test functions independently
- **Better separation of concerns** - each function has one job
- **More modular** - functions can be reused or swapped
- **Easier to understand** - main class shows the "what", functions show the "how"

**Testing Benefits:**
- **Unit tests** for each function in isolation
- **Integration tests** for the main ChatMemory class
- **Mock testing** of LLM components without real API calls
- **Test coverage** for each component independently
- **Regression testing** when modifying individual functions

**Import Benefits:**
- **Clean imports**: `from llamabot.components.chat_memory import ChatMemory`
- **Function access**: `from llamabot.components.chat_memory import append_linear`
- **Selector access**: `from llamabot.components.chat_memory import LLMNodeSelector`
- **Top-level exports**: All main functionality available from module root

### NetworkX Backend
- Direct use of NetworkX DiGraph for storage
- No abstraction layer needed
- Leverages NetworkX algorithms for graph operations
- Efficient for small to medium conversation graphs

### Implementation Details

#### Auto-Incremented IDs
- Node IDs start at 1 and increment for each new node
- Provides natural chronological ordering
- Simple integer-based identification
- No UUID complexity or collision concerns

#### NetworkX Graph Storage
- Each node stores a `ConversationNode` object as node data
- Node ID is the NetworkX node identifier
- Edges represent parent-child relationships
- Graph maintains conversation tree structure

#### Node Selection Process
1. **Linear Mode**: Find leaf assistant node (no out-edges, role="assistant")
2. **Graph Mode**:
   - Get all assistant nodes as candidates
   - Use LLM to select best parent based on message content
   - Validate selection is an assistant node
   - If no candidates exist, message becomes root

### Error Handling

The system uses **actionable error handling** - only raising errors for issues that humans can actually fix:

#### Actionable Errors (User Can Fix)
- **Configuration errors**: Invalid parameters like negative `context_depth`
- **File system errors**: Permission denied, disk full, invalid file paths
- **Input validation**: Wrong message types, empty message content

#### Graceful Handling (No Errors)
- **Empty memory**: Returns empty list instead of error
- **LLM selection failures**: Falls back to most recent valid node
- **Summarization failures**: Continues without summary
- **Graph corruption**: Clear error message with reset instruction

#### Error Message Examples

**Configuration Error (Actionable):**
```python
# Validated at instantiation
if context_depth < 0:
    raise ValueError("context_depth must be non-negative")
```

**File System Error (Actionable):**
```python
if "Permission denied" in str(e):
    raise PersistenceError(
        f"Cannot save to {file_path}. Check file permissions or choose a different location."
    )
elif "No space left" in str(e):
    raise PersistenceError(
        f"Disk is full. Free up space or choose a different location."
    )
```

**Graph Corruption (Actionable):**
```python
raise InvalidGraphStateError(
    "Conversation graph has become corrupted. "
    "This can happen if the same message was processed multiple times. "
    "Use memory.reset() to clear the conversation and start fresh."
)
```

**Graceful Handling Examples:**
```python
# Empty memory - no error, just empty result
if not graph.nodes():
    return []

# LLM failure - fallback to most recent node
if llm_response not in valid_candidates:
    return valid_candidates[-1] if valid_candidates else None

# Summarization failure - continue without summary
try:
    summary = summarizer.summarize(message)
except Exception:
    summary = None
```

### Performance Considerations
- Optional summarization for linear mode
- Lazy loading of summaries when needed
- Efficient graph traversal for retrieval
- Memory-efficient storage of large conversations

## Benefits

1. **Simplified API**: Single class for all memory operations
2. **Better Performance**: Optional summarization reduces LLM calls
3. **Clearer Separation**: Storage and retrieval are distinct concerns
4. **Easier Testing**: Smaller, focused components
5. **Future Extensibility**: Pluggable node selectors and retrieval strategies
6. **Type Safety**: Clear interfaces and error handling

## Persistence Design

### Storage Format

The conversation memory uses a **JSON-based format** for persistence that is easily parseable and human-readable:

```json
{
  "version": "1.0",
  "metadata": {
    "created_at": "2024-01-15T10:30:00Z",
    "last_modified": "2024-01-15T14:45:00Z",
    "mode": "graph",
    "total_messages": 12
  },
  "nodes": [
    {
      "id": 1,
      "role": "user",
      "content": "Let's talk about Python",
      "timestamp": "2024-01-15T10:30:00Z",
      "summary": {
        "title": "Python Discussion Start",
        "summary": "User wants to discuss Python programming."
      },
      "parent_id": null
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "Python is great for data science",
      "timestamp": "2024-01-15T10:30:05Z",
      "summary": {
        "title": "Python Benefits",
        "summary": "Assistant explains Python's benefits for data science."
      },
      "parent_id": 1
    }
  ],
  "edges": [
    {"from": 1, "to": 2},
    {"from": 2, "to": 3},
    {"from": 2, "to": 5}
  ]
}
```

### Persistence Operations

#### save(file_path: str) -> None
- Serializes conversation memory to JSON file
- Includes metadata for versioning and tracking
- Preserves all node data and edge relationships
- Handles BaseMessage serialization

#### load(file_path: str) -> ChatMemory
- Deserializes JSON file to recreate memory
- Validates graph structure integrity
- Reconstructs NetworkX graph from JSON data
- Handles version compatibility

#### export(format: str = "json") -> str
- Exports conversation in various formats
- **JSON**: Full conversation with metadata
- **JSONL**: OpenAI-compatible format for fine-tuning
- **Mermaid**: Visualization format
- **Plain text**: Simple conversation transcript

### Implementation Details

#### JSON Serialization Strategy
```python
def to_json(self) -> dict:
    """Convert conversation memory to JSON-serializable dict."""
    return {
        "version": "1.0",
        "metadata": {
            "created_at": self.created_at.isoformat(),
            "last_modified": datetime.now().isoformat(),
            "mode": self.mode,
            "total_messages": len(self.graph.nodes())
        },
        "nodes": [
            {
                "id": node_id,
                "role": node_data["node"].message.role,
                "content": node_data["node"].message.content,
                "timestamp": node_data["node"].timestamp.isoformat(),
                "summary": node_data["node"].summary.dict() if node_data["node"].summary else None,
                "parent_id": node_data["node"].parent_id
            }
            for node_id, node_data in self.graph.nodes(data=True)
        ],
        "edges": [
            {"from": u, "to": v}
            for u, v in self.graph.edges()
        ]
    }
```

#### Graph Reconstruction
```python
def from_json(data: dict) -> ChatMemory:
    """Reconstruct conversation memory from JSON data."""
    memory = ChatMemory(mode=data["metadata"]["mode"])

    # Reconstruct nodes
    for node_data in data["nodes"]:
        message = create_message(node_data["role"], node_data["content"])
        node = ConversationNode(
            id=node_data["id"],
            message=message,
            summary=MessageSummary(**node_data["summary"]) if node_data["summary"] else None,
            parent_id=node_data["parent_id"],
            timestamp=datetime.fromisoformat(node_data["timestamp"])
        )
        memory.graph.add_node(node_data["id"], node=node)

    # Reconstruct edges
    for edge in data["edges"]:
        memory.graph.add_edge(edge["from"], edge["to"])

    return memory
```

### Benefits of JSON Format

1. **Human-readable**: Easy to inspect and debug
2. **Version control friendly**: Diff-able and merge-able
3. **Language agnostic**: Can be parsed by any language
4. **Extensible**: Easy to add new fields
5. **Standard format**: Well-supported across tools
6. **No security risks**: Unlike pickle, no code execution

### File Naming Convention

```
conversations/
├── session_2024-01-15_10-30-00.json
├── session_2024-01-15_14-45-00.json
└── backup_2024-01-15_18-00-00.json
```

## Open Questions and Future Enhancements

### Concurrency Handling
- How should multiple threads/processes access the same memory file?
- Should we use file locking or database backend for concurrent access?
- What happens if two processes try to append simultaneously?

### Advanced Retrieval Strategies
- Semantic search across message content
- Time-based retrieval (messages from last hour/day)
- User-specific retrieval (only messages from specific user)
- Context-aware retrieval (messages related to current topic)

### Performance Optimizations
- Lazy loading of large conversation histories
- Caching frequently accessed message paths
- Compression for long conversations
- Incremental graph updates

### Integration with External Systems
- Export to chat platforms (Slack, Discord, etc.)
- Integration with vector databases for semantic search
- Webhook support for real-time updates
- API endpoints for external access

## Migration Strategy

### Phase 1: Deprecation (v0.13.0)
- Add deprecation warnings to existing `ChatMemory` class
- Document new `ChatMemory` API
- Update examples to use new API
- **Ensure `ChatMemory` is top-level in `llamabot/__init__.py`** ✅
- **Update tests to reflect modular component structure** ✅

### Phase 2: Transition (v0.14.0)
- Make `ChatMemory` the default
- Keep `ChatMemory` as alias with deprecation warning
- Update all internal usage

### Phase 3: Removal (v0.15.0)
- Remove `ChatMemory` class entirely
- Remove deprecated methods
- Clean up imports and references

### Migration Guide

**Old API:**
```python
from llamabot.components.chat_memory import ChatMemory

memory = ChatMemory()
memory.add_message("user", "Hello")
messages = memory.get_messages()
```

**New API:**
```python
from llamabot.components.chat_memory import ChatMemory

memory = ChatMemory()
memory.append("user", "Hello")
messages = memory.retrieve()
```

**Key Changes:**
- `ChatMemory` → `ChatMemory` (same name, new implementation)
- `add_message()` → `append()`
- `get_messages()` → `retrieve()`
- New threading support with `ChatMemory.threaded()`
- New persistence methods: `save()`, `load()`, `export()`

**Imports:**

```python
import llamabot as lmb

memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")
```

## Conclusion

This unified design addresses the core issues with the current implementation while maintaining the flexibility needed for different use cases. The separation of storage and retrieval concerns makes the system more maintainable and easier to understand, while the multiple API levels provide the right level of abstraction for different users.
