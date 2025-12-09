# Messages API Reference

The messages system provides unified message types for LLM communication.

## BaseMessage

Base class for all message types.

```python
class BaseMessage(BaseModel):
    """A base message class."""

    role: str
    content: str
    prompt_hash: str | None = None
    tool_calls: list = []
```

### Attributes

- **role** (`str`): The role of the message sender (`"system"`, `"user"`, `"assistant"`, `"tool"`)
- **content** (`str`): The message content
- **prompt_hash** (`str | None`, default: `None`): Optional hash of the prompt
- **tool_calls** (`list`, default: `[]`): List of tool calls associated with the message

### Methods

#### `__len__`

```python
def __len__(self) -> int
```

Get the length of the message content.

#### `__getitem__`

```python
def __getitem__(self, index) -> BaseMessage
```

Get a slice of the message content.

#### `__add__` / `__radd__`

```python
def __add__(self, other: str) -> BaseMessage
def __radd__(self, other: str) -> BaseMessage
```

Concatenate strings with message content.

## Message Types

### SystemMessage

Message from the system (system prompt).

```python
class SystemMessage(BaseMessage):
    """A message from the system."""

    content: str
    role: str = "system"
```

#### Example

```python
from llamabot.components.messages import SystemMessage

msg = SystemMessage(content="You are a helpful assistant.")
```

### HumanMessage

Message from a human user.

```python
class HumanMessage(BaseMessage):
    """A message from a human."""

    content: str
    role: str = "user"
```

#### Example

```python
from llamabot.components.messages import HumanMessage

msg = HumanMessage(content="What is Python?")
```

### AIMessage

Message from the AI assistant.

```python
class AIMessage(BaseMessage):
    """A message from the AI."""

    content: str
    role: str = "assistant"
```

#### Example

```python
from llamabot.components.messages import AIMessage

msg = AIMessage(content="Python is a programming language.")
```

### DeveloperMessage

Message from the developer (for development context).

```python
class DeveloperMessage(BaseMessage):
    """A message from the developer."""

    content: str
    role: str = "developer"
```

#### Example

```python
from llamabot.components.messages import DeveloperMessage

msg = DeveloperMessage(content="Add error handling to this function.")
```

### ToolMessage

Message representing tool execution results.

```python
class ToolMessage(BaseMessage):
    """A message from a tool."""

    content: str
    role: str = "tool"
```

#### Example

```python
from llamabot.components.messages import ToolMessage

msg = ToolMessage(content="Tool execution result")
```

### RetrievedMessage

Message retrieved from document store or memory.

```python
class RetrievedMessage(BaseMessage):
    """A message retrieved from the history."""

    content: str
    role: str = "system"
```

#### Example

```python
from llamabot.components.messages import RetrievedMessage

msg = RetrievedMessage(content="Retrieved document content")
```

### ImageMessage

Message containing an image.

```python
class ImageMessage(BaseMessage):
    """A message containing an image."""

    content: str  # Base64-encoded image or file path
    role: str = "user"
```

#### Constructor

```python
def __init__(
    self,
    content: Union[str, Path],
    role: str = "user",
    prompt_hash: str | None = None,
    tool_calls: list = None,
)
```

#### Parameters

- **content** (`Union[str, Path]`): Path to image file, URL of image, or base64-encoded image
- **role** (`str`, default: `"user"`): Role of the message sender
- **prompt_hash** (`str | None`, default: `None`): Optional prompt hash
- **tool_calls** (`list`, default: `None`): Optional tool calls

#### Example

```python
from llamabot.components.messages import ImageMessage
from pathlib import Path

# From file path
msg = ImageMessage(content=Path("image.jpg"))

# From URL
msg = ImageMessage(content="https://example.com/image.jpg")
```

## Helper Functions

### `user`

```python
def user(*content: Union[str, Path]) -> Union[HumanMessage, ImageMessage, list]
```

Create user messages, automatically detecting images.

#### Parameters

- **content**: One or more strings or image paths

#### Returns

- **Union[HumanMessage, ImageMessage, list]**: Message(s) created from content

#### Example

```python
from llamabot.components.messages import user

# Text message
msg = user("What is this?")

# Image message
msg = user("/path/to/image.jpg")

# Text and image
msgs = user("What is this?", "/path/to/image.jpg")
```

### `dev`

```python
def dev(*content: str) -> Union[DeveloperMessage, list[DeveloperMessage]]
```

Create developer messages for development context.

#### Parameters

- **content**: One or more strings

#### Returns

- **Union[DeveloperMessage, list[DeveloperMessage]]**: Developer message(s)

#### Example

```python
from llamabot.components.messages import dev

msg = dev("Add error handling")
msgs = dev("Refactor code", "Add tests")
```

### `system`

```python
def system(content: str) -> SystemMessage
```

Create a system message.

#### Parameters

- **content** (`str`): System prompt content

#### Returns

- **SystemMessage**: System message

#### Example

```python
from llamabot.components.messages import system

msg = system("You are a helpful assistant.")
```

### `to_basemessage`

```python
def to_basemessage(
    messages: Union[
        str,
        BaseMessage,
        list[Union[str, BaseMessage]],
        Callable,
    ]
) -> list[BaseMessage]
```

Convert various input types to a list of BaseMessage objects.

#### Parameters

- **messages**: Can be:
  - String (converted to HumanMessage)
  - BaseMessage object
  - List of strings or BaseMessage objects
  - Callable function (called and result converted)

#### Returns

- **list[BaseMessage]**: List of BaseMessage objects

#### Example

```python
from llamabot.components.messages import to_basemessage, HumanMessage

# String
msgs = to_basemessage("Hello")

# Message object
msgs = to_basemessage(HumanMessage(content="Hello"))

# List
msgs = to_basemessage(["Hello", "World"])

# Mixed
msgs = to_basemessage([
    "Hello",
    HumanMessage(content="World")
])
```

## Usage Examples

### Creating Messages

```python
from llamabot.components.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    user,
    dev,
    system
)

# System prompt
sys_msg = system("You are a helpful assistant.")

# User message
user_msg = user("What is Python?")

# AI response
ai_msg = AIMessage(content="Python is a programming language.")
```

### With Images

```python
from llamabot.components.messages import user, ImageMessage
from pathlib import Path

# Using user() helper
msg = user("/path/to/image.jpg")

# Direct ImageMessage
msg = ImageMessage(content=Path("image.jpg"))

# Text and image
msgs = user("What is this?", "/path/to/image.jpg")
```

### With Bots

```python
import llamabot as lmb
from llamabot.components.messages import HumanMessage

bot = lmb.SimpleBot("You are helpful.")

# String (automatically converted)
response = bot("Hello")

# Message object
response = bot(HumanMessage(content="Hello"))

# Multiple messages
response = bot("First message", "Second message")
```

## Message Hierarchy

```text
BaseMessage (base class)
├── SystemMessage (system prompts)
├── HumanMessage (user input)
├── DeveloperMessage (development context)
├── AIMessage (AI responses)
│   ├── ThoughtMessage (agent reasoning)
│   └── ObservationMessage (tool observations)
├── ToolMessage (tool execution results)
├── RetrievedMessage (retrieved documents)
└── ImageMessage (image content)
```

## Related Classes

- **SimpleBot**: Uses messages for LLM communication
- **ChatMemory**: Stores and retrieves messages
- **QueryBot**: Uses RetrievedMessage for document retrieval

## See Also

- [SimpleBot Reference](../reference/bots/simplebot.md)
- [Chat Memory Reference](./chat_memory.md)
- [Which Bot Should I Use?](../getting-started/which-bot.md)
