# SimpleBot API Reference

SimpleBot is a stateless or stateful chatbot that can be primed with a system prompt and used for general-purpose conversations.

## Class Definition

```python
class SimpleBot:
    """Simple Bot that is primed with a system prompt, accepts a human message,
    and sends back a single response.

    This bot does not retain chat history by default, but can be configured
    to do so by passing in a chat memory component.
    """
```

## Constructor

```python
def __init__(
    self,
    system_prompt: str,
    temperature: float = 0.0,
    memory: Optional[AbstractDocumentStore] = None,
    model_name: str = default_language_model(),
    stream_target: str = "stdout",
    json_mode: bool = False,
    api_key: Optional[str] = None,
    mock_response: Optional[str] = None,
    **completion_kwargs,
)
```

### Parameters

- **system_prompt** (`str`): The system prompt to use for the bot. This defines the bot's behavior and personality.

- **temperature** (`float`, default: `0.0`): The model temperature to use. Controls randomness in responses. Range: 0.0 (deterministic) to 2.0 (more creative). See [OpenAI's temperature documentation](https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature) for more information.

- **memory** (`Optional[AbstractDocumentStore]`, default: `None`): An optional chat memory component to use. If provided, the bot will retain chat history. For conversational memory, use `ChatMemory` (e.g., `lmb.ChatMemory()`). For RAG/document retrieval, use an `AbstractDocumentStore` (such as `BM25DocStore` or `LanceDBDocStore`).

- **model_name** (`str`, default: `default_language_model()`): The name of the model to use. Supports all models from LiteLLM. Examples: `"gpt-4o-mini"`, `"ollama_chat/llama2:13b"`, `"anthropic/claude-3-5-sonnet"`.

- **stream_target** (`str`, default: `"stdout"`): The target to stream the response to. Should be one of `"stdout"`, `"panel"`, `"api"`, or `"none"`.

- **json_mode** (`bool`, default: `False`): Whether to enable JSON mode for structured output. Note: This does not guarantee schema validation (use `StructuredBot` for that).

- **api_key** (`Optional[str]`, default: `None`): The API key to use. If not provided, will use environment variables (e.g., `OPENAI_API_KEY`).

- **mock_response** (`Optional[str]`, default: `None`): A mock response to use, for testing purposes only.

- **completion_kwargs**: Additional keyword arguments to pass to the completion function of `litellm`. See [LiteLLM documentation](https://docs.litellm.ai/) for available options.

### Special Cases

- For `o1-preview` and `o1-mini` models, the system prompt is automatically converted to a human message (as required by these models).

## Methods

### `__call__`

```python
def __call__(
    self,
    *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]], Callable],
) -> AIMessage
```

Process messages and return an AI response.

#### Parameters

- **messages**: One or more messages to process. Can be:
  - Strings (converted to `HumanMessage`)
  - `BaseMessage` objects (`HumanMessage`, `AIMessage`, `SystemMessage`)
  - Lists of messages
  - Callable functions that return strings

#### Returns

- **AIMessage**: The AI's response message containing:
  - `content`: The response text
  - `role`: `"assistant"`
  - Additional metadata

#### Example

```python
import llamabot as lmb

bot = lmb.SimpleBot("You are a helpful assistant.")

# Single string message
response = bot("Hello!")
print(response.content)

# Multiple messages
response = bot("What is Python?", "Tell me more about it.")

# Using message objects
from llamabot.components.messages import HumanMessage
response = bot(HumanMessage(content="Hello!"))
```

### `generate_response`

```python
def generate_response(
    self,
    messages: List[BaseMessage],
    stream: bool = True,
) -> AIMessage
```

Generate a response from a list of messages. This is the internal method used by `__call__`.

#### Parameters

- **messages** (`List[BaseMessage]`): List of message objects to process
- **stream** (`bool`, default: `True`): Whether to stream the response

#### Returns

- **AIMessage**: The AI's response message

## Attributes

- **system_prompt** (`SystemMessage`): The system prompt message
- **model_name** (`str`): The model name being used
- **temperature** (`float`): The temperature setting
- **memory** (`Optional[AbstractDocumentStore]`): The memory component (if any)
- **stream_target** (`str`): The streaming target
- **json_mode** (`bool`): Whether JSON mode is enabled

## Usage Examples

### Basic Usage

```python
import llamabot as lmb

bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini"
)

response = bot("What is machine learning?")
print(response.content)
```

### With Chat Memory

```python
import llamabot as lmb

# Linear memory (fast, no LLM calls)
memory = lmb.ChatMemory()

# Or threaded memory (intelligent conversation threading)
memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")

bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    memory=memory,
    model_name="gpt-4o-mini"
)

# Bot remembers previous conversations
response1 = bot("My name is Alice.")
response2 = bot("What's my name?")  # Bot remembers!
```

### With Custom Model

```python
import llamabot as lmb

# Using Ollama local model
bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="ollama_chat/llama2:13b"
)

# Using Anthropic Claude
bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    model_name="anthropic/claude-3-5-sonnet"
)
```

### Streaming to Panel

```python
import llamabot as lmb

bot = lmb.SimpleBot(
    system_prompt="You are a helpful assistant.",
    stream_target="panel",
    model_name="gpt-4o-mini"
)

response = bot("Tell me a long story.")
```

## Related Classes

- **ToolBot**: Extends SimpleBot with tool execution capabilities
- **QueryBot**: Extends SimpleBot with document retrieval
- **StructuredBot**: Extends SimpleBot with Pydantic validation
- **AgentBot**: Uses SimpleBot internally for decision-making

## See Also

- [SimpleBot Tutorial](../../tutorials/simplebot.md)
- [Which Bot Should I Use?](../../getting-started/which-bot.md)
- [Chat Memory Component](../components/chat_memory.md)
