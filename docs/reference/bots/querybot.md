# QueryBot API Reference

QueryBot is a bot that can answer questions based on a set of documents. It uses a document store to retrieve relevant documents for a given query.

## Class Definition

```python
class QueryBot(SimpleBot):
    """Initialize QueryBot.

    QueryBot is a bot that can answer questions based on a set of documents.
    It uses a document store to retrieve relevant documents for a given query.
    """
```

## Constructor

```python
def __init__(
    self,
    system_prompt: str,
    docstore: AbstractDocumentStore,
    memory: Optional[AbstractDocumentStore] = None,
    mock_response: str | None = None,
    temperature: float = 0.0,
    model_name: str = default_language_model(),
    stream_target: str = "stdout",
    **kwargs,
)
```

### Parameters

- **system_prompt** (`str`): The system prompt to use for the bot. This defines how the bot interprets and answers questions based on retrieved documents.

- **docstore** (`AbstractDocumentStore`): The document store to use for document retrieval. Must be an instance of `AbstractDocumentStore` (e.g., `LanceDBDocStore`, `BM25DocStore`).

- **memory** (`Optional[AbstractDocumentStore]`, default: `None`): Optional chat memory component. For conversational memory, use `ChatMemory` (e.g., `lmb.ChatMemory()`). This is separate from the document store used for RAG.

- **mock_response** (`str | None`, default: `None`): Optional mock response for testing purposes.

- **temperature** (`float`, default: `0.0`): The model temperature to use. Controls randomness in responses.

- **model_name** (`str`, default: `default_language_model()`): The name of the model to use. Supports all models from LiteLLM.

- **stream_target** (`str`, default: `"stdout"`): The target to stream the response to. Should be one of `"stdout"`, `"panel"`, `"api"`, or `"none"`.

- **kwargs**: Additional keyword arguments passed to `SimpleBot`.

## Methods

### `__call__`

```python
def __call__(
    self,
    query: Union[str, HumanMessage, BaseMessage],
    n_results: int = 20,
) -> AIMessage
```

Query documents within QueryBot's document store and return an answer.

#### Parameters

- **query** (`Union[str, HumanMessage, BaseMessage]`): The query to search for. Can be a string or a message object.

- **n_results** (`int`, default: `20`): The number of document results to retrieve and use for answering the query.

#### Returns

- **AIMessage**: The AI's response message containing:
  - `content`: The answer based on retrieved documents
  - `role`: `"assistant"`
  - Additional metadata

#### Example

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(table_name="my_docs")
docstore.extend([doc1, doc2, doc3])

bot = lmb.QueryBot(
    system_prompt="You are an expert on these documents.",
    docstore=docstore
)

response = bot("What does the documentation say about authentication?")
print(response.content)
```

## Attributes

- **docstore** (`AbstractDocumentStore`): The document store used for retrieval
- **memory** (`Optional[AbstractDocumentStore]`): The chat memory component (if any)

## Usage Examples

### Basic Document Q&A

```python
import llamabot as lmb
from pathlib import Path

# Create a document store
docstore = lmb.LanceDBDocStore(table_name="my_documents")

# Add documents
docs_paths = Path("docs").rglob("*.md")
docs_texts = [p.read_text() for p in docs_paths]
docstore.extend(docs_texts)

# Create QueryBot
bot = lmb.QueryBot(
    system_prompt="You are an expert on these documents.",
    docstore=docstore
)

# Query the documents
response = bot("What is the main topic of these documents?")
```

### With Chat Memory

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(table_name="my_docs")
docstore.extend(documents)

# Create chat memory for conversation context
memory = lmb.ChatMemory()

bot = lmb.QueryBot(
    system_prompt="You are an expert on these documents.",
    docstore=docstore,
    memory=memory
)

# Bot remembers previous questions
response1 = bot("What is authentication?")
response2 = bot("How does it work?")  # Bot remembers context
```

### With Threaded Memory

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(table_name="my_docs")
docstore.extend(documents)

# Use intelligent threading for better context
memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")

bot = lmb.QueryBot(
    system_prompt="You are an expert on these documents.",
    docstore=docstore,
    memory=memory
)
```

### Custom Number of Results

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(table_name="my_docs")
docstore.extend(documents)

bot = lmb.QueryBot(
    system_prompt="You are an expert on these documents.",
    docstore=docstore
)

# Retrieve more documents for comprehensive answers
response = bot("Explain the architecture", n_results=50)
```

## Document Store Options

### LanceDBDocStore (Default)

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(
    table_name="my-documents",
    embedding_registry="sentence-transformers",
    embedding_model="minishlab/potion-base-8M"
)
```

### BM25DocStore (Keyword-based)

```python
import llamabot as lmb

docstore = lmb.BM25DocStore()
docstore.extend(documents)
```

## Differences from SimpleBot

- **QueryBot**: Retrieves relevant documents before answering
- **SimpleBot**: Answers based on training data only (no document retrieval)

## Best Practices

1. **Use descriptive system prompts**: Help the bot understand how to use the documents
2. **Adjust n_results**: More results = more context, but slower and more expensive
3. **Use chat memory**: Maintain conversation context across queries
4. **Pre-process documents**: Clean and structure documents before adding to the store

## Related Classes

- **SimpleBot**: Base class that QueryBot extends
- **AbstractDocumentStore**: Document store interface
- **LanceDBDocStore**: Default vector-based document store
- **BM25DocStore**: Keyword-based document store
- **ChatMemory**: Conversation memory component

## See Also

- [QueryBot Tutorial](../tutorials/querybot.md)
- [Which Bot Should I Use?](../getting-started/which-bot.md)
- [DocStore Component](../components/docstore.md)
- [Chat Memory Component](../components/chat_memory.md)
