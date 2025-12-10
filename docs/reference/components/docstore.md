# Document Store API Reference

Document stores provide persistent storage and retrieval for documents in RAG (Retrieval-Augmented Generation) applications.

## AbstractDocumentStore

Abstract base class defining the document store interface.

```python
class AbstractDocumentStore:
    """Abstract document store for LlamaBot."""
```

### Methods

All document stores implement these methods:

#### `append`

```python
def append(self, document: str) -> None
```

Append a single document to the store.

#### Append Parameters

- **document** (`str`): The document text to append

#### `extend`

```python
def extend(self, documents: list[str]) -> None
```

Extend the store with multiple documents (bulk operation).

#### Extend Parameters

- **documents** (`list[str]`): List of document texts to add

#### `retrieve`

```python
def retrieve(self, query: str, n_results: int = 10) -> list[str]
```

Retrieve relevant documents for a query.

#### Retrieve Parameters

- **query** (`str`): The search query
- **n_results** (`int`, default: `10`): Number of results to return

#### Returns

- **list[str]**: List of retrieved document texts

#### `reset`

```python
def reset(self) -> None
```

Reset the document store, clearing all documents.

## LanceDBDocStore

Vector-based document store using LanceDB for semantic search.

```python
class LanceDBDocStore(AbstractDocumentStore):
    """A document store for LlamaBot that wraps around LanceDB.

    Supports optional partitioning to organize documents into logical groups.
    """
```

### Constructor

```python
def __init__(
    self,
    table_name: str,
    storage_path: Path = Path.home() / ".llamabot" / "lancedb",
    embedding_registry: str = "sentence-transformers",
    embedding_model: str = "minishlab/potion-base-8M",
    auto_create_fts_index: bool = True,
    enable_partitioning: bool = False,
    default_partition: str = "default",
)
```

### Parameters

- **table_name** (`str`): Name of the table to create or open. Will be automatically slugified.

- **storage_path** (`Path`, default: `~/.llamabot/lancedb`): Path to the LanceDB storage directory.

- **embedding_registry** (`str`, default: `"sentence-transformers"`): Registry name for the embedding function.

- **embedding_model** (`str`, default: `"minishlab/potion-base-8M"`): Model name for the embedding function.

- **auto_create_fts_index** (`bool`, default: `True`): Whether to automatically create a full-text search index on the document field.

- **enable_partitioning** (`bool`, default: `False`): If `True`, enables partitioning support. When enabled, documents must be assigned to partitions and retrieval can be filtered by partition.

- **default_partition** (`str`, default: `"default"`): Default partition name to use when partition is not specified and partitioning is enabled.

### Methods

#### `append`

```python
def append(
    self,
    document: str,
    partition: Optional[str] = None,
) -> None
```

Append a document to the store.

#### LanceDB Append Parameters

- **document** (`str`): The document text to append
- **partition** (`Optional[str]`, default: `None`): Partition name (required if partitioning is enabled)

#### `extend`

```python
def extend(
    self,
    documents: list[str],
    partitions: Optional[list[str]] = None,
) -> None
```

Extend the store with multiple documents.

#### LanceDB Extend Parameters

- **documents** (`list[str]`): List of document texts
- **partitions** (`Optional[list[str]]`, default: `None`): List of partition names (required if partitioning is enabled)

#### `retrieve`

```python
def retrieve(
    self,
    query: str,
    n_results: int = 10,
    partition: Optional[str] = None,
) -> list[str]
```

Retrieve relevant documents using semantic search.

#### LanceDB Retrieve Parameters

- **query** (`str`): The search query
- **n_results** (`int`, default: `10`): Number of results to return
- **partition** (`Optional[str]`, default: `None`): Filter results by partition (if partitioning is enabled)

#### Returns

- **list[str]**: List of retrieved document texts

#### `reset`

```python
def reset(self) -> None
```

Reset the document store, clearing all documents.

### Usage Examples

#### Basic Usage

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(table_name="my_documents")

# Add documents
docstore.append("Document 1 text")
docstore.extend(["Document 2 text", "Document 3 text"])

# Retrieve documents
results = docstore.retrieve("search query", n_results=5)
```

#### With Custom Embeddings

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(
    table_name="my_documents",
    embedding_registry="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2"
)
```

#### With Partitioning

```python
import llamabot as lmb

docstore = lmb.LanceDBDocStore(
    table_name="my_documents",
    enable_partitioning=True
)

# Add documents to specific partitions
docstore.append("Document 1", partition="category_a")
docstore.extend(
    ["Doc 2", "Doc 3"],
    partitions=["category_b", "category_b"]
)

# Retrieve from specific partition
results = docstore.retrieve("query", partition="category_a")
```

## BM25DocStore

Keyword-based document store using BM25 for full-text search.

```python
class BM25DocStore(AbstractDocumentStore):
    """A document store for LlamaBot that uses BM25 for keyword-based search."""
```

### Constructor

```python
def __init__(self)
```

BM25DocStore has no configuration parameters.

### Methods

Implements all methods from `AbstractDocumentStore`:

- `append(document: str) -> None`
- `extend(documents: list[str]) -> None`
- `retrieve(query: str, n_results: int = 10) -> list[str]`
- `reset() -> None`

### Usage Examples

#### Basic Usage

```python
import llamabot as lmb

docstore = lmb.BM25DocStore()

# Add documents
docstore.append("Document 1 text")
docstore.extend(["Document 2 text", "Document 3 text"])

# Retrieve using keyword search
results = docstore.retrieve("keyword search", n_results=5)
```

## Comparison

| Feature | LanceDBDocStore | BM25DocStore |
|---------|------------------|--------------|
| **Search Type** | Semantic (vector) | Keyword-based |
| **Embeddings** | Required | Not needed |
| **Best For** | Semantic similarity | Exact keyword matching |
| **Speed** | Fast (indexed) | Fast (in-memory) |
| **Persistence** | Yes (disk) | No (in-memory) |
| **Partitioning** | Supported | Not supported |

## Usage with QueryBot

```python
import llamabot as lmb
from pathlib import Path

# Create document store
docstore = lmb.LanceDBDocStore(table_name="my_docs")

# Add documents
docs_paths = Path("docs").rglob("*.md")
docs_texts = [p.read_text() for p in docs_paths]
docstore.extend(docs_texts)

# Use with QueryBot
bot = lmb.QueryBot(
    system_prompt="You are an expert on these documents.",
    docstore=docstore
)

response = bot("What is the main topic?")
```

## Best Practices

1. **Choose the right store**: Use `LanceDBDocStore` for semantic search, `BM25DocStore` for keyword search
2. **Batch operations**: Use `extend()` for adding multiple documents (faster than multiple `append()` calls)
3. **Partitioning**: Use partitioning in `LanceDBDocStore` to organize large document collections
4. **Reset when needed**: Use `reset()` to clear and rebuild indexes when documents change significantly

## Related Classes

- **QueryBot**: Bot that uses document stores for RAG
- **AbstractDocumentStore**: Base interface for document stores

## See Also

- [QueryBot Reference](../bots/querybot.md)
- [QueryBot Tutorial](../../tutorials/querybot.md)
- [Which Bot Should I Use?](../../getting-started/which-bot.md)
