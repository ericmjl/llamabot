# QueryBot Tutorial

In this tutorial, we will learn how to use the `QueryBot` class to create a chatbot that can query documents using an LLM. The `QueryBot` class allows us to index documents and use an LLM to generate responses based on the indexed documents.

## Using QueryBot with a Document Store and Chat Memory

The recommended way to use QueryBot is to explicitly create and manage your own document store using `LanceDBDocStore` and chat memory using `ChatMemory`. This gives you full control over storage, persistence, and memory management. Below is the standard usage pattern (inspired by `notebooks/llamabot_docs.py`).

```python
from llamabot.components.docstore import LanceDBDocStore
from llamabot import QueryBot
from pyprojroot import here

# Create a document store for your knowledge base
docstore = LanceDBDocStore(
    table_name="my-documents",
    # Optional: Configure embedding model settings
    embedding_registry="sentence-transformers",  # Default registry
    embedding_model="minishlab/potion-base-8M",  # Default model
)

docstore.reset()  # Optionally clear all documents

# Add documents (e.g., all Markdown files in the docs folder)
docs_paths = (here() / "docs").rglob("*.md")
docs_texts = [p.read_text() for p in docs_paths]
docstore.extend(docs_texts)

# Create a separate store for chat memory
# For simple linear memory (fast, no LLM calls)
chat_memory = lmb.ChatMemory()

# For intelligent threading (uses LLM for smart connections)
# chat_memory = lmb.ChatMemory.threaded(model="gpt-4o-mini")

chat_memory.reset()  # Optionally clear previous chat history

# Define a system prompt (optionally using the @prompt decorator)
system_prompt = "You are a helpful assistant for my project."

# Initialize QueryBot with both docstore and chat memory
bot = QueryBot(
    system_prompt=system_prompt,
    docstore=docstore,
    memory=chat_memory,
)

# Use the bot in a conversational loop
while True:
    user_input = input("Ask a question: ")
    if user_input.lower() in {"exit", "quit"}:
        break
    response = bot(user_input)
    print(response.content)
```

## Organizing Documents with Partitions

When working with large document collections, you may want to organize documents into logical groups called **partitions**. Partitions are useful for:

- Organizing documents by source (e.g., "tutorials", "reference", "api_docs")
- Categorizing by topic or domain
- Separating documents by date or version
- Any other logical grouping that helps you filter and search more effectively

### Enabling Partitioning

To use partitioning, enable it when creating your document store:

```python
from llamabot.components.docstore import LanceDBDocStore

# Create a document store with partitioning enabled
docstore = LanceDBDocStore(
    table_name="my-documents",
    enable_partitioning=True,
    default_partition="general",  # Optional: default partition name
)
```

### Adding Documents to Partitions

You can assign documents to partitions in several ways:

**Using `append()` with a partition:**

```python
# Add a single document to a specific partition
docstore.append("Python tutorial content", partition="tutorials")
docstore.append("API reference documentation", partition="reference")
```

**Using `extend()` with a single partition (all documents go to the same partition):**

```python
# Add multiple documents to the same partition
tutorial_docs = [
    "Python tutorial part 1",
    "Python tutorial part 2",
    "Python tutorial part 3",
]
docstore.extend(tutorial_docs, partition="tutorials")
```

**Using `extend()` with multiple partitions (one partition per document):**

```python
# Add documents to different partitions in one call
documents = [
    "Python tutorial",
    "Python reference",
    "Python API docs",
]
partitions = ["tutorials", "reference", "api_docs"]
docstore.extend(documents, partitions=partitions)
```

### Querying Specific Partitions

You can query documents from specific partitions using the `partitions` parameter:

```python
# Query only the tutorials partition
results = docstore.retrieve("python programming", partitions=["tutorials"])

# Query multiple partitions
results = docstore.retrieve(
    "python", partitions=["tutorials", "reference"]
)

# Query all partitions (default behavior)
results = docstore.retrieve("python")  # Searches all partitions
```

**Note:** Currently, `QueryBot` doesn't support partition filtering directly. If you need to query specific partitions with QueryBot, you have a few options:

1. **Access the docstore directly** before creating the bot:
   ```python
   # Get partition-filtered results
   relevant_docs = docstore.retrieve("python", partitions=["tutorials"])

   # Then use these results with your bot
   # (You'd need to manually construct the context)
   ```

2. **Create separate QueryBot instances** for different partitions:
   ```python
   # Create separate docstores or filter results per partition
   tutorial_bot = QueryBot(
       system_prompt="You are a Python tutorial assistant.",
       docstore=docstore,  # Same docstore, but filter in your queries
   )
   ```

3. **Use the docstore's `retrieve()` method** and manually pass results to your LLM.

### Helper Methods for Partition Management

The `LanceDBDocStore` provides several helper methods for working with partitions:

**List all available partitions:**

```python
partitions = docstore.list_partitions()
print(partitions)  # ['tutorials', 'reference', 'api_docs']
```

**Get the count of documents in a partition:**

```python
count = docstore.get_partition_count("tutorials")
print(f"Tutorials partition has {count} documents")
```

**Reset (clear) a specific partition:**

```python
# Delete all documents in the tutorials partition
docstore.reset_partition("tutorials")
```

### Complete Example with Partitions

Here's a complete example showing partitioning in action:

```python
from llamabot.components.docstore import LanceDBDocStore
from llamabot import QueryBot
from pathlib import Path

# Create partitioned document store
docstore = LanceDBDocStore(
    table_name="project-docs",
    enable_partitioning=True,
    default_partition="general",
)

docstore.reset()

# Organize documents by category
tutorial_files = list(Path("docs/tutorials").glob("*.md"))
reference_files = list(Path("docs/reference").glob("*.md"))

# Add tutorials
for file in tutorial_files:
    docstore.append(file.read_text(), partition="tutorials")

# Add reference docs
for file in reference_files:
    docstore.append(file.read_text(), partition="reference")

# Query specific partition directly
tutorial_results = docstore.retrieve(
    "how do I get started?",
    partitions=["tutorials"]  # Only search tutorials partition
)
print(f"Found {len(tutorial_results)} results in tutorials partition")

# Query multiple partitions
tutorial_and_ref_results = docstore.retrieve(
    "python syntax",
    partitions=["tutorials", "reference"]  # Search both partitions
)

# See what partitions exist
print(f"Available partitions: {docstore.list_partitions()}")
print(f"Tutorials count: {docstore.get_partition_count('tutorials')}")

# Create QueryBot (searches all partitions by default)
bot = QueryBot(
    system_prompt="You are a helpful documentation assistant.",
    docstore=docstore,
)

# QueryBot will search across all partitions
response = bot("How do I use the API?")

# To query only a specific partition, use docstore.retrieve() directly
# and then manually construct your prompt with those results
tutorial_only_results = docstore.retrieve(
    "getting started",
    partitions=["tutorials"]  # Only tutorials partition
)
# You can then use these results with your LLM of choice
```

**Tips:**

- You can use `.reset()` on either store to clear its contents.
- The `LanceDBDocStore` uses the following default settings:
  - `embedding_registry`: "sentence-transformers"
  - `embedding_model`: "minishlab/potion-base-8M"
  You can customize these settings when initializing the store to use different embedding models.
- For more details, see the source code in [`llamabot/bot/querybot.py`](../../llamabot/bot/querybot.py) and [`llamabot/components/docstore.py`](../../llamabot/components/docstore.py).
- This pattern is ideal for interactive apps, notebooks, or production bots where you want persistent memory and document storage.
