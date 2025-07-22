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

**Tips:**

- You can use `.reset()` on either store to clear its contents.
- The `LanceDBDocStore` uses the following default settings:
  - `embedding_registry`: "sentence-transformers"
  - `embedding_model`: "minishlab/potion-base-8M"
  You can customize these settings when initializing the store to use different embedding models.
- For more details, see the source code in [`llamabot/bot/querybot.py`](../../llamabot/bot/querybot.py) and [`llamabot/components/docstore.py`](../../llamabot/components/docstore.py).
- This pattern is ideal for interactive apps, notebooks, or production bots where you want persistent memory and document storage.
