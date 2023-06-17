# QueryBot Tutorial

!!! note
    This tutorial was written by GPT4 and edited by a human.

In this tutorial, we will learn how to use the `QueryBot` class to create a chatbot that can query documents using GPT-4. The `QueryBot` class allows us to index documents and use GPT-4 to generate responses based on the indexed documents.

## Initializing QueryBot

To create a new instance of `QueryBot`, we need to provide a system message, a list of document paths, or a saved index path. The system message is used to instruct the chatbot on how to behave. The document paths are used to index the documents, and the saved index path is used to load a pre-built index.

Here's an example of how to initialize a `QueryBot`:

```python
from pathlib import Path
from llamabot import QueryBot

system_message = "You are a helpful assistant that can answer questions based on the provided documents."
doc_paths = [Path("document1.txt"), Path("document2.txt")]

query_bot = QueryBot(system_message=system_message, doc_paths=doc_paths)
```

## Querying the Index

To query the index, we can call the `QueryBot` instance with a query string. The `QueryBot` will return the top `similarity_top_k` documents from the index and use them to generate a response using GPT-4.

Here's an example of how to query the index:

```python
query = "What is the main idea of document1?"
response = query_bot(query)
print(response.content)
```

## Saving and Loading the Index

We can save the index to disk using the `save` method and load it later using the `__init__` method with the `saved_index_path` parameter.

Here's an example of how to save and load the index:

```python
# Save the index
query_bot.save("index.json")

# Load the index
loaded_query_bot = QueryBot(system_message=system_message, saved_index_path="index.json")
```

## Inserting Documents into the Index

We can insert new documents into the index using the `insert` method. This method takes a file path as an argument and inserts the document into the index.

Here's an example of how to insert a document into the index:

```python
query_bot.insert(Path("new_document.txt"))
```

## Conclusion

In this tutorial, we learned how to use the `QueryBot` class to create a chatbot that can query documents using GPT-4. We covered how to initialize a `QueryBot`, query the index, save and load the index, and insert new documents into the index. With this knowledge, you can now create your own chatbot that can answer questions based on a set of documents.
