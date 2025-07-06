# Automatically Record QueryBot Calls with PromptRecorder

In this tutorial, we will learn how to use the `PromptRecorder` class to automatically record calls made to the `QueryBot`. The `PromptRecorder` class is designed to record prompts and responses, making it a perfect fit for logging interactions with the `QueryBot`.

## Prerequisites

Before we begin, make sure you have the following Python libraries installed:

- pandas
- panel

You can install them using pip:

```bash
pip install pandas panel
```

## Step 1: Import the necessary classes

First, we need to import the `PromptRecorder` and `QueryBot` classes from their respective source files. You can do this by adding the following lines at the beginning of your script:

```python
from llamabot.recorder import PromptRecorder, autorecord
from llamabot.bot.querybot import QueryBot
```

## Step 2: Initialize the QueryBot

Next, we need to create an instance of the `QueryBot` class. You can do this by providing the necessary parameters, such as the system message, model name, and document paths. For example:

```python
system_message = "You are a helpful assistant that can answer questions based on the provided documents."
model_name = "gpt-4"
doc_paths = ["document1.txt", "document2.txt"]

query_bot = QueryBot(system_message, model_name=model_name, document_paths=doc_paths)
```

## Step 3: Use the PromptRecorder context manager

Now that we have an instance of the `QueryBot`, we can use the `PromptRecorder` context manager to automatically record the prompts and responses. To do this, simply wrap your interactions with the `QueryBot` inside a `with` statement, like this:

```python
with PromptRecorder() as recorder:
    # Interact with the QueryBot here
```

## Step 4: Interact with the QueryBot

Inside the `with` statement, you can now interact with the `QueryBot` by calling it with your queries. For example:

```python
with PromptRecorder() as recorder:
    query = "What is the main idea of document1?"
    response = query_bot(query)
    print(response.content)

    query = "How does document2 support the main idea?"
    response = query_bot(query)
    print(response.content)
```

The `PromptRecorder` will automatically record the prompts and responses for each interaction with the `QueryBot`.

## Step 5: Access the recorded data

After you have finished interacting with the `QueryBot`, you can access the recorded data using the `PromptRecorder` instance. For example, you can print the recorded data as a pandas DataFrame:

```python
print(recorder.dataframe())
```

Or, you can display the recorded data as an interactive panel:

```python
recorder.panel().show()
```

## Complete Example

Here's the complete example that demonstrates how to use the `PromptRecorder` to automatically record `QueryBot` calls:

```python
from llamabot.recorder import PromptRecorder, autorecord
from llamabot.bot.querybot import QueryBot

system_message = "You are a helpful assistant that can answer questions based on the provided documents."
model_name = "gpt-4"
doc_paths = ["document1.txt", "document2.txt"]

query_bot = QueryBot(system_message, model_name=model_name, document_paths=doc_paths)

with PromptRecorder() as recorder:
    query = "What is the main idea of document1?"
    response = query_bot(query)
    print(response.content)

    query = "How does document2 support the main idea?"
    response = query_bot(query)
    print(response.content)

print(recorder.dataframe())
recorder.panel().show()
```

That's it! You now know how to use the `PromptRecorder` class to automatically record calls made to the `QueryBot`. This can be a useful tool for logging and analyzing interactions with your chatbot.
