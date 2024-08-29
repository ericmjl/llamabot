---
intents:
- How do we use the llamabot ChatBot class in a Jupyter notebook?
- How to serve up a Panel app based on that ChatBot class.
- Specific details on how the ChatBot retrieval works when composing an API call,
  such as which messages are retrieved from history.
linked_files:
- llamabot/bot/chatbot.py
- llamabot/__init__.py
---

# Using the llamabot ChatBot Class in a Jupyter Notebook

To use the `ChatBot` class from llamabot in a Jupyter notebook, you can follow these steps:

1. Import the `ChatBot` class from the `llamabot.bot.chatbot` module:

```python
from llamabot.bot.chatbot import ChatBot
```

2. Create an instance of the `ChatBot` class by providing the required parameters such as the system prompt, session name, and any additional configuration options:

```python
system_prompt = "Your system prompt here"
session_name = "Your session name here"
chatbot = ChatBot(system_prompt, session_name)
```

3. Interact with the `ChatBot` instance by calling it with a human message:

```python
human_message = "Hello, how are you?"
response = chatbot(human_message)
print(response)
```

# Serving a Panel App Based on the ChatBot Class

To serve a Panel app based on the `ChatBot` class, you can use the `stream_panel` method of the `ChatBot` class. Here's an example of how to do this:

```python
panel_app = chatbot.stream_panel(messages)
panel_app.servable()
```

# ChatBot Retrieval and API Composition

When composing an API call using the `ChatBot` class, the retrieval of messages from history is handled internally. The `retrieve` method of the `ChatBot` class is used to retrieve messages from the chat history based on the provided human message and response budget. The retrieved messages include the system prompt, historical messages, and the human message itself.

For example, when making an API call to the `ChatBot` instance, the retrieval process ensures that the historical context is considered when generating the response.

This covers the specific details on how the `ChatBot` retrieval works when composing an API call.

Please let me know if you need further details or examples.
