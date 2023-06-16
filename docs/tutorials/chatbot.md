# ChatBot Tutorial

> NOTE: This tutorial was written by GPT4 and edited by a human.

In this tutorial, we will learn how to use the `ChatBot` class to create a simple chatbot that can interact with users. The chatbot is built using the OpenAI GPT-4 model and can be used in a Panel app.

## Getting Started

First, let's import the `ChatBot` class:

```python
from llamabot import ChatBot
```

Now, let's create a new instance of the `ChatBot` class. We need to provide a system prompt, which will be used to prime the chatbot. Optionally, we can also set the temperature and model name:

```python
system_prompt = "Hello, I am a chatbot. How can I help you today?"
chatbot = ChatBot(system_prompt, temperature=0.0, model_name="gpt-4")
```

## Interacting with the ChatBot

To interact with the chatbot, we can simply call the chatbot instance with a human message:

```python
human_message = "What is the capital of France?"
response = chatbot(human_message)
print(response.content)
```

The chatbot will return an `AIMessage` object containing the response to the human message, primed by the system prompt.

## Chat History

The chatbot automatically manages the chat history. To view the chat history, we can use the `__repr__` method:

```python
print(chatbot)
```

This will return a string representation of the chat history, with each message prefixed by its type (System, Human, or AI).

## Creating a Panel App

The `ChatBot` class also provides a `panel` method to create a Panel app that wraps the chatbot. This allows users to interact with the chatbot through a web interface.

To create a Panel app, simply call the `panel` method on the chatbot instance:

```python
app = chatbot.panel(show=False)
```

By default, the app will be shown in a new browser window. If you want to return the app directly, set the `show` parameter to `False`.

You can customize the appearance of the app by providing additional parameters, such as `site`, `title`, and `width`:

```python
app = chatbot.panel(show=False, site="My ChatBot", title="My ChatBot", width=768)
```

To run the app, you can either call the `show` method on the app or use the Panel `serve` function:

```python
app.show()
```

or

```python
import panel as pn
pn.serve(app)
```

Now you have a fully functional chatbot that can interact with users through a web interface!

## Conclusion

In this tutorial, we learned how to use the `ChatBot` class to create a simple chatbot that can interact with users. We also learned how to create a Panel app to provide a web interface for the chatbot. With this knowledge, you can now create your own chatbots and customize them to suit your needs. Happy chatting!
