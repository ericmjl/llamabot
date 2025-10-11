# Create a Chat Interface with SimpleBot and Panel

- [Create a Chat Interface with SimpleBot and Panel](#create-a-chat-interface-with-simplebot-and-panel)
  - [Prerequisites](#prerequisites)
  - [Code Breakdown](#code-breakdown)
    - [Import Necessary Libraries](#import-necessary-libraries)
    - [Initialize Panel Extension](#initialize-panel-extension)
    - [Create a SimpleBot Instance](#create-a-simplebot-instance)
    - [Define the Callback Function](#define-the-callback-function)
    - [Create the Chat Interface](#create-the-chat-interface)
    - [Send an Initial Message](#send-an-initial-message)
    - [Make the Chat Interface Servable](#make-the-chat-interface-servable)
    - [Serve up the Panel app](#serve-up-the-panel-app)
  - [All the code together](#all-the-code-together)
    - [The python script](#the-python-script)
    - [The terminal command](#the-terminal-command)
  - [Conclusion](#conclusion)

In this tutorial, we will explore how to set up a simple chat interface
using the `SimpleBot` class from the `llamabot` library and the `Panel` library.
By the end of this tutorial,
you'll be able to integrate a bot into a chat interface and see how it interacts.

## Prerequisites

- Familiarity with Python programming.
- The `llamabot` and `panel` libraries installed.

## Code Breakdown

### Import Necessary Libraries

```python
from llamabot import SimpleBot
import panel as pn
```

- `SimpleBot`: Class from the `llamabot` library that allows you to create chatbot instances.
- `panel` (aliased as `pn`): A Python library for creating web-based interactive apps and dashboards.

### Initialize Panel Extension

```python
pn.extension()
```

Before using Panel's functionality, you need to initialize its extension with `pn.extension()`. This prepares your Python environment to work with Panel components.

### Create a SimpleBot Instance

```python
bot = SimpleBot("You are Richard Feynman.")
```

Here, we're creating an instance of the `SimpleBot` class. The string argument, "You are Richard Feynman.", serves as a context or persona for the bot. Essentially, this bot will behave as if it's Richard Feynman.

### Define the Callback Function

```python
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    response = bot(contents)
    yield response.content
```

- This asynchronous function will be called whenever a user sends a message to the chat interface.
- It accepts three parameters:
  - `contents`: The message sent by the user.
  - `user`: The name of the user sending the message.
  - `instance`: The chat interface instance.
- Inside the function, the message `contents` is passed to the bot, and the bot's response is yielded.

### Create the Chat Interface

```python
chat_interface = pn.chat.ChatInterface(
    callback=callback, callback_user="Feynman Bot", show_clear=False
)
```

- We're creating an instance of `ChatInterface` from Panel's chat module.
- The `callback` parameter is set to the previously defined `callback` function. This tells the chat interface to use our function to handle messages.
- `callback_user` is the name that will be displayed for the bot's messages.
- `show_clear=False` means the chat interface won't have a clear button to erase the chat history.

### Send an Initial Message

```python
chat_interface.send(
    "Send a message to get a reply from the bot!",
    user="System",
    respond=False,
)
```

- This sends an initial message to the chat interface to prompt users to interact with the bot.
- The message is sent from the "System" user and does not expect a reply (`respond=False`).

### Make the Chat Interface Servable

```python
chat_interface.servable()
```

By calling `.servable()` on the chat interface,
you're telling Panel to treat this interface as the main component when you run the app.

### Serve up the Panel app

Now, you can serve up the app typing

```bash
panel serve chat_interface.py
```

in your terminal.
This will open up a new browser window with the chat interface.

## All the code together

### The python script

```python
# chat_interface.py
from llamabot import SimpleBot
import panel as pn

pn.extension()

bot = SimpleBot("You are Richard Feynman.")


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    response = bot(contents)
    yield response.content


chat_interface = pn.chat.ChatInterface(
    callback=callback, callback_user="Feynman Bot", show_clear=False
)
chat_interface.send(
    "Send a message to get a reply from the bot!",
    user="System",
    respond=False,
)
chat_interface.servable()
```

### The terminal command

```bash
panel serve chat_interface.py
```

## Conclusion

With just a few lines of code, you can create a chat interface
and integrate it with a bot using the `llamabot` and `Panel` libraries.
This setup provides a foundational step towards creating
more interactive and dynamic chatbot applications.
Happy coding!
