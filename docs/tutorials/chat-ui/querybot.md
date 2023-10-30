# Building a QueryBot Chat Interface with File Upload in Panel

- [Building a QueryBot Chat Interface with File Upload in Panel](#building-a-querybot-chat-interface-with-file-upload-in-panel)
  - [Prerequisites](#prerequisites)
  - [Code Breakdown](#code-breakdown)
    - [Import Necessary Libraries](#import-necessary-libraries)
    - [Initialize Panel Extension](#initialize-panel-extension)
    - [Set Up Widgets and Global Variables](#set-up-widgets-and-global-variables)
    - [Define the File Upload Function](#define-the-file-upload-function)
    - [Interact with the Bot and Update Chat Interface](#interact-with-the-bot-and-update-chat-interface)
    - [Monitor File Uploads](#monitor-file-uploads)
    - [Define the Callback Function for Chat Interface](#define-the-callback-function-for-chat-interface)
    - [Set Up the Chat Interface](#set-up-the-chat-interface)
    - [Combine Widgets and Chat Interface into a Single App](#combine-widgets-and-chat-interface-into-a-single-app)
  - [All the code together](#all-the-code-together)
    - [The script](#the-script)
    - [The terminal command](#the-terminal-command)
  - [Conclusion](#conclusion)

In this tutorial, we will walk through how to create a chat interface
that allows users to upload a PDF file, which the `QueryBot`
from the `llamabot` library will then summarize.
This is all presented in a neat web app using the `Panel` library.

## Prerequisites

- Familiarity with Python programming.
- The `llamabot` and `panel` libraries installed.

## Code Breakdown

### Import Necessary Libraries

```python
from llamabot import QueryBot
import tempfile
import panel as pn
from pathlib import Path
```

- `QueryBot`: A class from the `llamabot` library designed to query and extract information from a given document.
- `tempfile`: A module to generate temporary files and directories.
- `panel` (aliased as `pn`): A Python library for creating web-based interactive apps and dashboards.
- `Path`: A class from the `pathlib` library for manipulating filesystem paths.

### Initialize Panel Extension

```python
pn.extension()
```

This initializes Panel's extension, preparing the environment to work with Panel components.

### Set Up Widgets and Global Variables

```python
file_input = pn.widgets.FileInput(mime_type=["application/pdf"])
spinner = pn.indicators.LoadingSpinner(value=False, width=30, height=30)
global bot
bot = None
```

- `file_input`: A widget that allows users to upload PDF files.
- `spinner`: A loading spinner indicator to show when the bot is processing.
- `bot`: A global variable to store the `QueryBot` instance.

### Define the File Upload Function

```python
def upload_file(event):
    spinner.value = True
    raw_contents = event.new

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", mode="wb"
    ) as temp_file:
        temp_file.write(raw_contents)
        global bot
        bot = QueryBot("You are Richard Feynman", doc_paths=[Path(temp_file.name)])
    ...
```

This function is triggered when a file is uploaded:

- It sets the spinner to active.
- Retrieves the raw contents of the uploaded file.
- Creates a temporary PDF file and writes the uploaded content to it.
- Initializes the `QueryBot` instance with the context "You are Richard Feynman" and the path to the temporary file.

### Interact with the Bot and Update Chat Interface

```python
    chat_interface.send(
        "Please allow me to summarize the paper for you. One moment...",
        user="System",
        respond=False,
    )
    response = bot("Please summarize this paper for me.")
    chat_interface.send(response.content, user="System", respond=False)
    spinner.value = False
```

After initializing the bot:

- A system message is sent to inform the user that the bot is working on summarizing.
- The bot is then asked to summarize the uploaded paper.
- The bot's response is sent to the chat interface.
- The spinner is deactivated.

### Monitor File Uploads

```python
file_input.param.watch(upload_file, "value")
```

This line sets up an event listener on the `file_input` widget. When a file is uploaded (i.e., its value changes), the `upload_file` function is triggered.

### Define the Callback Function for Chat Interface

```python
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    spinner.value = True
    global bot
    response = bot(contents)
    spinner.value = False
    yield response.content
```

This function is called whenever a user sends a message:

- Activates the spinner.
- Queries the `bot` with the user's message.
- Deactivates the spinner.
- Yields the bot's response.

### Set Up the Chat Interface

```python
chat_interface = pn.chat.ChatInterface(
    callback=callback,
    callback_user="QueryBot",
    show_clear=False,
)
chat_interface.send(
    "Send a message to get a reply from the bot!",
    user="System",
    respond=False,
)
```

This sets up the chat interface and sends an initial message prompting the user to interact.

### Combine Widgets and Chat Interface into a Single App

```python
app = pn.Column(pn.Row(file_input, spinner), chat_interface)
app.servable()
```

The file upload widget, spinner, and chat interface are arranged in a layout. The `app` is made servable, marking it as the main component when the app runs.

## All the code together

### The script

```python
# chat_interface.py
from llamabot import QueryBot
import tempfile
import panel as pn
from pathlib import Path

pn.extension()

file_input = pn.widgets.FileInput(mime_type=["application/pdf"])
spinner = pn.indicators.LoadingSpinner(value=False, width=30, height=30)
global bot
bot = None


def upload_file(event):
    spinner.value = True
    raw_contents = event.new

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf", mode="wb"
    ) as temp_file:
        temp_file.write(raw_contents)
        global bot
        bot = QueryBot("You are Richard Feynman", doc_paths=[Path(temp_file.name)])

    chat_interface.send(
        "Please allow me to summarize the paper for you. One moment...",
        user="System",
        respond=False,
    )
    response = bot("Please summarize this paper for me.")
    chat_interface.send(response.content, user="System", respond=False)
    spinner.value = False


file_input.param.watch(upload_file, "value")


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    spinner.value = True
    global bot
    response = bot(contents)
    spinner.value = False
    yield response.content


chat_interface = pn.chat.ChatInterface(
    callback=callback,
    callback_user="QueryBot",
    show_clear=False,
)
chat_interface.send(
    "Send a message to get a reply from the bot!",
    user="System",
    respond=False,
)

app = pn.Column(pn.Row(file_input, spinner), chat_interface)
app.show()
```

### The terminal command

```bash
panel serve chat_interface.py
```

## Conclusion

By integrating the `QueryBot` and `Panel` libraries,
we've built a dynamic chat interface that can summarize uploaded PDF files.
This tutorial serves as a foundation to develop more sophisticated chatbot applications
with file processing capabilities.
Happy coding!
