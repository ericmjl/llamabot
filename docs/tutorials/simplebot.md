# SimpleBot Tutorial

> NOTE: This tutorial was written by GPT4 and edited by a human.

In this tutorial, we will explore the `SimpleBot` class,
which is a basic chatbot that can be primed with a system prompt,
accept a human message, and send back a single response.
The bot does not retain chat history.
We will also learn how to create a Panel app that wraps the `SimpleBot`.

## SimpleBot Class

The `SimpleBot` class is initialized with a system prompt,
a model temperature, and a model name.
The system prompt is used to prime the bot,
while the model temperature and model name are used to configure
the underlying OpenAI model.

### Initialization

To initialize a `SimpleBot` instance, you can use the following code:

```python
system_prompt = "You are an AI assistant that helps users with their questions."
bot = SimpleBot(system_prompt, temperature=0.5, model_name="gpt-4")
```

### Calling the SimpleBot

To call the `SimpleBot` with a human message, you can use the following code:

```python
human_message = "What is the capital of France?"
response = bot(human_message)
print(response.content)
```

This will return the AI's response to the human message, primed by the system prompt.

### Creating a Panel App

The `SimpleBot` class also provides a `panel` method
that creates a Panel app to interact with the bot.
The method accepts several optional parameters
to customize the appearance and behavior of the app:

- `input_text_label`: The label for the input text area.
- `output_text_label`: The label for the output text area.
- `submit_button_label`: The label for the submit button.
- `site_name`: The name of the site.
- `title`: The title of the site.
- `show`: Whether to show the app. If `False`, the method returns the Panel app directly. If `True`, the method calls `.show()` on the app.

To create a Panel app for the `SimpleBot`, you can use the following code:

```python
app = bot.panel(
    input_text_label="Input",
    output_text_label="Output",
    submit_button_label="Submit",
    site_name="SimpleBot",
    title="SimpleBot",
    show=False,
)
```

To show the app, you can either set the `show` parameter to `True`
or call the `.show()` method on the returned app:

```python
app.show()
```

This will launch a web server and display the Panel app in your web browser,
allowing you to interact with the `SimpleBot`.

In summary, the `SimpleBot` class provides a simple way
to create a chatbot that can be primed with a system prompt and accept human messages.
The class also allows you to create a Panel app
to interact with the bot in a user-friendly manner.
