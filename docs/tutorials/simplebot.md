# SimpleBot Tutorial

!!! note
This tutorial was written by GPT4 and edited by a human.

In this tutorial, we will learn how to use the `SimpleBot` class, a Python implementation of a chatbot that interacts with OpenAI's GPT-4 model. The `SimpleBot` class is designed to be simple and easy to use, allowing you to create a chatbot that can respond to human messages based on a given system prompt.

## Getting Started

First, let's import the `SimpleBot` class:

```python
from llamabot.bot.simplebot import SimpleBot
```

### Initializing the SimpleBot

To create a new instance of `SimpleBot`, you need to provide a system prompt. The system prompt is used to prime the GPT-4 model, giving it context for generating responses. You can also optionally set the `temperature` and `model_name` parameters.

```python
system_prompt = "You are an AI assistant that helps users with their questions."
bot = SimpleBot(system_prompt)
```

### Interacting with the SimpleBot

To interact with the `SimpleBot`, simply call the instance with a human message as a parameter. The bot will return an `AIMessage` object containing the generated response.

```python
human_message = "What is the capital of France?"
response = bot(human_message)
print(response.content)
```

## AIMessage

When interacting with the `SimpleBot`, it's important to note that the response returned is not a simple string, but an `AIMessage` object. This object contains the generated response and additional metadata. The structure of an `AIMessage` is as follows:

```python
from llamabot.components.messages import AIMessage

# Example AIMessage structure
{
"content": "Generated response content",
"role": "assistant"
}
```

## Example

Here's a complete example of how to create and interact with a `SimpleBot`:

```python
from llamabot.bot.simplebot import SimpleBot

# Initialize the SimpleBot
system_prompt = "You are an AI assistant that helps users with their questions."
bot = SimpleBot(system_prompt)

# Interact with the SimpleBot
human_message = "What is the capital of France?"
response = bot(human_message)
print(response.content)
```

## Conclusion

In this tutorial, we learned how to use the `SimpleBot` class to create a simple chatbot that interacts with OpenAI's GPT-4 model. With this knowledge, you can now create your own chatbots and experiment with different system prompts and settings.

## Additional Information

For more detailed information on the `SimpleBot` class and its methods, please refer to the source code and documentation provided in the `llamabot` package.
