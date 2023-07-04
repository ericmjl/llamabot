# LlamaBot Tutorial

Welcome to the LlamaBot tutorial. LlamaBot is a command-line interface (CLI) tool that allows you to interact with various applications, Python scripts, Git repositories, tutorials, Zotero references, and documentation.
This tutorial will guide you through the process of setting up and using LlamaBot.

## Installation

Before we start, ensure that you have LlamaBot installed. If not, you can install it using pip:

```bash
pip install -U llamabot
```

## Configuring LlamaBot

Before you can use LlamaBot, you need to configure it with your OpenAI API key. This is done using the `configure` command. The API key is passed as an argument, and it is hidden from the console for security reasons.
If you run:

```bash
llamabot configure
```

You will be prompted to enter your API key.

## Checking the Version

You can check the version of LlamaBot using the `version` command.

```bash
llamabot version
```

## Chatting with LlamaBot

LlamaBot includes a chatbot that you can interact with. To start a chat, use the `chat` command. By default, the chat will be saved to a file. If you don't want to save the chat, you can pass `--no-save` as an argument.

```bash
llamabot chat
# OR
llamabot chat --no-save
```

The chatbot uses a `PromptRecorder` to record the chat. The chat is saved in a Markdown file with the current date and time as the filename.

## Conclusion

That's it! You now know how to use LlamaBot. Remember, you can always get help on the command line by typing `llamabot --help`. Happy coding!
