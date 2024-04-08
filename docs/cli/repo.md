# Chatting with a Code Repository: Llamabot CLI Guide

Welcome to the guide on using the Llamabot CLI for interacting with code repositories. This innovative tool leverages AI to facilitate engaging and insightful conversations with your codebase. Discover how to effectively use this tool to read and understand documentation within a repository.

## Getting started

Before you can start chatting with your code repository, ensure that Llamabot CLI is installed on your system. You can install it via pip with the following command:

```bash
pip install -U llamabot
```

Once installed, you can access the CLI using the `llamabot repo chat` command.

## Key commands

Llamabot CLI introduces the `chat` command, allowing for dynamic interactions with your code repository.

### Chat with your repository

The `chat` command allows you to interact with your code repository in a conversational manner.

#### Usage

Run the command below to start a conversation with your repository:

```bash
llamabot repo chat --repo-url https://github.com/yourusername/yourrepo --checkout="branch_or_tag" --source-file-extensions py --source-file-extensions md --model-name="gpt-4-0125-preview"
```

Here are the key parameters to understand:

- `--repo-url`: URL of the git repository you want to interact with.
- `--checkout`: Specify the branch or tag you wish to use. The default is "main".
- `--source-file-extensions`: Define the types of source files to include in the conversation. Supports a variety of file extensions.
- `--model-name`: AI model to be used for generating responses.
- `--initial-message` (optional): Initial message to start the conversation.
- `--panel` (optional): Set to `true` to launch a Panel web app to chat.

After executing the command, Llamabot clones the repository into a temporary directory, processes the files as specified, and starts the chat interface. If `--panel` is true, the chat interface will be served in a browser.

## Conclusion

This guide covers the essential aspects of the Llamabot CLI, a tool designed to enhance your coding experience through AI-powered conversations about a code repository. Embrace these capabilities to make your coding more efficient and insightful. Happy coding!
