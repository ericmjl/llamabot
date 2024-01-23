# Chatting with a Code Repository: Llamabot CLI Guide

Welcome to the guide on using the Llamabot CLI for interacting with code repositories. This tool facilitates engaging conversations with your codebase, leveraging the power of AI to understand and read documentation within a repo. Let’s get started on how to utilize this tool.

## Getting Started

Before diving into the commands, ensure you have Llamabot CLI installed. Install it via pip:

```bash
pip install -U llamabot
```

After installation, access the CLI with the `llamabot repo` command.

## Key Commands

Llamabot CLI offers several commands:

1. `chat`: Engage in a conversation with your code repository.

### Chat with Your Repository

The `chat` command allows you to interact with your code repository in a conversational manner.

#### Usage

```bash
llamabot python chat --repo-url https://github.com/ericmjl/llamabot --checkout="main" --source-file-extensions md --source-file-extensions py --model-name="mistral/mistral-medium"
```

- `--repo-url`: URL of the git repository.
- `--checkout`: Branch or tag to use (default: "main").
- `--source-file-extensions`: File types to include in the conversation.
- `--model-name`: AI model to use for generating responses.

Once you have executed this command, LlamaBot will automatically clone the repository to a temporary directory, embed the files as specified by the source-file extensions, and fire up LlamaBot's usual command line-based chat interface.

## Conclusion

This guide covers the essential aspects of the Llamabot CLI, a tool designed to enhance your coding experience through AI-powered conversations about a code repository. Embrace these capabilities to make your coding more efficient and insightful. Happy coding!
