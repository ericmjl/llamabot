# Llamabot Zotero CLI Tutorial

In this tutorial, we will learn how to use the Llamabot Zotero CLI, a command-line interface for interacting with your Zotero library. The CLI allows you to sync your Zotero items to a local JSON file, chat with a paper, and configure the environment variables for the CLI.

## Prerequisites

We assume that you have `llamabot` installed:

```bash
pip install -U llamabot
```

This will install the necessary packages and dependencies.

## Configuration

Before using the CLI, you need to configure the environment variables for your Zotero library. To do this, run the following command:

```bash
llamabot zotero configure
```

You will be prompted to enter your Zotero library ID, API key, and library type. The library type can be either "user" or "group".

```plaintext
Library ID: <your_library_id>
API Key: <your_api_key>
Library Type: user
```

This will set the environment variables for the CLI.

## Syncing Zotero Items

To sync your Zotero items to a local JSON file, run the following command:

```bash
llamabot zotero sync
```

This will download all your Zotero items and save them to a JSON file located at `~/.llamabot/zotero/zotero_index.json`.

## Chat with a Paper

To chat with a paper from your Zotero library, run the following command:

```bash
llamabot zotero chat-paper "<query>"
```

Replace `<query>` with a search term related to the paper, such as the title, author, or other metadata. For example:

```bash
llamabot zotero chat-paper "Deep Learning"
```

The CLI will search your Zotero library for the most relevant paper, download the PDF, and initialize a chatbot that can answer questions about the paper. You can ask questions by typing them in the command prompt:

```plaintext
Ask me a question: What is the main contribution of this paper?
```

To exit the chat, press `Ctrl+C`.

## Conclusion

In this tutorial, we learned how to use the Llamabot Zotero CLI to configure environment variables, sync Zotero items to a local JSON file, and chat with a paper from your Zotero library. This powerful tool can help you better understand and interact with your research papers, making it an invaluable resource for researchers and students alike.
