# Llamabot Zotero CLI Tutorial

In this tutorial, we will walk through the Llamabot Zotero CLI, a command-line interface for interacting with your Zotero library. This tool allows you to chat with a paper, retrieve keys, and download papers from your Zotero library.

## Prerequisites

Before we start, make sure you have `llamabot` installed in your environment:

```bash
pip install -U llamabot
```

## Getting Started

First, we need to configure the Llamabot Zotero CLI environment variables. This is done using the `configure` command. You will be prompted to enter your Zotero library ID, API key, and library type.

```bash
llamabot zotero configure
```

## Chatting with a Paper

To chat with a paper, use the `chat` command. You can specify the paper you want to chat about as an argument. If you don't provide a paper, you will be prompted to enter one.

```bash
llamabot zotero chat "The title of the paper"
```

If you want to synchronize your Zotero library before chatting, you can use the `--sync` option.

```bash
llamabot zotero chat "The title of the paper" --sync
```

## Retrieving Keys

When you chat with a paper, the Llamabot Zotero CLI will retrieve the keys for the paper. These keys are unique identifiers for each paper in your Zotero library. The keys are displayed in the console.

## Downloading Papers

After retrieving the keys, you can choose a paper to download. You will be prompted to choose a paper from the list of keys. The paper will be downloaded to a temporary directory.

```bash
Please choose an option: The title of the paper
```

## Asking Questions

Once the paper is downloaded, you can start asking questions about the paper. The Llamabot Zotero CLI uses a QueryBot to answer your questions. Simply type your question at the prompt.

```bash
Ask me a question: What is the main argument of the paper?
```

To exit the chat, type `exit`.

```bash
Ask me a question: exit
```

And that's it! You now know how to use the Llamabot Zotero CLI to chat with a paper, retrieve keys, download papers, and ask questions about a paper. Happy chatting!
