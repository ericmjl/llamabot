# How to Run Llamabot with Ollama

## Overview

In this guide, you'll learn how to run a chatbot using `llamabot` and `Ollama`. We'll cover how to install Ollama, start its server, and finally, run the chatbot within a Python session.

---

## Installation & Setup

### Install Ollama

1. **macOS Users**: [Download here](https://ollama.ai/download/Ollama-darwin.zip)
2. **Linux & WSL2 Users**: Run `curl https://ollama.ai/install.sh | sh` in your terminal
3. **Windows Users**: Support coming soon.

For more detailed instructions, refer to [Ollama's official site](https://ollama.ai/).

---

## Running Ollama Server

1. Open your terminal and start the Ollama server with your chosen model.

```bash
ollama run <model_name>
```

**Example:**

```bash
ollama run vicuna
```

For a list of available models, visit [Ollama's Model Library](https://ollama.ai/library).

> **Note**: Ensure you have adequate RAM for the model you are running.

---

## Running Llamabot in Python

1. Open a Python session and import the `SimpleBot` class from the `llamabot` library.

```python
from llamabot import SimpleBot  # you can also use QueryBot or StructuredBot

bot = SimpleBot("You are a conversation expert", model_name="ollama_chat/vicuna:7b-16k")
```

> Note: `vicuna:7b-16k` includes tags from the [vicuna model page](https://ollama.ai/library/vicuna/tags).

---

And there you have it! You're now ready to run your own chatbot with Ollama and Llamabot.
