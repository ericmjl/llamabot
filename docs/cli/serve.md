# Serving a QueryBot: Llamabot CLI Guide

Llamabot CLI introduces the `serve` command, creating an Ollama compatible server
for your querybot.

#### Usage

Run the command below to start an Ollama compatible server on port 

```bash
llamabot serve querybot --system-prompt "be smart" --collection-name example --document-paths /path/to/documents.md
```

Here are the key parameters to understand:

- `--system-prompt`: System prompt.
- `--collection-name`: Name of the collection.
- `--document-paths`: Paths to the documents.
- `--model-name`: AI model to be used for generating responses.
- `--host`: Host to serve the API on.
- `--port`: Port to serve the API on.

This creates an Ollama-compatible server so that your QueryBot can be used by any software
that supports Ollama.
