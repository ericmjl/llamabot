# LlamaBot overview

LlamaBot is a Python library for working with large language models through a
simple, composable API. It uses LiteLLM so you can swap providers and models
without rewriting application code.

The main building blocks are **bots** (callable objects that wrap prompts and
model calls), **messages** (system, human, and assistant content), and optional
**components** such as document stores for retrieval.

The package encourages a functional style in helpers and a small set of
parameterized bot classes for LLM-facing code paths.
