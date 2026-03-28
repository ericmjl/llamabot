# QueryBot and AsyncQueryBot

**QueryBot** answers questions using retrieval-augmented generation (RAG). It
holds a reference to a document store. For each user query it retrieves
relevant text chunks from the store, prepends them to the conversation, and
asks the language model to answer using that context.

**AsyncQueryBot** is the async variant. It exposes ``stream_async`` for
token-by-token streaming after retrieval, which pairs well with Server-Sent
Events in a web UI.

Common document stores include **BM25DocStore** (lexical, in-memory, no
embeddings) and **LanceDBDocStore** (dense embeddings with LanceDB). This demo
uses BM25 so it runs quickly without downloading embedding weights.

The system prompt should instruct the model to rely on the retrieved chunks and
to admit when the context does not contain an answer.
