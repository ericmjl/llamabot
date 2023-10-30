"""Class definition for QueryBot."""
import contextvars
import hashlib
import os
from pathlib import Path
from typing import List, Union
import tiktoken
from copy import deepcopy
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from llama_index import (
    Document,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from loguru import logger

from llamabot.config import default_language_model
from llamabot.doc_processor import magic_load_doc, split_document
from llamabot.recorder import autorecord
from llamabot.bot.model_tokens import model_chat_token_budgets
from llamabot.bot.model_dispatcher import create_model

load_dotenv()

DEFAULT_SIMILARITY_TOP_KS = {
    "gpt-3.5-turbo": 2,
    "gpt-3.5-turbo-0301": 2,
    "gpt-3.5-turbo-0613": 2,
    "gpt-3.5-turbo-16k": 5,
    "gpt-3.5-turbo-16k-0613": 5,
    "gpt-4": 3,
    "gpt-4-0314": 3,
    "gpt-4-0613": 3,
    "gpt-4-32k": 10,
}


SIMILARITY_TOP_K = DEFAULT_SIMILARITY_TOP_KS.get(os.getenv("OPENAI_DEFAULT_MODEL"), 3)


CACHE_DIR = Path.home() / ".llamabot" / "cache"
prompt_recorder_var = contextvars.ContextVar("prompt_recorder")


class QueryBot:
    """QueryBot is a bot that lets us use GPT4 to query documents.

    Pass in either the doc_path or saved_index_path to initialize the QueryBot.

    NOTE: QueryBot is not designed to have memory!

    The default text splitter is the TokenTextSplitter from LangChain.
    The default index that we use is the GPTVectorStoreIndex from LlamaIndex.
    We also default to using GPT4 with temperature 0.0.

    :param system_message: The system message to send to the chatbot.
    :param model_name: The name of the OpenAI model to use.
    :param temperature: The model temperature to use.
        See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
        for more information.
    :param doc_paths: A path to a document,
        or a list of paths to multiple documents,
        to use for the chatbot.
    :param saved_index_path: The path to the saved index to use for the chatbot.
    :param response_tokens: The number of tokens to use for responses.
    :param history_tokens: The number of tokens to use for history.
    :param chunk_sizes: The chunk sizes to use for the LlamaIndex TokenTextSplitter.
        Defaults to [2000], but can be a list of integers.
    :param streaming: Whether to stream the chatbot or not.
    :param verbose: (LangChain config) Whether to print debug messages.
    :param use_cache: Whether to use the cache or not.
    """

    def __init__(
        self,
        system_message: str,
        model_name: str = default_language_model(),
        temperature: float = 0.0,
        doc_paths: List[str] | List[Path] | str | Path = None,
        saved_index_path: str | Path = None,
        response_tokens: int = 2000,
        history_tokens: int = 2000,
        chunk_sizes: list[int] = [2000],
        streaming: bool = True,
        verbose: bool = True,
        use_cache: bool = True,
    ):
        chat = create_model(
            model_name=model_name,
            temperature=temperature,
            streaming=streaming,
            verbose=verbose,
        )
        llm_predictor = LLMPredictor(llm=chat)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        # Initialize vector index or load it from disk.
        vector_index = GPTVectorStoreIndex(nodes=[], service_context=service_context)
        if saved_index_path is not None:
            vector_index = GPTVectorStoreIndex.load_from_disk(
                saved_index_path, service_context=service_context
            )

        # Override vector_index if doc_paths is specified.
        if doc_paths is not None:
            vector_index = make_or_load_vector_index(
                service_context, doc_paths, chunk_sizes=chunk_sizes, use_cache=use_cache
            )

        # Set object attributes.
        self.model_name = model_name
        self.system_message = system_message
        self.vector_index = vector_index
        self.doc_paths = doc_paths
        self.chat = chat
        self.chat_history = [
            SystemMessage(content=system_message),
            SystemMessage(
                content="""Do not hallucinate content.
If you cannot answer something, respond by saying that you don't know.
"""
            ),
        ]
        # Store a mapping of query to source nodes.
        self.source_nodes: dict = {}
        self.chunk_sizes = chunk_sizes
        self.response_tokens = response_tokens
        self.history_tokens = history_tokens
        self.service_context = service_context

    def __call__(
        self,
        query: str,
        similarity_top_k: int = 20,
    ) -> AIMessage:
        """Call the QueryBot.

        :param query: The query to send to the document index.
        :param similarity_top_k: The number of documents to return from the index.
            These documents are added to the context of the chat history
            and then used to synthesize the response.
            Default value is dynamically set based on the default model.
        :raises ValueError: if the index is not initialized.
        :return: The response to the query generated by GPT4.
        """
        if self.vector_index is None:
            raise ValueError(
                "You need to provide a document for querybot to index against!"
            )
        # Step 1: Get documents from the index that are deemed to be matching the query.
        logger.info(f"Querying index for top {similarity_top_k} documents...")
        retriever = VectorIndexRetriever(
            index=self.vector_index, similarity_top_k=similarity_top_k
        )

        # Get source nodes to include in faux chat history,
        # whose token budget is determined by
        # the maximum number of tokens - the response and history tokens.
        source_nodes = retriever.retrieve(query)

        faux_chat_history = []
        faux_chat_history.append(SystemMessage(content=self.system_message))

        # Step 2: Grab as many messages from chat history
        # as permissible by the history_tokens budget,
        enc = tiktoken.encoding_for_model("gpt-4")
        token_budget = deepcopy(self.history_tokens)
        for message in self.chat_history[::-1]:
            if token_budget > 0:
                faux_chat_history.append(message)
                tokens = enc.encode(message.content)
                token_budget -= len(tokens)

        # Step 3: Stuff in as many source nodes as are permitted by the response token
        faux_chat_history.append(
            SystemMessage(content="Here is the context you will be working with:")
        )
        token_budget = (
            model_chat_token_budgets[self.model_name]
            - self.response_tokens
            - self.history_tokens
        )
        actual_source_nodes = []  # the actual source nodes that were stuffed in
        for n in source_nodes:
            if token_budget > 0:
                faux_chat_history.append(SystemMessage(content=n.node.text))
                tokens = enc.encode(n.node.text)
                token_budget -= len(tokens)
                actual_source_nodes.append(n)

        faux_chat_history.append(
            SystemMessage(content="Based on this context, answer the following query:")
        )

        faux_chat_history.append(HumanMessage(content=query))

        # Step 4: Send the chat history through the model
        response = self.chat(faux_chat_history)

        # Step 5: Record only the human response
        # and the GPT response but not the original.
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(response)

        # Step 6: Record the source nodes of the query.
        self.source_nodes[query] = actual_source_nodes

        autorecord(query, response.content)

        # Step 7: Return the response.
        return response

    # @validate_call
    def save(self, path: Union[str, Path]):
        """Save the QueryBot index to disk.

        :param path: The path to save the QueryBot index.
        """
        path = Path(path)
        if not path.suffix == ".json":
            path = path.with_suffix(".json")
        self.vector_index.save_to_disk(path)

    def retrieve(
        self,
        query: str,
        similarity_top_k: int = SIMILARITY_TOP_K,
    ):
        """Retrieve the source nodes associated with a query using similarity search.

        :param query: The query to send to the document index.
        :param similarity_top_k: The number of documents to return from the index.
        :raises ValueError: if the index is not initialized.
        :return: The source nodes associated with the query.
        """
        if self.vector_index is None:
            raise ValueError(
                "You need to provide a document for querybot to index against!"
            )
        retriever = VectorIndexRetriever(
            index=self.vector_index, similarity_top_k=similarity_top_k
        )
        source_nodes = retriever.retrieve(query)
        return source_nodes


# @validate_call
def load_index(persist_dir: Path, service_context: ServiceContext):
    """Load an index from disk.

    :param persist_dir: The directory to load the index from.
    :param service_context: The service context to use for the index.
    :returns: The index.
    """
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
    )
    index = load_index_from_storage(storage_context, service_context=service_context)
    return index


def make_default_storage_context() -> StorageContext:
    """Make a default storage context for the QueryBot.

    :returns: A storage context.
    """
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore(),
        vector_store=SimpleVectorStore(),
        index_store=SimpleIndexStore(),
    )
    return storage_context


def make_vector_index(
    docs: List[Document],
    file_hash: str,
    persist_dir: Path,
    service_context: ServiceContext,
    chunk_size: int = 2000,
    chunk_overlap: int = 0,
) -> GPTVectorStoreIndex:
    """Make an index from a list of documents.

    :param docs: A list of documents to index.
    :param file_hash: The hash of the file to use as the index id.
    :param persist_dir: The directory to persist the index to.
    :param service_context: The service context to use for the index.
    :param chunk_size: The chunk size to use for the llama_index text splitter.
    :param chunk_overlap: The chunk overlap to use for the llama_index text splitter.
    :returns: The index.
    """
    # create parser and parse document into nodes
    parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    nodes = parser.get_nodes_from_documents(docs)

    # create (or load) docstore and add nodes
    storage_context = make_default_storage_context()
    storage_context.docstore.add_documents(nodes)

    vector_index = GPTVectorStoreIndex(
        nodes,
        storage_context=storage_context,
        index_id=file_hash,
        service_context=service_context,
    )
    vector_index.storage_context.persist(persist_dir=persist_dir)
    return vector_index


def make_or_load_vector_index(
    service_context: ServiceContext,
    doc_paths: List[Path] | List[str],
    chunk_sizes: list[int] = [200, 500, 1000, 2000],
    chunk_overlap: int = 0,
    use_cache: bool = True,
) -> GPTVectorStoreIndex:
    """Make or load an index for a collection of documents.

    :param doc_paths: The path to the document to make or load an index for.
    :param chunk_sizes: The chunk sizes to use for the LangChain TokenTextSplitter.
    :param chunk_overlap: The chunk overlap to use for the LangChain TokenTextSplitter.
    :param use_cache: Whether to use the cache.
    :returns: The index.
    """
    # An index is constructed over a collection of documents.
    # As such, we construct a mapping of a hash of
    # the collection of documents to the index.

    # Step 1: Compute file hash for all of the documents
    # + the chunk size and chunk overlap.
    file_hash = hashlib.sha256()
    from tqdm.auto import tqdm

    for doc_path in tqdm(doc_paths, desc="hash files"):
        file_hash.update(Path(doc_path).read_bytes())
    file_hash.update(str(chunk_sizes).encode())
    file_hash.update(str(chunk_overlap).encode())
    file_hash_hexdigest = file_hash.hexdigest()
    logger.info(f"File hash: {file_hash_hexdigest[0:8]}")

    # Make persist_dir based on the file hash's hexdigest.
    persist_dir = CACHE_DIR / file_hash_hexdigest
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Create the index's split documents.
    split_docs = []
    for doc_path in tqdm(doc_paths, desc="split documents"):
        document = magic_load_doc(doc_path)
        for chunk_size in chunk_sizes:
            splitted_document = split_document(document[0], chunk_size, chunk_overlap=0)
            split_docs.extend(splitted_document)

    # Check that the persist directory exists and that we have made a docstore,
    # which is a sentinel test for the rest of the index.
    if persist_dir.exists() and (persist_dir / "docstore.json").exists() and use_cache:
        return load_index(persist_dir, service_context=service_context)

    return make_vector_index(
        split_docs,
        file_hash_hexdigest,
        persist_dir,
        service_context=service_context,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
