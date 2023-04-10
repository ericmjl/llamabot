"""Bot abstractions that let me quickly build new GPT-based applications."""

import os
from pathlib import Path
from typing import List, Union

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import TokenTextSplitter
from llama_index import Document, GPTSimpleVectorIndex, LLMPredictor, ServiceContext

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class SimpleBot:
    """Simple Bot that is primed with a system prompt, accepts a human message, and sends back a single response.

    This bot does not retain chat history.
    """

    def __init__(self, system_prompt, temperature=0.0):
        """Initialize the SimpleBot.

        :param system_prompt: The system prompt to use.
        :param temperature: The temperature to use.
        """
        self.system_prompt = system_prompt
        self.model = ChatOpenAI(model_name="gpt-4", temperature=temperature)

    def __call__(self, human_message):
        """Call the SimpleBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_message),
        ]
        return self.model(messages)


class ChatBot:
    """Chat Bot that is primed with a system prompt, accepts a human message.

    Automatic chat memory management happens.

    h/t Andrew Giessel/GPT4 for the idea.
    """

    def __init__(self, system_prompt, temperature=0.0):
        """Initialize the ChatBot.

        :param system_prompt: The system prompt to use.
        :param temperature: The temperature to use.
        """
        self.model = ChatOpenAI(model_name="gpt-4", temperature=temperature)
        self.chat_history = [SystemMessage(content=system_prompt)]

    def __call__(self, human_message) -> str:
        """Call the ChatBot.

        :param human_message: The human message to use.
        :return: The response to the human message, primed by the system prompt.
        """
        self.chat_history.append(HumanMessage(content=human_message))
        response = self.model(self.chat_history)
        self.chat_history.append(response)
        return response.content

    def __repr__(self):
        """Return a string representation of the ChatBot.

        :return: A string representation of the ChatBot.
        """
        representation = ""

        for message in self.chat_history:
            if isinstance(message, SystemMessage):
                prefix = "[System]\n"
            elif isinstance(message, HumanMessage):
                prefix = "[Human]\n"
            elif isinstance(message, AIMessage):
                prefix = "[AI]\n"

            representation += f"{prefix}{message.content}" + "\n\n"
        return representation


class QueryBot:
    """QueryBot is a bot that lets us use GPT4 to query documents."""

    def __init__(
        self,
        system_message: str,
        doc_paths: List[Union[str, Path]] = None,
        saved_index_path: Union[str, Path] = None,
    ):
        """Initialize QueryBot.

        Pass in either the doc_paths or saved_index_path to initialize the QueryBot.

        NOTE: QueryBot is not designed to have memory!

        The default text splitter is the TokenTextSplitter from LangChain.
        The default index that we use is the GPTSimpleVectorIndex from LlamaIndex.
        We also default to using GPT4 with temperature 0.0.

        :param system_message: The system message to send to the chatbot.
        :param doc_paths: A list of paths to the documents to use for the chatbot.
            These are assumed to be plain text files.
        :param saved_index_path: The path to the saved index to use for the chatbot.
        """

        self.system_message = system_message

        chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)
        llm_predictor = LLMPredictor(llm=chat)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        # Build index
        if saved_index_path is not None:
            index = GPTSimpleVectorIndex.load_from_disk(
                saved_index_path, service_context=service_context
            )

        else:
            self.doc_paths = doc_paths
            splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=0)
            documents = []
            for fname in doc_paths:
                with open(fname, "r") as f:
                    docs = splitter.split_text(f.read())
                    documents.extend([Document(d) for d in docs])
            index = GPTSimpleVectorIndex.from_documents(
                documents, service_context=service_context
            )
        self.index = index

    def __call__(self, query: str, **kwargs) -> str:
        """Call the QueryBot.

        :param query: The query to send to the document index.
        :param kwargs: Additional keyword arguments to pass to the chatbot.
            These are passed into LlamaIndex's index.query() method.
        :return: The response to the query generated by GPT4.
        """
        q = ""
        q += self.system_message + "\n\n"
        q += query + "\n\n"
        result = self.index.query(q, **kwargs)
        return result

    def save(self, path: Union[str, Path]):
        """Save the QueryBot and index to disk.

        :param path: The path to save the QueryBot index.
        """
        path = Path(path)
        if not path.suffix == ".json":
            path = path.with_suffix(".json")
        self.index.save_to_disk(path)
