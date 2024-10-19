"""A bot designed to use questions and answers to generate a response."""

from hashlib import sha256
from pathlib import Path
from llamabot.bot.simplebot import SimpleBot
from llamabot.components.docstore import LanceDBDocStore
from llamabot.components.messages import AIMessage
from llamabot.doc_processor import magic_load_doc, split_document
from llamabot.prompt_manager import prompt
import json
import os
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@prompt(role="system")
def jeopardy_bot_sysprompt() -> str:
    """
    You are an expert at taking texts and constructing questions from them.
    You will be given a text.
    Extract as many question-and-answer pairs.
    Each answer may have multiple questions; be sure to cover as many as possible.
    Return a JSON array of the following schema:

    {
        "questions_and_answers": [
            {
                "question": "string",
                "answer": "string",
            },
            ...
        ]
    }
    """


@prompt(role="user")
def q_and_a_prompt(query, q_and_a_results, relevant_documents) -> str:
    """Q&A Results:

        {{ q_and_a_results }}

    Relevant documents:

        {{ relevant_documents }}

    Query:

        {{ query }}
    """


class DocQABot:
    """A bot designed to use pre-computed questions and answers to generate a response."""

    def __init__(self, collection_name: str):
        self.question_store = LanceDBDocStore(
            collection_name=f"{collection_name}_questions"
        )
        self.document_store = LanceDBDocStore(collection_name=f"{collection_name}")

        self.response_bot = SimpleBot(
            "Based on Q&A results and relevant documents, please answer the query.",
            model_name="mistral/mistral-medium",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        self.jeopardy_bot = SimpleBot(
            system_prompt=jeopardy_bot_sysprompt(),
            model_name="gpt-3.5-turbo-1106",
            json_mode=True,
            stream_target="stdout",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def add_documents(
        self, document_paths: Path | list[Path], chunk_size=1_024, chunk_overlap=256
    ):
        """Add documents for the bot to query.

        :param document_paths: The paths to the documents to add.
        :param chunk_size: The size of the chunks to split the documents into.
        :param chunk_overlap: The amount of overlap between chunks.
        """
        if isinstance(document_paths, Path):
            document_paths = [document_paths]
        for document_path in tqdm(document_paths, desc="Document"):
            document = magic_load_doc(document_path)
            chunks = split_document(
                document, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            for chunk in tqdm(chunks, desc="Chunk Q&A"):
                doc_id = sha256(chunk.encode()).hexdigest()
                self.document_store.append(chunk, metadata=dict(doc_id=doc_id))
                q_and_a = json.loads(self.jeopardy_bot(chunk).content)
                for q_and_a in tqdm(
                    q_and_a["questions_and_answers"], desc="Storing Q&A"
                ):
                    q_a_concat = f"Q: {q_and_a['question']} A: {q_and_a['answer']}"
                    self.question_store.append(
                        q_a_concat, metadata=dict(parent_doc=doc_id)
                    )

    def __call__(
        self,
        query_text: str,
        num_questions_retrieved: int = 20,
        num_documents_retrieved: int = 3,
    ) -> AIMessage:
        """Call the QABot.

        This will retrieve the top `num_questions_retrieved` questions and answers
        from the question store, and the top `num_documents_retrieved` documents
        from the document store, and feed them into the response bot.

        :param query_text: The query text to use.
        :param num_questions_retrieved: The number of questions to retrieve.
        :param num_documents_retrieved: The number of documents to retrieve.
        :return: The response to the query, primed by the retrieved questions and
            documents.
        """
        # Retrieve from the question store
        q_and_as = self.question_store.retrieve(query_text)

        # Get the unique parent_doc IDs while preserving order
        result = self.question_store.collection.query(
            query_texts=query_text, n_results=num_questions_retrieved
        )
        parent_doc_ids = []
        for metadata in result["metadatas"][0]:
            if metadata["parent_doc"] not in parent_doc_ids:
                parent_doc_ids.append(metadata["parent_doc"])

        results = self.document_store.collection.query(
            query_texts=query_text,
            where={"doc_id": {"$in": list(parent_doc_ids)}},
            n_results=num_documents_retrieved,
        )

        output = self.response_bot(
            q_and_a_prompt(query_text, q_and_as, results["documents"][0])
        )
        return output
