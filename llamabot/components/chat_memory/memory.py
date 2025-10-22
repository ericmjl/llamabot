"""Main ChatMemory class for unified chat memory system."""

import json
import networkx as nx
from typing import List, Optional
from datetime import datetime
from llamabot.components.messages import (
    BaseMessage,
    AIMessage,
)
from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.chat_memory.models import ConversationNode, MessageSummary
from llamabot.components.chat_memory.selectors import (
    NodeSelector,
    LinearNodeSelector,
    LLMNodeSelector,
)
from llamabot.components.chat_memory.storage import (
    append_linear_message,
    append_threaded_message,
)
from llamabot.components.chat_memory.retrieval import (
    get_recent_messages,
    semantic_search_with_context,
)
from llamabot.components.chat_memory.visualization import to_mermaid
from llamabot.prompt_manager import prompt


@prompt("system")
def message_summarizer_system_prompt() -> str:
    """You are an expert at summarizing messages. Create concise, informative summaries that capture the key points and intent of each message.

    Your task is to analyze the content of a message and provide:
    1. A clear, descriptive title that captures the main topic
    2. A concise summary (maximum two sentences) that highlights the key points

    Focus on:
    - The main intent or purpose of the message
    - Key concepts or topics discussed
    - Any specific requests or questions
    - Important details that would be useful for conversation threading
    """


class Summarizer:
    """Abstract base class for message summarization."""

    def summarize(self, message: BaseMessage) -> Optional[MessageSummary]:
        """Summarize a message.

        :param message: The message to summarize
        :return: Message summary or None if summarization fails
        """
        raise NotImplementedError()


class LLMSummarizer(Summarizer):
    """LLM-based message summarizer.

    :param model: LLM model name to use for summarization
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.bot = StructuredBot(
            system_prompt=message_summarizer_system_prompt(),
            pydantic_model=MessageSummary,
            model_name=self.model,
        )

    def summarize(self, message: BaseMessage) -> Optional[MessageSummary]:
        """Summarize a message using LLM.

        :param message: The message to summarize
        :return: Message summary or None if summarization fails
        """
        try:
            prompt = f"Summarize this message:\n\n{message.content}"
            result = self.bot(prompt)
            if isinstance(result, MessageSummary):
                return result
            return None
        except Exception:
            return None


class ChatMemory:
    """Unified chat memory system with configurable threading and retrieval.

    :param node_selector: Strategy for selecting parent nodes (None = LinearNodeSelector)
    :param summarizer: Optional summarization strategy (None = no summarization)
    :param context_depth: Default depth for context retrieval
    """

    def __init__(
        self,
        node_selector: Optional[NodeSelector] = None,
        summarizer: Optional[Summarizer] = None,
        context_depth: int = 5,
    ):
        # Initialize NetworkX graph for storage
        self.graph = nx.DiGraph()

        # Set node selector (linear by default, LLM-based if provided)
        self.node_selector = node_selector or LinearNodeSelector()

        # Set optional summarizer
        self.summarizer = summarizer

        # Validate and store context depth
        if context_depth < 0:
            raise ValueError("context_depth must be non-negative")
        self.context_depth = context_depth

        # Track next node ID for auto-incrementing
        self._next_node_id = 1

    @classmethod
    def threaded(cls, model: str = "gpt-4o-mini", **kwargs) -> "ChatMemory":
        """Create ChatMemory with LLM-based threading.

        :param model: LLM model name for node selection and summarization
        :param kwargs: Additional arguments passed to ChatMemory constructor
        """
        return cls(
            node_selector=LLMNodeSelector(model=model),
            summarizer=LLMSummarizer(
                model=model
            ),  # Optional but recommended for threading
            **kwargs,
        )

    def retrieve(
        self, query: str, n_results: int = 10, context_depth: Optional[int] = None
    ) -> List[BaseMessage]:
        """Smart retrieval that adapts based on memory configuration.

        :param query: The search query
        :param n_results: Number of results to return
        :param context_depth: Context depth (uses default if None)
        :return: List of relevant messages
        """
        context_depth = context_depth or self.context_depth

        if isinstance(self.node_selector, LinearNodeSelector):
            return get_recent_messages(self.graph, n_results)
        else:
            return semantic_search_with_context(
                self.graph, query, n_results, context_depth
            )

    def append(self, message: BaseMessage):
        """Add a message to memory.

        :param message: Any message to append (HumanMessage, AIMessage, etc.)
        """
        if isinstance(self.node_selector, LinearNodeSelector):
            append_linear_message(self.graph, message, self._next_node_id)
        else:
            append_threaded_message(
                self.graph, message, self.node_selector, self._next_node_id
            )
        self._next_node_id += 1

    def reset(self):
        """Reset the memory system."""
        self.graph.clear()
        self._next_node_id = 1

    def to_mermaid(self, **kwargs) -> str:
        """Convert conversation to Mermaid diagram.

        :param kwargs: Arguments passed to to_mermaid function
        :return: Mermaid diagram string
        """
        return to_mermaid(self.graph, **kwargs)

    def save(self, file_path: str):
        """Save conversation memory to JSON file.

        :param file_path: Path to save the file
        """
        data = self._to_json()

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, file_path: str) -> "ChatMemory":
        """Load conversation memory from JSON file.

        :param file_path: Path to load the file from
        :return: ChatMemory instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        return cls._from_json(data)

    def export(self, format: str = "json") -> str:
        """Export conversation in various formats.

        :param format: Export format ("json", "jsonl", "plain_text", "mermaid")
        :return: Exported data as string
        """
        if format == "json":
            return json.dumps(self._to_json(), indent=2, default=str)
        elif format == "jsonl":
            return self._to_jsonl()
        elif format == "plain_text":
            return self._to_plain_text()
        elif format == "mermaid":
            return self.to_mermaid()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _to_json(self) -> dict:
        """Convert conversation memory to JSON-serializable dict."""
        nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            node = node_data["node"]
            node_dict = {
                "id": node.id,
                "role": node.message.role,
                "content": node.message.content,
                "timestamp": node.timestamp.isoformat(),
                "parent_id": node.parent_id,
            }

            if node.summary:
                node_dict["summary"] = {
                    "title": node.summary.title,
                    "summary": node.summary.summary,
                }

            nodes.append(node_dict)

        edges = [{"from": u, "to": v} for u, v in self.graph.edges()]

        return {
            "version": "1.0",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "mode": (
                    "graph"
                    if isinstance(self.node_selector, LLMNodeSelector)
                    else "linear"
                ),
                "total_messages": len(self.graph.nodes()),
            },
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def _from_json(cls, data: dict) -> "ChatMemory":
        """Reconstruct conversation memory from JSON data."""
        # Determine mode from metadata
        mode = data.get("metadata", {}).get("mode", "linear")

        if mode == "graph":
            memory = cls(node_selector=LLMNodeSelector())
        else:
            memory = cls()

        # Reconstruct nodes
        for node_data in data["nodes"]:
            from llamabot.components.messages import user

            # Create message
            if node_data["role"] == "user":
                user_result = user(node_data["content"])
                # Handle the case where user() returns a list
                if isinstance(user_result, list):
                    message = user_result[0]  # Take first message if list
                else:
                    message = user_result
            else:
                message = AIMessage(content=node_data["content"])

            # Create summary if present
            summary = None
            if "summary" in node_data:
                summary = MessageSummary(**node_data["summary"])

            # Create node
            node = ConversationNode(
                id=node_data["id"],
                message=message,
                summary=summary,
                parent_id=node_data["parent_id"],
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
            )

            memory.graph.add_node(node_data["id"], node=node)

        # Reconstruct edges
        for edge in data["edges"]:
            memory.graph.add_edge(edge["from"], edge["to"])

        # Update next node ID
        if memory.graph.nodes():
            memory._next_node_id = max(memory.graph.nodes()) + 1

        return memory

    def _to_jsonl(self) -> str:
        """Convert to JSONL format for fine-tuning."""
        lines = []
        for node_id in sorted(self.graph.nodes()):
            node_data = self.graph.nodes[node_id]["node"]
            message = node_data.message

            line = {"role": message.role, "content": message.content}
            lines.append(json.dumps(line))

        return "\n".join(lines)

    def _to_plain_text(self) -> str:
        """Convert to plain text format."""
        lines = []
        for node_id in sorted(self.graph.nodes()):
            node_data = self.graph.nodes[node_id]["node"]
            message = node_data.message

            role = "User" if message.role == "user" else "Assistant"
            lines.append(f"{role}: {message.content}")

        return "\n\n".join(lines)
