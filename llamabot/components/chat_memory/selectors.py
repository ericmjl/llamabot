"""Node selection strategies for chat memory."""

from abc import ABC, abstractmethod
from typing import List, Optional
import networkx as nx
from pydantic import BaseModel, Field
from llamabot.components.messages import BaseMessage, SystemMessage
from llamabot.bot.structuredbot import StructuredBot


class NodeSelection(BaseModel):
    """Pydantic model for LLM node selection output."""

    selected_node_id: int = Field(..., description="The ID of the selected parent node")
    reasoning: str = Field(
        ..., description="Brief explanation of why this node was selected"
    )


class NodeSelector(ABC):
    """Abstract base class for node selection strategies."""

    @abstractmethod
    def select_parent(self, graph: nx.DiGraph, message: BaseMessage) -> Optional[int]:
        """Select the best parent node for a new message.

        :param graph: The conversation graph
        :param message: The message to find a parent for
        :return: Node ID of the selected parent, or None for root
        """
        raise NotImplementedError()


class LinearNodeSelector(NodeSelector):
    """Linear node selector that chooses the leaf node (node with no out_edges)."""

    def select_parent(self, graph: nx.DiGraph, message: BaseMessage) -> Optional[int]:
        """Select the leaf node as parent.

        :param graph: The conversation graph
        :param message: The message to find a parent for
        :return: Node ID of the leaf node, or None if no nodes exist
        """
        if not graph.nodes():
            return None

        # Find nodes with no out_edges (leaf nodes)
        leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

        if not leaf_nodes:
            return None

        # Return the leaf node (in a linear graph, there should be only one)
        return leaf_nodes[0]


class LLMNodeSelector(NodeSelector):
    """LLM-based node selector that uses intelligent threading.

    :param model: LLM model name to use for node selection
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._bot = None  # Lazy initialization

    @property
    def bot(self) -> StructuredBot:
        """Get or create the StructuredBot for node selection."""
        if self._bot is None:
            self._bot = StructuredBot(
                system_prompt=SystemMessage(
                    content="""You are an expert at analyzing conversation threads and selecting the most appropriate parent node for new messages.

Your task is to examine the conversation context and choose which existing assistant message should be the parent of a new user message.

Consider:
1. Direct topic continuation - if the new message directly continues a topic
2. Related concepts - if the new message relates to concepts discussed in a previous message
3. Temporal relevance - recent messages are often more relevant than older ones
4. Contextual flow - maintain logical conversation flow

You must select a valid assistant node ID from the provided candidates."""
                ),
                pydantic_model=NodeSelection,
                model_name=self.model,
            )
        return self._bot

    def select_parent(self, graph: nx.DiGraph, message: BaseMessage) -> Optional[int]:
        """Select the best parent node using LLM intelligence.

        :param graph: The conversation graph
        :param message: The message to find a parent for
        :return: Node ID of the selected parent, or None for root
        """
        candidates = self._get_candidate_nodes(graph)

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Use StructuredBot to select best parent
        try:
            context = self._build_selection_context(graph, message, candidates)

            prompt = f"""Conversation Context:
{context}

New Message: {message.content}

Available parent nodes (assistant messages only):
{", ".join(map(str, candidates))}

Select the node ID that is most semantically relevant to the new message."""

            response = self.bot(prompt)

            # Validate the selected node
            if response.selected_node_id in candidates:
                return response.selected_node_id

        except Exception:
            # Fall back to most recent node on any error
            pass

        # Fallback to most recent valid node
        return candidates[-1] if candidates else None

    def _get_candidate_nodes(self, graph: nx.DiGraph) -> List[int]:
        """Get candidate assistant nodes for selection.

        :param graph: The conversation graph
        :return: List of assistant node IDs
        """
        candidates = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data["node"].message.role == "assistant":
                candidates.append(node_id)
        return sorted(candidates)

    def _build_selection_context(
        self, graph: nx.DiGraph, message: BaseMessage, candidates: List[int]
    ) -> str:
        """Build context string for LLM node selection.

        :param graph: The conversation graph
        :param message: The message to find a parent for
        :param candidates: List of candidate node IDs
        :return: Context string for LLM
        """
        context_parts = []

        for node_id in candidates:
            node_data = graph.nodes[node_id]["node"]
            context_parts.append(f"Node {node_id}: {node_data.message.content}")

        return "\n".join(context_parts)
