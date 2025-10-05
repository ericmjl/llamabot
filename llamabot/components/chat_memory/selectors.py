"""Node selection strategies for chat memory."""

from abc import ABC, abstractmethod
from typing import List, Optional
import networkx as nx
from pydantic import BaseModel, Field, model_validator
from llamabot.components.messages import BaseMessage
from llamabot.bot.structuredbot import StructuredBot
from llamabot.prompt_manager import prompt
from loguru import logger


def get_candidate_nodes(graph: nx.DiGraph) -> List[int]:
    """Get candidate assistant nodes for selection.

    :param graph: The conversation graph
    :return: List of assistant node IDs
    """
    candidates = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data["node"].message.role == "assistant":
            candidates.append(node_id)
    return sorted(candidates)


def build_selection_context(
    graph: nx.DiGraph, message: BaseMessage, candidates: List[int]
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


def validate_node_selection(
    graph: nx.DiGraph, node_id: int, candidates: List[int]
) -> bool:
    """Validate that a node selection is valid.

    :param graph: The conversation graph
    :param node_id: The selected node ID
    :param candidates: List of valid candidate node IDs
    :return: True if valid, False otherwise
    """
    try:
        # Check if node exists
        if node_id not in graph.nodes():
            return False

        # Check if node is an assistant node
        node_data = graph.nodes[node_id]["node"]
        if node_data.message.role != "assistant":
            return False

        # Check if node is in candidate list
        if node_id not in candidates:
            return False

        return True

    except Exception:
        return False


@prompt("system")
def node_selection_system_prompt(candidate_nodes: str):
    """You are an expert at analyzing conversation threads and selecting the most appropriate parent node for new messages.

    Your task is to examine the conversation context and choose which existing assistant message should be the parent of a new user message.

    Consider:
    1. Direct topic continuation - if the new message directly continues a topic
    2. Related concepts - if the new message relates to concepts discussed in a previous message
    3. Temporal relevance - recent messages are often more relevant than older ones
    4. Contextual flow - maintain logical conversation flow

    You must select a valid assistant node ID from the provided candidates.

    Available candidate nodes: {{candidate_nodes}}

    If your selection is invalid, you will receive feedback and must try again with a valid node ID.
    """


class NodeSelection(BaseModel):
    """Pydantic model for LLM node selection output with validation."""

    selected_node_id: int = Field(..., description="The ID of the selected parent node")
    reasoning: str = Field(
        ..., description="Brief explanation of why this node was selected"
    )

    @model_validator(mode="after")
    def validate_node_selection(self):
        """Validate that the selected node ID is valid.

        This validator will be called by StructuredBot and any errors will be
        fed back to the LLM for retry attempts.
        """
        # Get validation context from the model's metadata
        # Note: This requires StructuredBot to pass context through model_validate
        # For now, we'll do basic validation and let StructuredBot handle retries

        # The actual graph validation will happen in the LLMNodeSelector
        # after the LLM response, but this provides a foundation for
        # more sophisticated validation if needed

        if self.selected_node_id < 0:
            raise ValueError("Selected node ID must be a non-negative integer")

        if not self.reasoning.strip():
            raise ValueError("Reasoning cannot be empty")

        return self


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
    """Linear node selector that chooses the last assistant node."""

    def select_parent(self, graph: nx.DiGraph, message: BaseMessage) -> Optional[int]:
        """Select the last assistant node as parent.

        :param graph: The conversation graph
        :param message: The message to find a parent for
        :return: Node ID of the last assistant node, or None if no assistant nodes exist
        """
        if not graph.nodes():
            return None

        # Get all assistant nodes
        assistant_nodes = get_candidate_nodes(graph)

        if not assistant_nodes:
            return None

        # Return the last assistant node (most recent)
        return assistant_nodes[-1]


class LLMNodeSelector(NodeSelector):
    """LLM-based node selector that uses intelligent threading.

    :param model: LLM model name to use for node selection
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        # We'll create the system prompt dynamically with candidate_nodes
        self.bot = None  # Will be initialized in select_parent

    def select_parent(self, graph: nx.DiGraph, message: BaseMessage) -> Optional[int]:
        """Select the best parent node using LLM intelligence.

        :param graph: The conversation graph
        :param message: The message to find a parent for
        :return: Node ID of the selected parent, or None for root
        """
        candidates = get_candidate_nodes(graph)

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Initialize bot with system prompt that includes candidate nodes
        candidate_nodes_str = ", ".join(map(str, candidates))
        system_prompt = node_selection_system_prompt(candidate_nodes_str)

        if self.bot is None or self.bot.system_prompt != system_prompt:
            self.bot = StructuredBot(
                system_prompt=system_prompt,
                pydantic_model=NodeSelection,
                model_name=self.model,
            )

        # Use StructuredBot to select best parent
        try:
            context = build_selection_context(graph, message, candidates)

            prompt = f"""Conversation Context:
{context}

New Message: {message.content}

Available parent nodes (assistant messages only): {candidate_nodes_str}

Select the node ID that is most semantically relevant to the new message."""

            response = self.bot(prompt, num_attempts=5)

            # Validate the selected node after LLM response
            selected_id = response.selected_node_id
            if validate_node_selection(graph, selected_id, candidates):
                return selected_id
            else:
                # If validation fails, fall back to most recent
                logger.warning(
                    f"LLM selected invalid node {selected_id}. Falling back to most recent."
                )
                return candidates[-1]

        except Exception as e:
            # Fall back to most recent node on any error
            logger.warning(
                f"LLM node selection failed: {e}. Falling back to most recent node."
            )
            return candidates[-1] if candidates else None
