"""
This module provides a Panel Markdown callback handler for streaming.

Classes:
    - PanelMarkdownCallbackHandler: Callback handler for streaming. Only works with LLMs that support streaming.

Exceptions:
    - None

Functions:
    - None
"""
from typing import Any, Dict, List, Union

import panel as pn
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class PanelMarkdownCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, markdown_object: pn.pane.Markdown):
        self.md = markdown_object

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running.

        # noqa: DAR101
        """

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled.

        # noqa: DAR101
        """
        self.md.object += f"{token}"

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        # noqa: DAR101
        """

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors.

        # noqa: DAR101
        #"""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running.

        # noqa: DAR101
        """

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running.

        # noqa: DAR101
        #"""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors.

        # noqa: DAR101
        """

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running.

        # noqa: DAR101
        """

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action.

        # noqa: DAR101
        """
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running.

        # noqa: DAR101
        """

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors.

        # noqa: DAR101
        """

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text.

        # noqa: DAR101
        """

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end.

        # noqa: DAR101
        """
