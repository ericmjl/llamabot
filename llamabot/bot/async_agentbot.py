"""Async :class:`~pocketflow.AsyncFlow` wrapper around the same graph as :class:`~llamabot.bot.agentbot.AgentBot`.

:class:`~llamabot.bot.agentbot.AgentBot` uses PocketFlow's synchronous :class:`~pocketflow.Flow`.
This module provides :class:`AsyncAgentBot`, which wires the same ``DecideNode`` and tool
nodes through :class:`~pocketflow.AsyncFlow` with :class:`~pocketflow.AsyncNode` wrappers.

:class:`AsyncDecideNode` runs the decision step natively async: :meth:`~llamabot.components.pocketflow.DecideNode.aexec`
uses :class:`~llamabot.bot.toolbot.AsyncToolBot` and :meth:`~llamabot.bot.toolbot.AsyncToolBot.__call__`
with LiteLLM ``acompletion`` (no :func:`asyncio.to_thread` around the LLM). Tool :class:`~pocketflow.AsyncNode` wrappers still
run sync ``FuncNode`` bodies in :func:`asyncio.to_thread` so CPU-bound or sync tools do not
block other async work.

Use :meth:`AsyncAgentBot.arun` from async contexts (e.g. FastAPI handlers) instead of
``asyncio.to_thread(agent, query)`` on :class:`~llamabot.bot.agentbot.AgentBot`.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Callable, List, Optional

from pocketflow import AsyncFlow, AsyncNode, Node

from llamabot.bot.agentbot import _validate_tools
from llamabot.components.pocketflow import DecideNode
from llamabot.components.tools import DEFAULT_TOOLS


def _tool_routing_name(tool_node: Callable) -> str:
    """Resolve PocketFlow edge label for a tool node.

    :param tool_node: A ``@tool`` / ``FuncNode`` instance.
    :return: Function name used for ``decide >> tool_name >> tool``.
    """
    if hasattr(tool_node, "func"):
        return tool_node.func.__name__
    if hasattr(tool_node, "name"):
        return tool_node.name
    return tool_node.__name__


class AsyncDecideNode(AsyncNode):
    """Runs :class:`DecideNode` with async LLM tool selection (:meth:`DecideNode.aexec`)."""

    def __init__(self, inner: DecideNode) -> None:
        super().__init__()
        self._inner = inner

    async def prep_async(self, shared: dict) -> object:
        """Delegate to sync :meth:`DecideNode.prep` (pure shared-state updates).

        :param shared: PocketFlow shared store.
        :return: Prep result for :meth:`exec_async`.
        """
        return self._inner.prep(shared)

    async def exec_async(self, prep_res: object) -> object:
        """Await :meth:`DecideNode.aexec` (LiteLLM ``acompletion`` via :class:`~llamabot.bot.toolbot.AsyncToolBot`).

        :param prep_res: Output from :meth:`prep_async`.
        :return: Exec result for :meth:`post_async`.
        """
        return await self._inner.aexec(prep_res)

    async def post_async(
        self, shared: dict, prep_res: object, exec_res: object
    ) -> object:
        """Delegate to sync :meth:`DecideNode.post`.

        :param shared: PocketFlow shared store.
        :param prep_res: Prep output.
        :param exec_res: Exec output.
        :return: Routing action for the next node.
        """
        return self._inner.post(shared, prep_res, exec_res)


class AsyncFuncNode(AsyncNode):
    """Runs a sync tool :class:`~pocketflow.Node` inside :meth:`asyncio.to_thread`."""

    def __init__(self, inner: Node) -> None:
        super().__init__()
        self._inner = inner
        self.loopback_name = getattr(inner, "loopback_name", None)

    async def prep_async(self, shared: dict) -> object:
        """Delegate to the wrapped node's :meth:`~pocketflow.Node.prep`.

        :param shared: PocketFlow shared store.
        :return: Prep result for :meth:`exec_async`.
        """
        return await asyncio.to_thread(self._inner.prep, shared)

    async def exec_async(self, prep_res: object) -> object:
        """Delegate to the wrapped node's :meth:`~pocketflow.Node.exec`.

        :param prep_res: Output from :meth:`prep_async`.
        :return: Exec result for :meth:`post_async`.
        """
        return await asyncio.to_thread(self._inner.exec, prep_res)

    async def post_async(
        self, shared: dict, prep_res: object, exec_res: object
    ) -> object:
        """Delegate to the wrapped node's :meth:`~pocketflow.Node.post`.

        :param shared: PocketFlow shared store.
        :param prep_res: Prep output.
        :param exec_res: Exec output.
        :return: Loopback action or ``None`` for terminal tools.
        """
        return await asyncio.to_thread(self._inner.post, shared, prep_res, exec_res)


class AsyncAgentBot:
    """PocketFlow agent using :class:`~pocketflow.AsyncFlow` and :meth:`AsyncAgentBot.arun`.

    Graph topology matches :class:`~llamabot.bot.agentbot.AgentBot`; only the flow runner
    and node wrappers differ. Visualization and span APIs mirror :class:`AgentBot`.
    """

    def __init__(
        self,
        tools: List[Callable],
        decide_node: Optional[Node] = None,
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4.1",
        max_iterations: Optional[int] = None,
        **completion_kwargs,
    ) -> None:
        _validate_tools(tools)

        all_tools = list(DEFAULT_TOOLS) + tools
        self.tools = all_tools

        if decide_node is None:
            if system_prompt is None:
                from llamabot.components.messages import BaseMessage
                from llamabot.prompt_library.agentbot import decision_bot_system_prompt

                prompt_result = decision_bot_system_prompt()
                if isinstance(prompt_result, BaseMessage):
                    system_prompt = prompt_result.content
                else:
                    system_prompt = prompt_result

            decide_node = DecideNode(
                tools=all_tools,
                system_prompt=system_prompt,
                model_name=model_name,
                max_iterations=max_iterations,
                **completion_kwargs,
            )

        self.decide_node = decide_node
        self.max_iterations = max_iterations

        if hasattr(decide_node, "tools"):
            decide_node.tools = all_tools

        async_decide = AsyncDecideNode(decide_node)
        for tool_node in all_tools:
            tool_name = _tool_routing_name(tool_node)
            async_tool = AsyncFuncNode(tool_node)
            async_decide - tool_name >> async_tool
            if (
                hasattr(tool_node, "loopback_name")
                and tool_node.loopback_name is not None
            ):
                async_tool - tool_node.loopback_name >> async_decide

        self.flow = AsyncFlow(start=async_decide)
        self.shared: dict = dict(memory=[], globals_dict={})
        self._trace_ids: list = []

    async def arun(self, query: str, globals_dict: Optional[dict] = None) -> object:
        """Execute the agent asynchronously via :meth:`AsyncFlow.run_async`.

        :param query: User query string.
        :param globals_dict: Optional globals merged into shared state (see :class:`AgentBot`).
        :return: Value stored in ``shared[\"result\"]`` by a terminal tool, if any.
        """
        from llamabot.recorder import (
            Span,
            get_caller_variable_name,
            get_current_span,
        )

        if not hasattr(self, "shared") or self.shared is None:
            self.shared = dict(memory=[], globals_dict={})

        if "memory" not in self.shared:
            self.shared["memory"] = []
        self.shared["memory"].append(query)

        if globals_dict is not None:
            if "globals_dict" not in self.shared:
                self.shared["globals_dict"] = {}
            self.shared["globals_dict"].update(globals_dict)

        if "globals_dict" not in self.shared:
            self.shared["globals_dict"] = {}

        if "iteration_count" not in self.shared:
            self.shared["iteration_count"] = 0

        operation_name = get_caller_variable_name(self)
        if operation_name is None:
            operation_name = "async_agentbot_arun"

        current_span = get_current_span()
        if current_span:
            agent_span = Span(
                operation_name,
                trace_id=current_span.trace_id,
                parent_span_id=current_span.span_id,
                query=query,
                max_iterations=self.max_iterations,
            )
            if agent_span.trace_id not in self._trace_ids:
                self._trace_ids.append(agent_span.trace_id)
        else:
            new_trace_id = str(uuid.uuid4())
            agent_span = Span(
                operation_name,
                trace_id=new_trace_id,
                query=query,
                max_iterations=self.max_iterations,
            )
            if agent_span.trace_id not in self._trace_ids:
                self._trace_ids.append(agent_span.trace_id)

        with agent_span:
            self.shared["trace_id"] = agent_span.trace_id
            await self.flow.run_async(self.shared)
            result = self.shared.get("result")
            agent_span["result"] = str(result)[:200] if result else None
            agent_span["iterations"] = self.shared.get("iteration_count", 0)
            return result

    def _display_(self):
        """Display the agent's flow graph as a Mermaid diagram (requires marimo)."""
        try:
            import marimo as mo
        except ImportError:
            raise ImportError(
                "marimo is required for AgentBot visualization. "
                "Please install marimo in your environment."
            )
        from llamabot.components.pocketflow import flow_to_mermaid

        return mo.mermaid(flow_to_mermaid(self.flow))

    @property
    def spans(self):
        """Return all spans from all :meth:`arun` calls as a :class:`~llamabot.recorder.SpanList`."""
        from llamabot.recorder import SpanList, get_spans

        if not self._trace_ids:
            return SpanList([])

        all_spans_objects = []
        for trace_id in self._trace_ids:
            spans = get_spans(trace_id=trace_id)
            all_spans_objects.extend(spans)

        return SpanList(all_spans_objects)

    def display_spans(self) -> str:
        """Return HTML for span visualization."""
        return self.spans._repr_html_()

    def _repr_html_(self) -> str:
        """HTML for marimo: flow graph (use ``.spans`` for traces)."""
        try:
            import marimo as mo
        except ImportError:
            return (
                '<div style="padding: 1rem; color: #2E3440;">'
                "marimo is required for AgentBot visualization. "
                "Please install marimo in your environment.</div>"
            )
        from llamabot.components.pocketflow import flow_to_mermaid

        mermaid_diagram = flow_to_mermaid(self.flow)
        mermaid_element = mo.mermaid(mermaid_diagram)
        if hasattr(mermaid_element, "_repr_html_"):
            return mermaid_element._repr_html_()
        return f'<div class="mermaid">{mermaid_diagram}</div>'
