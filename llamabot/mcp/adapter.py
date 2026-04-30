"""Adapt MCP tool definitions into LlamaBot ``@tool``-compatible callables."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List

from loguru import logger

from llamabot.components.pocketflow import nodeify as nodeify_decorator
from llamabot.recorder import span as span_decorator

from llamabot.mcp.session import PersistentMCPClientSession
from llamabot.mcp.specs import MCPIntegrationOptions, MCPStartupMode


def sanitize_python_identifier(name: str) -> str:
    """Turn an MCP tool name into a valid Python identifier fragment.

    :param name: Raw tool or parameter name from the MCP server.
    :return: Safe identifier fragment.
    """
    import re

    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if s and s[0].isdigit():
        s = f"_{s}"
    return s or "tool"


def prefixed_tool_name(server_name: str, tool_name: str, sep: str) -> str:
    """Build a namespaced tool name used for allow/deny lists.

    :param server_name: Logical MCP server name.
    :param tool_name: Remote tool name.
    :param sep: Namespace separator.
    :return: Prefixed name ``server{sep}tool``.
    """
    a = sanitize_python_identifier(server_name)
    b = sanitize_python_identifier(tool_name)
    return f"{a}{sep}{b}"


def normalize_openai_parameters(input_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize MCP JSON Schema input into OpenAI function ``parameters`` shape.

    :param input_schema: MCP tool ``inputSchema`` (may be ``None``).
    :return: Dict with ``type``, ``properties``, and ``required`` (wire keys).
    """
    if not input_schema:
        return {"type": "object", "properties": {}, "required": []}

    if input_schema.get("type") == "object":
        return {
            "type": "object",
            "properties": dict(input_schema.get("properties") or {}),
            "required": list(input_schema.get("required") or []),
        }

    return {"type": "object", "properties": {}, "required": []}


def remap_schema_properties_for_python(
    properties: Dict[str, Any],
    required_wire: List[str],
) -> tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """Map JSON Schema property keys to valid Python parameter names.

    :param properties: Original ``properties`` object from MCP.
    :param required_wire: Required field names using wire keys.
    :return: Tuple of new properties dict, wire-key map ``py_name -> wire_name``, and
        required list using Python names.
    """
    new_props: Dict[str, Any] = {}
    wire_map: Dict[str, str] = {}
    used_py: set[str] = set()

    for wire_name, schema_fragment in properties.items():
        base = sanitize_python_identifier(wire_name)
        py_name = base
        n = 2
        while py_name in used_py:
            py_name = f"{base}_{n}"
            n += 1
        used_py.add(py_name)
        wire_map[py_name] = wire_name
        new_props[py_name] = schema_fragment

    required_set_wire = set(required_wire)
    new_required: List[str] = []
    for py_name, wname in wire_map.items():
        if wname in required_set_wire:
            new_required.append(py_name)

    return new_props, wire_map, new_required


def format_call_tool_result(result: Any) -> Any:
    """Serialize a FastMCP tool call result for the model.

    :param result: Parsed tool call result from FastMCP.
    :return: JSON-friendly Python value (typically ``dict`` or ``str``).
    """
    if result is None:
        return None

    if getattr(result, "is_error", False):
        payload: Dict[str, Any] = {"error": True}
        if getattr(result, "data", None) is not None:
            payload["data"] = result.data
        if getattr(result, "structured_content", None):
            payload["structured_content"] = result.structured_content
        blocks = getattr(result, "content", None) or []
        texts: List[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text is not None:
                texts.append(text)
        if texts:
            payload["text"] = "\n".join(texts)
        return payload

    data = getattr(result, "data", None)
    if data is not None:
        return data
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured

    blocks = getattr(result, "content", None) or []
    texts = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    if texts:
        return "\n".join(texts)
    try:
        return result.model_dump()
    except Exception:
        return str(result)


def build_function_dict_for_mcp_tool(
    python_name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build the nested ``function`` object attached to ``json_schema``.

    :param python_name: Exposed Python function name for routing.
    :param description: Combined human-readable description.
    :param parameters: OpenAI-style parameters dict (Python param keys).
    :return: Function dictionary for tools JSON schema.
    """
    props = parameters.get("properties") or {}
    required = list(parameters.get("required") or [])
    return {
        "name": python_name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
        },
    }


def apply_llamabot_tool_wrappers(
    fn: Callable[..., Any],
    function_dict: dict[str, Any],
    *,
    loopback_name: str | None = "decide",
    exclude_args: List[str] | None = None,
) -> Callable[..., Any]:
    """Attach ``json_schema``, spans, and PocketFlow node wiring like ``@tool``.

    :param fn: Inner Python callable implementing the MCP proxy.
    :param function_dict: OpenAI function metadata dict.
    :param loopback_name: Routing loopback target (default ``decide``).
    :param exclude_args: Span exclusions for sensitive argument names.
    :return: Fully decorated tool callable.
    """
    wrapped = fn
    wrapped.json_schema = {
        "type": "function",
        "function": function_dict,
    }
    wrapped = span_decorator(
        operation_name=function_dict.get("name"),
        exclude_args=exclude_args or [],
    )(wrapped)
    wrapped = nodeify_decorator(loopback_name=loopback_name)(wrapped)
    return wrapped


def make_mcp_proxy_callable(
    session: PersistentMCPClientSession,
    *,
    python_name: str,
    remote_tool_name: str,
    description: str,
    parameters_wire: dict[str, Any],
    call_timeout: float | None,
) -> Callable[..., Any]:
    """Create a sync callable that forwards to ``session.client.call_tool``.

    :param session: Active MCP session holder.
    :param python_name: Sanitized Python symbol for routing.
    :param remote_tool_name: Original MCP tool name on the server.
    :param description: Docstring / schema description text.
    :param parameters_wire: Normalized parameters using **wire** JSON keys.
    :param call_timeout: Optional per-call timeout in seconds.
    :return: Callable with ``__signature__`` for kwargs routing (Python param names).
    """
    props_wire = parameters_wire.get("properties") or {}
    required_wire = list(parameters_wire.get("required") or [])

    props_py, wire_map, required_py = remap_schema_properties_for_python(
        props_wire,
        required_wire,
    )

    parameters_py = {
        "type": "object",
        "properties": props_py,
        "required": required_py,
    }

    params: list[inspect.Parameter] = []
    for pname_py, pschema in props_py.items():
        default: Any = inspect.Parameter.empty
        if pname_py not in set(required_py):
            if isinstance(pschema, dict) and "default" in pschema:
                default = pschema["default"]
            else:
                default = None
        params.append(
            inspect.Parameter(
                pname_py,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=Any,
            )
        )

    sig = inspect.Signature(params)

    def impl(**kwargs: Any) -> Any:
        """Call the MCP ``call_tool`` endpoint using wire-schema argument keys."""
        bound = sig.bind_partial(**kwargs)
        bound.apply_defaults()
        arguments: Dict[str, Any] = {}
        for py_key, value in bound.arguments.items():
            wire_key = wire_map.get(py_key, py_key)
            arguments[wire_key] = value

        async def _call() -> Any:
            """Await FastMCP ``call_tool`` on the persistent session client."""
            return await session.client.call_tool(remote_tool_name, arguments)

        raw = session.run_coroutine(_call(), timeout=call_timeout)
        return format_call_tool_result(raw)

    impl.__name__ = python_name
    impl.__doc__ = description or f"MCP tool `{remote_tool_name}`."
    impl.__signature__ = sig  # type: ignore[attr-defined]
    impl.__annotations__ = {p.name: Any for p in params}

    function_dict = build_function_dict_for_mcp_tool(
        python_name, description or "", parameters_py
    )
    return apply_llamabot_tool_wrappers(impl, function_dict)


def mcp_tools_as_llamabot_tools(
    session: PersistentMCPClientSession,
    server_name: str,
    remote_tools: List[Any],
    *,
    namespace_sep: str,
    options: MCPIntegrationOptions,
    call_timeout: float | None,
) -> List[Callable[..., Any]]:
    """Convert MCP ``Tool`` metadata objects into LlamaBot tools.

    :param session: Connected MCP session for invocation.
    :param server_name: Logical server name for namespacing.
    :param remote_tools: Output of ``await client.list_tools()``.
    :param namespace_sep: Separator between server and tool names.
    :param options: Filtering options (allow/deny lists).
    :param call_timeout: Optional per-call timeout.
    :return: Decorated tool callables ready for :class:`~llamabot.bot.agentbot.AgentBot`.
    """
    out: List[Callable[..., Any]] = []
    allow = options.allow_tools
    deny = options.deny_tools or []

    for rt in remote_tools:
        remote_name: str = ""
        desc = ""
        input_schema: dict[str, Any] | None = None

        if isinstance(rt, dict):
            remote_name = str(rt.get("name", ""))
            desc = str(rt.get("description", "") or "")
            input_schema = rt.get("inputSchema") or rt.get("input_schema")
            if input_schema is not None and not isinstance(input_schema, dict):
                input_schema = None
        else:
            remote_name = str(getattr(rt, "name", "") or "")
            desc = str(getattr(rt, "description", "") or "")
            raw_schema = getattr(rt, "inputSchema", None)
            if raw_schema is None:
                input_schema = None
            elif isinstance(raw_schema, dict):
                input_schema = raw_schema
            elif hasattr(raw_schema, "model_dump"):
                dumped = raw_schema.model_dump()
                input_schema = dumped if isinstance(dumped, dict) else None
            else:
                input_schema = None

        prefixed = prefixed_tool_name(server_name, remote_name, namespace_sep)
        if allow is not None and prefixed not in allow:
            continue
        if prefixed in deny:
            continue

        py_name = prefixed
        normalized_wire = normalize_openai_parameters(input_schema)

        header = f"[MCP:{server_name}] {remote_name}"
        description = f"{header}\n\n{desc}".strip()

        try:
            fn = make_mcp_proxy_callable(
                session,
                python_name=py_name,
                remote_tool_name=remote_name,
                description=description,
                parameters_wire=normalized_wire,
                call_timeout=call_timeout,
            )
            out.append(fn)
        except Exception:
            logger.exception(
                "Skipping MCP tool {!r} from server {!r}",
                remote_name,
                server_name,
            )
            if options.startup_mode == MCPStartupMode.STRICT:
                raise

    return out
