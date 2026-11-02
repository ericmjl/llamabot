"""Tests for library defaults that must stay off retired provider models."""

import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel

from llamabot.bot.agentbot import AgentBot
from llamabot.bot.async_agentbot import AsyncAgentBot
from llamabot.bot.simplebot import SimpleBot, model_supports_token_streaming
from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.chat_memory.memory import ChatMemory, LLMSummarizer
from llamabot.components.chat_memory.selectors import LLMNodeSelector
from llamabot.components.messages import SystemMessage
from llamabot.components.pocketflow.nodes import DecideNode
from llamabot.config import default_language_model

# Model IDs that must not be library defaults. Historical token-window rows
# live in model_tokens.py. The o1 workaround in simplebot.py is behavior for
# models a caller can still name, not a default.
RETIRED_MODEL_IDS = (
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-3.5-turbo-1106",
    "o1-preview",
    "o1-mini",
    "mistral/mistral-medium",
)
O1_ONLY_FILES = {Path("llamabot/bot/simplebot.py")}
HISTORICAL_TOKEN_TABLE = {Path("llamabot/bot/model_tokens.py")}


class TinySchema(BaseModel):
    """Minimal schema used to construct StructuredBot."""

    text: str


def test_default_language_model_fallback_is_gpt_5_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset config falls back to gpt-5.2.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.delenv("DEFAULT_LANGUAGE_MODEL", raising=False)
    monkeypatch.setattr("llamabot.config.load_dotenv", lambda *_args, **_kwargs: False)

    assert default_language_model() == "gpt-5.2"
    assert (
        inspect.signature(SimpleBot.__init__).parameters["model_name"].default
        == "gpt-5.2"
    )


def test_gpt_5_2_does_not_use_o1_workaround() -> None:
    """gpt-5.2 keeps a system prompt, temperature 0, and streaming.

    LiteLLM 1.80.15 accepts temperature for gpt-5.2 when reasoning_effort
    is omitted, which is how SimpleBot calls the model.
    """
    bot = SimpleBot("Stay concise.", model_name="gpt-5.2", stream_target="none")

    assert isinstance(bot.system_prompt, SystemMessage)
    assert bot.system_prompt.content == "Stay concise."
    assert bot.temperature == 0.0
    assert bot.stream_target == "none"
    assert model_supports_token_streaming("gpt-5.2") is True


def test_litellm_passes_gpt_5_2_temperature() -> None:
    """Pinned LiteLLM must not reject SimpleBot's default temperature."""
    from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config

    mapped = OpenAIGPT5Config().map_openai_params(
        non_default_params={"temperature": 0.0},
        optional_params={},
        model="gpt-5.2",
        drop_params=False,
    )

    assert mapped["temperature"] == 0.0


def test_structured_bot_accepts_gpt_5_2() -> None:
    """gpt-5.2 supports the response schema StructuredBot requires."""
    bot = StructuredBot(
        "Return the answer.",
        TinySchema,
        model_name="gpt-5.2",
        stream_target="none",
    )

    assert bot.model_name == "gpt-5.2"
    assert bot.json_mode is True
    assert bot.temperature == 0.0


def test_agent_defaults_are_gpt_5_2() -> None:
    """AgentBot, AsyncAgentBot, and DecideNode default to gpt-5.2."""
    assert (
        inspect.signature(AgentBot.__init__).parameters["model_name"].default
        == "gpt-5.2"
    )
    assert (
        inspect.signature(AsyncAgentBot.__init__).parameters["model_name"].default
        == "gpt-5.2"
    )
    assert (
        inspect.signature(DecideNode.__init__).parameters["model_name"].default
        == "gpt-5.2"
    )

    bot = AgentBot(tools=[])
    assert bot.decide_node.model_name == "gpt-5.2"

    async_bot = AsyncAgentBot(tools=[])
    assert async_bot.decide_node.model_name == "gpt-5.2"


def test_chat_memory_defaults_are_gpt_5_2() -> None:
    """Threaded memory selectors and summarizers default to gpt-5.2."""
    assert (
        inspect.signature(LLMNodeSelector.__init__).parameters["model"].default
        == "gpt-5.2"
    )
    assert (
        inspect.signature(LLMSummarizer.__init__).parameters["model"].default
        == "gpt-5.2"
    )
    assert (
        inspect.signature(ChatMemory.threaded).parameters["model"].default == "gpt-5.2"
    )


def test_cli_and_helper_defaults_are_gpt_5_2() -> None:
    """CLI model options and the git prompt helper default to gpt-5.2."""
    from llamabot.cli.doc import chat as doc_chat
    from llamabot.cli.docs import (
        docwriter_bot,
        ood_checker_bot,
        refine_bot,
        write,
    )
    from llamabot.cli.notebook import explain, explainer_bot
    from llamabot.cli.repo import chat as repo_chat
    from llamabot.prompt_library.git import commitbot

    defaults = {
        doc_chat: "model_name",
        repo_chat: "model_name",
        explain: "model_name",
        explainer_bot: "model_name",
        ood_checker_bot: "model_name",
        docwriter_bot: "model_name",
        refine_bot: "model_name",
        write: "ood_checker_model_name",
    }
    for function, parameter_name in defaults.items():
        parameter = inspect.signature(function).parameters[parameter_name]
        value = getattr(parameter.default, "default", parameter.default)
        assert value == "gpt-5.2", function.__name__

    write_signature = inspect.signature(write)
    for parameter_name in ("docwriter_model_name", "refiner_model_name"):
        assert write_signature.parameters[parameter_name].default == "gpt-5.2"

    assert commitbot().model_name == "gpt-5.2"


def test_docqa_bots_use_openai_gpt_5_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """DocQABot calls gpt-5.2 with the OpenAI key for both bots.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    from llamabot.bot.qabot import DocQABot

    def fake_docstore(**kwargs: object) -> object:
        """Skip LanceDB setup while constructing DocQABot.

        :param kwargs: Ignored store constructor arguments.
        :return: A placeholder store.
        """
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr("llamabot.bot.qabot.LanceDBDocStore", fake_docstore)

    bot = DocQABot("collection")

    assert bot.response_bot.model_name == "gpt-5.2"
    assert bot.response_bot.api_key == "test-openai-key"
    assert bot.jeopardy_bot.model_name == "gpt-5.2"
    assert bot.jeopardy_bot.api_key == "test-openai-key"


def test_library_source_does_not_default_to_retired_models() -> None:
    """Retired model IDs stay out of llamabot source except known exceptions."""
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in (root / "llamabot").rglob("*.py"):
        relative = path.relative_to(root)
        text = path.read_text()
        for model_id in RETIRED_MODEL_IDS:
            if model_id not in text:
                continue
            if (
                relative in HISTORICAL_TOKEN_TABLE
                and model_id == "mistral/mistral-medium"
            ):
                continue
            if relative in O1_ONLY_FILES and model_id in {"o1-preview", "o1-mini"}:
                continue
            offenders.append(f"{relative}: {model_id}")

    assert offenders == []
