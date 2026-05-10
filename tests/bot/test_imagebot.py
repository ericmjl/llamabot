"""Test the ImageBot class."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llamabot import ImageBot, SimpleBot
from llamabot.components.messages import AIMessage


def image_response(
    url: str | None = "http://image.url", b64_json: str | None = None
) -> SimpleNamespace:
    """Create a minimal LiteLLM image generation response for tests.

    :param url: The generated image URL to include in the response.
    :param b64_json: The base64-encoded generated image data to include.
    :return: A response object shaped like LiteLLM's image response.
    """
    return SimpleNamespace(data=[SimpleNamespace(url=url, b64_json=b64_json)])


def test_initialization_defaults():
    """Test the initialization of the ImageBot class with default parameters."""
    bot = ImageBot()
    assert bot.model == "dall-e-3"
    assert bot.size == "1024x1024"
    assert bot.quality is None
    assert bot.n == 1
    assert bot.response_format == "url"
    assert bot.api_key is None
    assert bot.api_base is None
    assert bot.api_version is None
    assert bot.timeout == 600
    assert bot.image_generation_kwargs == {}


def test_initialization_custom():
    """Test the initialization of the ImageBot class with custom parameters."""
    bot = ImageBot(
        model="custom-model",
        size="800x800",
        quality="standard",
        n=2,
        response_format="b64_json",
        api_key="test-api-key",
        api_base="https://example.com",
        api_version="2024-02-01",
        timeout=30,
        style="vivid",
    )
    assert bot.model == "custom-model"
    assert bot.size == "800x800"
    assert bot.quality == "standard"
    assert bot.n == 2
    assert bot.response_format == "b64_json"
    assert bot.api_key == "test-api-key"
    assert bot.api_base == "https://example.com"
    assert bot.api_version == "2024-02-01"
    assert bot.timeout == 30
    assert bot.image_generation_kwargs == {"style": "vivid"}


def test_call_uses_litellm_image_generation(mocker):
    """Test that ImageBot calls LiteLLM's image generation API."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    image_generation = mocker.patch(
        "litellm.image_generation",
        return_value=image_response(),
    )

    bot = ImageBot()
    result = bot("test prompt", return_url=True)

    assert result == "http://image.url"
    image_generation.assert_called_once_with(
        model="dall-e-3",
        prompt="test prompt",
        size="1024x1024",
        n=1,
        response_format="url",
        timeout=600,
    )


def test_call_forwards_litellm_provider_kwargs(mocker):
    """Test that provider-specific LiteLLM kwargs are forwarded."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    image_generation = mocker.patch(
        "litellm.image_generation",
        return_value=image_response(),
    )

    bot = ImageBot(
        model="openrouter/google/gemini-2.5-flash-image",
        api_key="test-api-key",
        api_base="https://example.com",
        api_version="2026-01-01",
        timeout=45,
        quality="high",
        response_format="url",
        user="user-123",
    )
    result = bot("test prompt", return_url=True)

    assert result == "http://image.url"
    image_generation.assert_called_once_with(
        model="openrouter/google/gemini-2.5-flash-image",
        prompt="test prompt",
        size="1024x1024",
        quality="high",
        n=1,
        response_format="url",
        timeout=45,
        api_key="test-api-key",
        api_base="https://example.com",
        api_version="2026-01-01",
        user="user-123",
    )


def test_call_in_jupyter():
    """Test the call method when running in a Jupyter notebook."""
    with patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=True):
        with patch("IPython.display.Image") as mock_image:
            with patch("IPython.display.display") as mock_display:
                # Setup
                bot = ImageBot()
                test_url = "http://image.url"

                # Action
                with patch(
                    "litellm.image_generation",
                    return_value=image_response(url=test_url),
                ):
                    result = bot("test prompt")

                # Assert
                assert result == test_url
                mock_image.assert_called_once_with(url=test_url)
                mock_display.assert_called_once_with(mock_image.return_value)


def test_call_outside_jupyter(mocker, tmp_path):
    """Test the call method when not running in a Jupyter notebook.

    This test tests that the call method returns the path to the generated image
    when not running in a Jupyter notebook.

    :param mocker: The pytest-mock fixture.
    :param tmp_path: The pytest tmp_path fixture.
    """
    # Mock the is_running_in_jupyter method
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)

    # Instantiate ImageBot
    bot = ImageBot()

    # Mock LiteLLM's image generation API to return the desired URL
    mocker.patch("litellm.image_generation", return_value=image_response())

    # Mock httpx.get to return a mock response with content
    mock_get_response = mocker.MagicMock()
    mock_get_response.content = b"image_data"
    mock_get = mocker.patch("httpx.get", return_value=mock_get_response)

    # Mock the SimpleBot's __call__ method
    mocker.patch.object(
        SimpleBot, "__call__", return_value=mocker.MagicMock(message="test_prompt")
    )

    # Call the method and perform the assertion
    result = bot("test prompt", save_path=tmp_path / "test_prompt.jpg")
    assert result == tmp_path / "test_prompt.jpg"
    mock_get.assert_called_with("http://image.url")
    mock_get_response.raise_for_status.assert_called_once_with()


def test_call_outside_jupyter_no_save_path(mocker, tmp_path):
    """Test the call method when not running in a Jupyter notebook
    and no save path is provided.

    This test checks that the call method
    saves the image to a generated filename based on the prompt
    when not running in a Jupyter notebook and no save path is provided.

    :param mocker: The pytest-mock fixture.
    """
    # Mock the is_running_in_jupyter method
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)

    # Instantiate ImageBot
    bot = ImageBot()

    # Mock LiteLLM's image generation API to return the desired URL
    mocker.patch("litellm.image_generation", return_value=image_response())

    # Mock httpx.get to return a mock response with content
    mock_get_response = mocker.MagicMock()
    mock_get_response.content = b"image_data"
    mock_get = mocker.patch("httpx.get", return_value=mock_get_response)

    # Mock the filename_bot method to return a generated filename
    generated_filename = str(tmp_path / "generated_filename")
    mocker.patch(
        "llamabot.bot.imagebot.filename_bot",
        return_value=AIMessage(content=generated_filename),
    )

    # Call the method and perform the assertion
    result = bot("test prompt")
    assert result == Path(f"{generated_filename}.jpg")
    mock_get.assert_called_with("http://image.url")
    mock_get_response.raise_for_status.assert_called_once_with()

    # Check if the file was saved correctly
    with open(result, "rb") as file:
        saved_content = file.read()
    assert saved_content == b"image_data"


def test_call_outside_jupyter_saves_b64_json_without_downloading(mocker, tmp_path):
    """Test that base64 image responses are saved without an image download."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    encoded_image = base64.b64encode(b"image_data").decode("utf-8")
    mocker.patch(
        "litellm.image_generation",
        return_value=image_response(url=None, b64_json=encoded_image),
    )
    mock_get = mocker.patch("httpx.get")

    bot = ImageBot(response_format="b64_json")
    result = bot("test prompt", save_path=tmp_path / "test_prompt.png")

    assert result == tmp_path / "test_prompt.png"
    assert result.read_bytes() == b"image_data"
    mock_get.assert_not_called()


def test_call_raises_when_litellm_returns_no_images(mocker):
    """Test that ImageBot raises when LiteLLM returns no generated images."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    mocker.patch(
        "litellm.image_generation",
        return_value=SimpleNamespace(data=[]),
    )

    bot = ImageBot()

    with pytest.raises(ValueError, match="No images found in response"):
        bot("test prompt", return_url=True)
