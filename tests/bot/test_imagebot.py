"""Test the ImageBot class."""

import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import pytest

from llamabot import ImageBot
from llamabot.bot.imagebot import ImageReference


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
        style="cinematic photography, golden-hour lighting",
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
    assert bot.style == "cinematic photography, golden-hour lighting"
    assert bot.image_generation_kwargs == {}


def test_call_returns_image_reference(mocker):
    """Test that ImageBot.__call__ returns an ImageReference."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    mocker.patch(
        "litellm.image_generation",
        return_value=image_response(),
    )

    bot = ImageBot()
    result = bot("test prompt")

    assert isinstance(result, ImageReference)
    assert result == "http://image.url"


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
    result = bot("test prompt")

    assert isinstance(result, ImageReference)
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
                bot = ImageBot()
                test_url = "http://image.url"

                with patch(
                    "litellm.image_generation",
                    return_value=image_response(url=test_url),
                ):
                    result = bot("test prompt")

                assert isinstance(result, ImageReference)
                assert result == test_url
                mock_image.assert_called_once_with(url=test_url)
                mock_display.assert_called_once_with(mock_image.return_value)


def test_call_prepends_style_to_prompt(mocker):
    """Test that the style string is prepended to the prompt."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    image_generation = mocker.patch(
        "litellm.image_generation",
        return_value=image_response(),
    )

    bot = ImageBot(style="cinematic photography, 35mm")
    result = bot("a red apple on a table")

    assert isinstance(result, ImageReference)
    image_generation.assert_called_once_with(
        model="dall-e-3",
        prompt="cinematic photography, 35mm, a red apple on a table",
        size="1024x1024",
        n=1,
        response_format="url",
        timeout=600,
    )


def test_call_without_style_passes_prompt_unchanged(mocker):
    """Test that prompts are passed through unchanged when no style is set."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    image_generation = mocker.patch(
        "litellm.image_generation",
        return_value=image_response(),
    )

    bot = ImageBot()
    bot("test prompt")

    image_generation.assert_called_once_with(
        model="dall-e-3",
        prompt="test prompt",
        size="1024x1024",
        n=1,
        response_format="url",
        timeout=600,
    )


def test_call_raises_when_litellm_returns_no_images(mocker):
    """Test that ImageBot raises when LiteLLM returns no generated images."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    mocker.patch(
        "litellm.image_generation",
        return_value=SimpleNamespace(data=[]),
    )

    bot = ImageBot()

    with pytest.raises(ValueError, match="No images found in response"):
        bot("test prompt")


def test_call_routes_ollama_models_to_native_api(mocker):
    """Test that Ollama image models are routed to /api/generate."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    litellm_image_generation = mocker.patch("litellm.image_generation")
    encoded_image = base64.b64encode(b"ollama_image_data").decode("utf-8")
    mock_post_response = mocker.MagicMock()
    mock_post_response.json.return_value = {"image": encoded_image}
    mock_post = mocker.patch("httpx.post", return_value=mock_post_response)

    bot = ImageBot(
        model="ollama/x/flux2-klein:4b-bf16",
        api_base="http://localhost:11434",
        response_format="b64_json",
        style="cinematic photography, 35mm",
    )
    result = bot("test prompt")

    assert isinstance(result, ImageReference)
    assert result.is_data_uri()
    assert "data:image/png;base64" in result
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": "x/flux2-klein:4b-bf16",
            "prompt": "cinematic photography, 35mm, test prompt",
            "stream": False,
            "width": 1024,
            "height": 1024,
        },
        timeout=600,
    )
    mock_post_response.raise_for_status.assert_called_once_with()
    litellm_image_generation.assert_not_called()


def test_ollama_route_raises_when_response_has_no_image(mocker):
    """Test that Ollama routing raises when /api/generate has no image field."""
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)
    mock_post_response = mocker.MagicMock()
    mock_post_response.json.return_value = {"response": "done", "done": True}
    mocker.patch("httpx.post", return_value=mock_post_response)

    bot = ImageBot(model="ollama/x/flux2-klein:4b-bf16")

    with pytest.raises(ValueError, match="No image field returned by Ollama"):
        bot("test prompt")


def test_image_reference_html_for_url():
    """Test that image references provide a renderable HTML representation."""
    reference = ImageReference("http://example.com/image.png")

    assert str(reference) == "http://example.com/image.png"
    assert "img" in reference._repr_html_()
    assert 'src="http://example.com/image.png"' in reference._repr_html_()


def test_image_reference_html_for_data_uri():
    """Test that base64 image references can render via HTML representation."""
    encoded_image = base64.b64encode(b"image_data").decode("utf-8")
    reference = ImageReference(f"data:image/png;base64,{encoded_image}")

    assert reference.is_data_uri()
    assert "data:image/png;base64" in reference._repr_html_()


def test_image_reference_to_bytes_from_data_uri():
    """Test that to_bytes decodes base64 from a data URI."""
    raw = b"image_data"
    encoded_image = base64.b64encode(raw).decode("utf-8")
    reference = ImageReference(f"data:image/png;base64,{encoded_image}")

    assert reference.to_bytes() == raw


def test_image_reference_to_bytes_from_url(mocker):
    """Test that to_bytes downloads from a URL."""
    mock_response = mocker.MagicMock()
    mock_response.content = b"downloaded_image"
    mocker.patch("httpx.get", return_value=mock_response)

    reference = ImageReference("http://example.com/image.png")
    assert reference.to_bytes() == b"downloaded_image"


def test_image_reference_save_writes_file(tmp_path):
    """Test that save writes the image bytes to disk."""
    img = Image.new("RGB", (10, 10), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    raw = buffer.getvalue()
    encoded_image = base64.b64encode(raw).decode("utf-8")
    reference = ImageReference(f"data:image/png;base64,{encoded_image}")

    dest = tmp_path / "output.png"
    result = reference.save(dest)

    assert result == dest
    assert dest.read_bytes() == raw


def test_image_reference_save_returns_path(tmp_path):
    """Test that save returns a Path object for chaining."""
    img = Image.new("RGB", (10, 10), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    raw = buffer.getvalue()
    encoded_image = base64.b64encode(raw).decode("utf-8")
    reference = ImageReference(f"data:image/png;base64,{encoded_image}")

    dest = tmp_path / "chain.png"
    assert reference.save(dest) == dest
