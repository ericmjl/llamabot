"""ImageBot module for generating images."""

import base64
import binascii
import html
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import httpx

from llamabot.components.messages import AIMessage


class ImageReference(str):
    """Displayable and saveable reference to a generated image.

    Subclasses ``str`` so the value is the URL or data URI.
    Supports notebook rendering via ``_repr_html_`` and
    disk persistence via ``save``.

    :param value: URL or data URI for the generated image.
    :param mime_type: MIME type used when constructing data URIs.
    """

    def __new__(cls, value: str, mime_type: str = "image/png") -> "ImageReference":
        reference = super().__new__(cls, value)
        reference.mime_type = mime_type
        return reference

    def is_data_uri(self) -> bool:
        """Return whether this image reference is a data URI.

        :return: True if the reference starts with ``data:``, else False.
        """
        return self.startswith("data:")

    def to_bytes(self) -> bytes:
        """Return the raw image bytes.

        Decodes from the base64 payload in a data URI, or downloads
        from the URL if the reference is a remote link.

        :return: Raw image bytes.
        :raises ValueError: If the reference is neither a data URI nor a URL.
        """
        if self.is_data_uri():
            _, encoded = str(self).split(",", 1)
            try:
                return base64.b64decode(encoded, validate=True)
            except binascii.Error as error:
                raise ValueError("Invalid base64 image data.") from error

        if self.startswith(("http://", "https://")):
            response = httpx.get(str(self))
            response.raise_for_status()
            return response.content

        raise ValueError("Cannot extract bytes: reference is not a data URI or URL.")

    def save(self, path: Path) -> Path:
        """Save the image to disk.

        :param path: Destination file path.
        :return: The ``path`` argument, for chaining.
        """
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path

    def to_html(self, alt: str = "Generated image") -> str:
        """Return an HTML ``img`` element for this image reference.

        :param alt: Alternate text for the rendered image.
        :return: HTML string with an ``img`` tag.
        """
        escaped_src = html.escape(str(self), quote=True)
        escaped_alt = html.escape(alt, quote=True)
        return (
            f'<img src="{escaped_src}" alt="{escaped_alt}" '
            'style="max-width: 100%; height: auto;" />'
        )

    def _repr_html_(self) -> str:
        """Return HTML representation used by notebook renderers.

        :return: HTML representation of this image reference.
        """
        return self.to_html()


class ImageBot:
    """ImageBot class for generating images.

    :param model: The model to use. Defaults to "dall-e-3".
    :param size: The size of the image to generate. Defaults to "1024x1024".
    :param quality: Optional quality setting for models that support it.
    :param n: The number of images to generate. Defaults to 1.
    :param response_format: The LiteLLM image response format to request.
        Defaults to "url".
    :param api_key: Optional API key to pass through to LiteLLM.
    :param api_base: Optional API base URL to pass through to LiteLLM.
    :param api_version: Optional API version to pass through to LiteLLM.
    :param timeout: Maximum time, in seconds, to wait for image generation.
    :param style: Optional visual-style string prepended to every prompt,
        acting like a system prompt for image generation. For example
        ``"cinematic photography, golden-hour lighting, 35mm"``.
    :param image_generation_kwargs: Additional provider-specific keyword arguments
        to pass through to ``litellm.image_generation``.
    """

    def __init__(
        self,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: Optional[str] = None,
        n: int = 1,
        response_format: str = "url",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: int = 600,
        style: Optional[str] = None,
        **image_generation_kwargs: Any,
    ):
        self.model = model
        self.size = size
        self.quality = quality
        self.n = n
        self.response_format = response_format
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.timeout = timeout
        self.style = style
        self.image_generation_kwargs = image_generation_kwargs

    def __call__(self, prompt: str) -> ImageReference:
        """Generate an image from a prompt and return a displayable reference.

        :param prompt: The prompt to generate an image from.
        :return: An ``ImageReference`` that can be displayed in notebooks
            or saved to disk via ``.save(path)``.
        :raises ValueError: If no generated image data is found in the response.
        """
        full_prompt = f"{self.style}, {prompt}" if self.style else prompt
        response = generate_image_response(self, full_prompt)
        image = first_generated_image(response)

        if is_running_in_jupyter():
            try:
                from IPython.display import Image, display

                if getattr(image, "url", None):
                    display(Image(url=image.url))
                else:
                    display(Image(data=generated_image_bytes(image)))
            except ImportError:
                pass

        return generated_image_reference(image)


def image_generation_kwargs_for_bot(bot: ImageBot, prompt: str) -> dict[str, Any]:
    """Build keyword arguments for LiteLLM image generation.

    :param bot: The ImageBot providing generation settings.
    :param prompt: The prompt to generate an image from.
    :return: Keyword arguments suitable for ``litellm.image_generation``.
    """
    generation_kwargs: dict[str, Any] = {
        "model": bot.model,
        "prompt": prompt,
        "n": bot.n,
        "timeout": bot.timeout,
    }
    optional_kwargs = {
        "size": bot.size,
        "quality": bot.quality,
        "response_format": bot.response_format,
        "api_key": bot.api_key,
        "api_base": bot.api_base,
        "api_version": bot.api_version,
    }
    generation_kwargs.update(
        {key: value for key, value in optional_kwargs.items() if value is not None}
    )
    generation_kwargs.update(bot.image_generation_kwargs)
    return generation_kwargs


def generate_image_response(bot: ImageBot, prompt: str) -> Any:
    """Generate an image response through LiteLLM.

    :param bot: The ImageBot providing generation settings.
    :param prompt: The prompt to generate an image from.
    :return: LiteLLM image generation response.
    """
    if model_uses_ollama_image_api(bot.model):
        return generate_ollama_image_response(bot, prompt)

    from litellm import image_generation

    return image_generation(**image_generation_kwargs_for_bot(bot, prompt))


def model_uses_ollama_image_api(model_name: str) -> bool:
    """Return whether a model should be routed to Ollama's native image API.

    :param model_name: Model identifier configured on ImageBot.
    :return: True when the model uses the ``ollama/`` prefix.
    """
    return model_name.startswith("ollama/")


def parsed_image_size(size: str) -> tuple[int, int] | None:
    """Parse ``<width>x<height>`` size text into integers.

    :param size: Image size string (for example ``1024x1024``).
    :return: ``(width, height)`` when parsing succeeds; otherwise ``None``.
    """
    parts = size.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        return None
    if not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return int(parts[0]), int(parts[1])


def ollama_image_payload(bot: ImageBot, prompt: str) -> dict[str, Any]:
    """Build the request payload for Ollama image generation.

    :param bot: The ImageBot providing generation settings.
    :param prompt: The prompt to generate an image from.
    :return: JSON payload for ``POST /api/generate`` on Ollama.
    """
    payload: dict[str, Any] = {
        "model": bot.model.removeprefix("ollama/"),
        "prompt": prompt,
        "stream": False,
    }
    size = parsed_image_size(bot.size)
    if size is not None:
        payload["width"], payload["height"] = size
    payload.update(bot.image_generation_kwargs)
    return payload


def generate_ollama_image_response(bot: ImageBot, prompt: str) -> SimpleNamespace:
    """Generate an image through Ollama's native ``/api/generate`` endpoint.

    :param bot: The ImageBot providing generation settings.
    :param prompt: The prompt to generate an image from.
    :return: A response object shaped like LiteLLM image generation responses.
    :raises ValueError: If the Ollama response does not include image data.
    """
    api_base = bot.api_base or "http://localhost:11434"
    response = httpx.post(
        f"{api_base.rstrip('/')}/api/generate",
        json=ollama_image_payload(bot, prompt),
        timeout=bot.timeout,
    )
    response.raise_for_status()
    response_json = response.json()
    b64_image = response_json.get("image")
    if not b64_image:
        raise ValueError(
            "No image field returned by Ollama /api/generate response. "
            f"Response keys: {sorted(response_json.keys())}"
        )
    return SimpleNamespace(
        data=[SimpleNamespace(url=None, b64_json=b64_image)],
    )


def first_generated_image(response: Any) -> Any:
    """Return the first image object from a LiteLLM image response.

    :param response: LiteLLM image generation response.
    :return: The first generated image object.
    :raises ValueError: If the response does not contain generated images.
    """
    images = getattr(response, "data", None)
    if not images:
        raise ValueError("No images found in response! Please try again.")
    return images[0]


def generated_image_reference(image: Any) -> ImageReference:
    """Return a displayable reference for a generated image.

    :param image: A LiteLLM image object.
    :return: The image URL, or a base64 data URI when only base64 data is present.
    :raises ValueError: If neither URL nor base64 image data is present.
    """
    image_url = getattr(image, "url", None)
    if image_url:
        return ImageReference(image_url)

    b64_json = getattr(image, "b64_json", None)
    if b64_json:
        return ImageReference(f"data:image/png;base64,{b64_json}")

    raise ValueError("No image URL or b64_json found in response! Please try again.")


def generated_image_bytes(image: Any) -> bytes:
    """Return image bytes from a generated image object.

    :param image: A LiteLLM image object.
    :return: Generated image bytes decoded from base64 data or downloaded from URL.
    :raises ValueError: If neither URL nor base64 image data is present.
    """
    b64_json = getattr(image, "b64_json", None)
    if b64_json:
        try:
            return base64.b64decode(b64_json, validate=True)
        except binascii.Error as error:
            raise ValueError("Invalid b64_json image data in response.") from error

    image_url = getattr(image, "url", None)
    if image_url:
        response = httpx.get(image_url)
        response.raise_for_status()
        return response.content

    raise ValueError("No image URL or b64_json found in response! Please try again.")


def is_running_in_jupyter() -> bool:
    """Check if running in a Jupyter notebook.

    :return: True if running in a Jupyter notebook, otherwise False.
    """
    try:
        get_ipython()
        return True
    except NameError:
        return False


def filename_bot(image_prompt: str) -> AIMessage:
    """Generate a filename from an image prompt.

    :param image_prompt: The image prompt to generate a filename from.
    :return: The filename generated from the image prompt within an AIMessage object.
    """
    from llamabot import SimpleBot

    bot = SimpleBot(
        "You are a helpful filenaming assistant. "
        "Filenames should use underscores instead of spaces, "
        "and should be all lowercase. "
        "Exclude the file extension. "
        "Give me a compact filename for the following prompt:"
    )
    response = bot(image_prompt)
    return response
