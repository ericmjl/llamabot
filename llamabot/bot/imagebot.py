"""ImageBot module for generating images."""

import base64
import binascii
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from llamabot.components.messages import AIMessage


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
        self.image_generation_kwargs = image_generation_kwargs

    def __call__(
        self, prompt: str, return_url: bool = False, save_path: Optional[Path] = None
    ) -> Union[str, Path]:
        """Generate an image from a prompt.

        :param prompt: The prompt to generate an image from.
        :param return_url: Whether to return a displayable image reference.
            Overrides the save_path parameter. Defaults to False. URL responses
            return a URL; base64 responses return a data URI.
        :param save_path: The path to save the generated image to.
            If it is empty, then we will generate a filename from the prompt.
        :return: The generated image reference if running in a Jupyter notebook
            or if ``return_url`` is True, otherwise a pathlib.Path object pointing
            to the generated image.
        :raises ValueError: If no generated image data is found in the response.
        """
        response = generate_image_response(self, prompt)
        image = first_generated_image(response)

        # Check if running in a Jupyter notebook
        if is_running_in_jupyter():
            try:
                from IPython.display import display, Image

                if getattr(image, "url", None):
                    display(Image(url=image.url))
                else:
                    display(Image(data=generated_image_bytes(image)))
                return generated_image_reference(image)
            except ImportError:
                pass  # IPython not available, fall through to normal behavior

        if return_url:
            return generated_image_reference(image)

        image_data = generated_image_bytes(image)

        if not save_path:
            filename = filename_bot(prompt).content
            save_path = Path(f"{filename}.jpg")
        save_path = Path(save_path)
        save_path.write_bytes(image_data)
        return save_path


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
    from litellm import image_generation

    return image_generation(**image_generation_kwargs_for_bot(bot, prompt))


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


def generated_image_reference(image: Any) -> str:
    """Return a displayable reference for a generated image.

    :param image: A LiteLLM image object.
    :return: The image URL, or a base64 data URI when only base64 data is present.
    :raises ValueError: If neither URL nor base64 image data is present.
    """
    image_url = getattr(image, "url", None)
    if image_url:
        return image_url

    b64_json = getattr(image, "b64_json", None)
    if b64_json:
        return f"data:image/png;base64,{b64_json}"

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
