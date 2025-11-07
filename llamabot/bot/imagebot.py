"""ImageBot module for generating images."""

import requests
from pathlib import Path
from typing import Optional, Union
from llamabot.components.messages import AIMessage


class ImageBot:
    """ImageBot class for generating images.

    :param model: The model to use. Defaults to "dall-e-3".
    :param size: The size of the image to generate. Defaults to "1024x1024".
    :param quality: The quality of the image to generate. Defaults to "standard".
    :param n: The number of images to generate. Defaults to 1.
    """

    def __init__(self, model="dall-e-3", size="1024x1024", quality="hd", n=1):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.size = size
        self.quality = quality
        self.n = n

    def __call__(
        self, prompt: str, return_url: bool = False, save_path: Optional[Path] = None
    ) -> Union[str, Path]:
        """Generate an image from a prompt.

        :param prompt: The prompt to generate an image from.
        :param return_url: Whether to return the URL of the generated image.
            Overrides the save_path parameter. Defaults to False.
        :param save_path: The path to save the generated image to.
            If it is empty, then we will generate a filename from the prompt.
        :return: The URL of the generated image if running in a Jupyter notebook (str),
            otherwise a pathlib.Path object pointing to the generated image.
        :raises Exception: If no image URL is found in the response.
        """
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            quality=self.quality,
            n=self.n,
        )
        image_url = response.data[0].url
        if not image_url:
            raise Exception("No image URL found in response! Please try again.")

        # Check if running in a Jupyter notebook
        if is_running_in_jupyter():
            try:
                from IPython.display import display, Image

                display(Image(url=image_url))
                return image_url
            except ImportError:
                pass  # IPython not available, fall through to normal behavior

        if return_url:
            return image_url

        image_data = requests.get(image_url).content

        if not save_path:
            filename = filename_bot(prompt).content
            save_path = Path(f"{filename}.jpg")
        with open(save_path, "wb") as file:
            file.write(image_data)
        return save_path


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
