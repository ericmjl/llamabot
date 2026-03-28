# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ImageBot

    This notebook shows how to use the ImageBot API to ingest or generate images from text.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Image Ingestion

    For image ingestion, we will use the `SimpleBot` class, which can take an iterable of messages and pass them to the LLM.  Making one of the messages an image URL or a local file path will automatically convert it into a format that can be used by the LLM.

    In this example, we will use a local LLM (Gemma 3n) hosted on LM Studio on an Apple Silicon Mac.  You can choose any LLM that is compatible with your computer architecture (including non-local models) as long as they can process images.

    First you need to set up the environment variable to point to your LM Studio instance.  You can skip this step if you are using a non-local model.
    """
    )
    return


@app.cell
def _():
    # Define the API base (for LM Studio), API key, and model name
    #
    # NOTE: If you are using another service with a real API key,
    # you should NOT store it in plain text here. You should probably
    # use environment variables to manage sensitive information.
    API_BASE = "http://localhost:1234/v1"
    API_KEY = "lm-studio"  # This is a dummy value to bypass the check
    MODEL_NAME = "lm_studio/gemma-3n-e4b-it-mlx"

    # Define the temperature for the model's responses
    TEMPERATURE = 0.2
    return API_BASE, API_KEY, MODEL_NAME, TEMPERATURE


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can create a `SimpleBot` instance and connect to the LLM.
    """
    )
    return


@app.cell
def _(API_BASE, API_KEY, MODEL_NAME, TEMPERATURE):
    import llamabot as lmb
    from llamabot import SimpleBot
    from pathlib import Path

    # This example code was written and tested on an Apple Silicon Mac
    # using the LM Studio application to host a Gemma 3n model downloaded
    # from Hugging Face:
    # https://huggingface.co/lmstudio-community/gemma-3n-E4B-it-MLX-bf16
    #
    # Use lm_studio/ prefix to access local models through LM Studio.
    # You can also use other models (e.g. OpenAI or Ollama models)
    # as long as they support image inputs.  See the documentation for details.

    system_prompt = """You are a helpful assistant that can analyze images and
    provide detailed descriptions of those images.  You will also try to answer
    any questions about the images to the best of your ability."""

    bot = SimpleBot(
        system_prompt,
        temperature=TEMPERATURE,
        api_base=API_BASE,
        api_key=API_KEY,
        model_name=MODEL_NAME,
    )
    return Path, bot, lmb


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we use the bot to process a message that includes an image.  We can do this by passing a list of messages to the bot, one of which is an image file path. The image we will use is shown below:

    ![Bearly There](./Bearly_There.JPG)

    **Image Credit**: Photo by [Juan Cabanela](http://web.mnstate.edu/cabanela/) and is provided under a [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
    """
    )
    return


@app.cell
def _(Path, bot, lmb):
    # Ask the bot to describe an image localed at the given path
    image_path = Path("./Bearly_There.JPG")

    first_message = [
        lmb.user("Briefly (in less than 25 words) describe the following image: "),
        lmb.user(image_path),
    ]

    response = bot(first_message)
    return first_message, response


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    So the previous cell properly ingested the image and passed it to the LLM.  The LLM then generated a response based on the image content.  However, when using `SimpleBot` the context is not saved, so we cannot ask follow-up questions about the image.

    For example, if we try to ask a follow-up question about the image, the bot will not remember the previous interaction, and thus will respond in a way that does not reference the image.
    """
    )
    return


@app.cell
def _(bot, lmb):
    # This will not work as you might expect because the bot has no memory
    followup_message = [lmb.user("What else can you tell me about the bear?")]
    _response2 = bot(followup_message)
    return (followup_message,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can address this by creating a memory store for the chat which can hold the context of the first conversation and the response.  Here we will use a simple list to hold the chat history.
    """
    )
    return


@app.cell
def _(first_message, response):
    # Create a memory store for the chat which can hold the context of the
    # conversation.
    chat_memory = []

    # Combine the initial message and the response into the chat memory
    chat_memory.extend([first_message, response])
    return (chat_memory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can ask follow-up questions about the image and the bot will remember the context.  **NOTE**: You may need to increase the number of tokens the model can use to ensure it has enough context to answer the question.
    """
    )
    return


@app.cell
def _(bot, chat_memory, followup_message):
    # Starting with the chat memory for the previous interaction,
    # ask a followup question about the image and then send all
    # of that to the bot with memory.
    messages = chat_memory + followup_message
    # Call the bot with the full message history
    _response2 = bot(messages)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Image Generation

    Image generation, due to the rather large memory requirements, is normally not available on local models. We will need to use an visual language model, which is available through the OpenAI API. It is assumed you have set up your OpenAI API key in the environment variable (as per [OpenAI's best practices](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) documentation).

    Once we have set up the environment variable, we can load the API key:
    """
    )
    return


@app.cell
def _():
    # Load an OpenAI API Key from an environment variable and select an
    # OpenAI model to use
    import os

    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # We will use the DALL-E 3 model for image generation, which is not
    # the newest model but is still quite capable.
    _OPENAI_MODEL = "dall-e-3"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now let's call the `ImageBot` to generate an image from a text prompt.  The generated image will be returned as a URL that is used to display the image in the notebook.  If you want to save the image locally, you can use the `requests` library to download it.
    """
    )
    return


@app.cell
def _(Path):
    from llamabot.bot.imagebot import ImageBot

    # Create an ImageBot instance
    # The supported sizes are: '1024x1024', '1024x1792', and '1792x1024'
    # with the default being '1024x1024'.
    img_gen_bot = ImageBot(
        size="1024x1024",  # The default size is 1024x1024
    )

    img_gen_bot(
        "A grizzly bear eating some berries at a picnic table in a forest.",
        return_url=True,
        save_path=Path("./generated_bear_image.png"),
    )
    return


if __name__ == "__main__":
    app.run()
