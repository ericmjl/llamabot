# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot>=0.17.0",
#     "pandas",
# ]
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
    # LLaMaBot's `StructuredBot` in under 5 minutes
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    When using LLMs, an ideal goal would be to
    pull structured data out of unstructured text.
    When the data is structured,
    we can then use it programmatically in later steps.

    In this example, we'll look at a small dataset of SciPy videos uploaded to YouTube.
    The videos are given a title and a description.
    We want to extract the name of the speaker giving the talk,
    and the topics the talk is about.
    We also want to be able to validate the data we've extracted
    not only matches the structured format we expect,
    but that it also meets some custom requirements.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Read video descriptions

    Firstly, let's look at the video descriptions file.
    It is stored as a JSON file.
    We can read it into pandas by using `pd.read_json`:
    """
    )
    return


@app.cell
def _():
    # load in unstructured text data
    import pandas as pd

    df = pd.read_json("../scipy_videos.json", orient="index")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's now define a Pydantic schema for the data that we wish to extract from movie entry.
    This is doen by defining a BaseModel class and field validators.
    """
    )
    return


@app.cell
def _():
    from typing import List, Optional
    from pydantic import BaseModel, Field, field_validator

    class TopicExtract(BaseModel):
        """This object stores the name of the speaker presenting the video.

        It also generates a list of topics
        that best describe what this talk is about.
        """

        speaker_name: Optional[str] = Field(
            default=None,
            description=(
                "The name of the speaker giving this talk. "
                "If there is no speaker named, leave empty."
            ),
        )
        topics: List[str] = Field(
            description=(
                "A list of upto 5 topics that this text is about. "
                "Each topic should be at most 1 or 2 word descriptions. "
                "All lowercase."
            )
        )

        @field_validator("topics")
        def validate_num_topics(cls, topics):
            # validate that the list of topics contains atleast 1, and no more than 5 topics
            if len(topics) <= 0 or len(topics) > 5:
                raise ValueError("The list of topics can be no more than 5 items")
            return topics

        @field_validator("topics")
        def validate_num_topic_words(cls, topics):
            # for each topic the model generated, ensure that the topic contains no more than 2 words
            for topic in topics:
                if len(topic.split()) > 2:
                    # make the validation message helpful to the LLM.
                    # Here we repeat which topic is failing validation, and remind it what it must do to pass the validation.
                    raise ValueError(
                        f'The topic "{topic}" has too many words, A topic can contain AT MOST 2 words'
                    )
            return topics

    return (TopicExtract,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can initialize the PydanticBot and assign this model to it.
    """
    )
    return


@app.cell
def _(TopicExtract):
    from llamabot import prompt, StructuredBot

    @prompt
    def topicbot_sysprompt() -> str:
        """You are an expert topic labeller.
        You read a video title and description
        and extract the speakers name and the topics the video is about.
        """

    # Will use the OpenAI API by default, which requires an API key.
    # If you want to, you can change this to a local LLM (from Ollama)
    # by specifying, say, `model_name="ollama/mistral"`.
    bot = StructuredBot(
        system_prompt=topicbot_sysprompt(),
        temperature=0,
        pydantic_model=TopicExtract,
        # model_name="ollama/mistral"
    )
    return (bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can pass in our text, and extract the topics
    """
    )
    return


@app.cell
def _(bot, df):
    video_extracts = []
    for index, video_row in df.iterrows():
        video_text = f"video title: {video_row['name']}\nvideo description: {video_row['description']}"

        extract = bot(video_text)

        video_extracts.append(extract)
    return (video_extracts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's now inspect what the topics looked like.
    """
    )
    return


@app.cell
def _(video_extracts):
    for video in video_extracts:
        print(video)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Look's pretty accurate!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    """
    )
    return


if __name__ == "__main__":
    app.run()
