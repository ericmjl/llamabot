# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm==1.70.0",
#     "llamabot[all]==0.12.0",
#     "marimo",
#     "nbclient==0.10.2",
#     "numpydoc==1.8.0",
#     "pydantic==2.11.4",
#     "pyprojroot==0.3.0",
#     "rich==14.0.0",
#     "tqdm==4.67.1",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# A Quick Introduction to LlamaBot""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    LlamaBot provides you, a Python programmer, with a Pythonic toolkit for writing LLM programs and learning about LLMs. It was designed with two goals in mind: (a) pedagogy -- to make it easy to learn, and (b) ergonomics -- to make it easy to build these programs.

    Let's dive right in.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## `SimpleBot`

    `SimpleBot` is the starting point for LlamaBot:
    """
    )
    return


@app.cell
def _():
    import llamabot as lmb

    comedian_sysprompt = lmb.system("You are an expert comedian.")

    simple_bot = lmb.SimpleBot(
        system_prompt=comedian_sysprompt,
        model_name="ollama_chat/mistral-small3.1",
    )

    sb_response = simple_bot("Tell me a joke about programming.")
    return comedian_sysprompt, lmb, simple_bot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""SimpleBot has no memory, as you can see below:""")
    return


@app.cell
def _(simple_bot):
    simple_bot("What have we talked about?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ChatBots are `SimpleBot`s with memory

    We can give a `SimpleBot` memory by using a document store, effectively turning it into a ChatBot.
    """
    )
    return


@app.cell
def _(comedian_sysprompt, lmb):
    comedian_memory = lmb.LanceDBDocStore(table_name="comedian_memory")
    comedian_memory.reset()

    chat_bot = lmb.SimpleBot(
        system_prompt=comedian_sysprompt,
        chat_memory=comedian_memory,
        model_name="ollama_chat/mistral-small3.1",
    )

    chat_bot("Tell me a joke about programming.")
    return chat_bot, comedian_memory


@app.cell
def _(chat_bot):
    chat_bot("What have we talked about?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Chat with Documents

    We can also chat with our documents. As an example, let's chat with LlamaBot's docs.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Step 1: Store documents in document store""")
    return


@app.cell
def _(lmb):
    from pyprojroot import here

    llamabot_docs = lmb.LanceDBDocStore(table_name="llamabot-docs")

    llamabot_mds = (f.read_text() for f in (here() / "docs").rglob("*.md"))
    llamabot_docs.extend(llamabot_mds)
    return (llamabot_docs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 2: Begin chatting

    Connect a QueryBot to the docstore and start chatting!
    """
    )
    return


@app.cell
def _(llamabot_docs, lmb):
    llamabot_docbot = lmb.QueryBot(
        system_prompt="You are an expert Python programmer.",
        docstore=llamabot_docs,
        model_name="ollama_chat/mistral-small3.1",
    )

    llamabot_docbot("How do I use StructuredBot to extract information from papers?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Transplant memories from one bot to another

    Because of the modular nature of memory, it is possible to transplant memory from one bot to another.
    """
    )
    return


@app.cell
def _(comedian_memory, simple_bot):
    simple_bot.chat_memory = comedian_memory

    simple_bot("What have we talked about so far?")

    return


@app.cell
def _(comedian_memory):
    comedian_memory.retrieve("What have we talked about so far?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Now, `simple_bot` and `chat_bot` are effectively neuralinked together via `comedian_memory`. We can make `simple_bot` amnesic again simply by setting `simple_bot.chat_memory = None`."""
    )
    return


@app.cell
def _(simple_bot):
    simple_bot.chat_memory = None

    simple_bot("What have we talked about so far?")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Structured Generation

    LlamaBot can also help you do structured generation easily.
    """
    )
    return


@app.cell
def _(lmb):
    from pydantic import BaseModel, Field

    class ComedicName(BaseModel):
        name: str
        backstory: str = Field(description="Backstory of the comedian.")

    amnesic_structured_bot = lmb.StructuredBot(
        system_prompt="You are a comedic name generator.",
        pydantic_model=ComedicName,
        model_name="ollama_chat/mistral-small3.1",
        temperature=0.8,
    )

    amnesic_structured_bot("Give me a comedian's name.")
    return BaseModel, ComedicName, Field


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Like other bots, we can add memory! However, because the `__call__` method doesn't explicitly handle `chat_memory`, it will get ignored."""
    )
    return


@app.cell
def _(ComedicName, lmb):
    sb_memory = lmb.LanceDBDocStore(table_name="comedian-generator-memory")

    structured_bot = lmb.StructuredBot(
        system_prompt="You are a comedic name generator.",
        pydantic_model=ComedicName,
        model_name="ollama_chat/mistral-small3.1",
        temperature=0.8,
        chat_memory=sb_memory,
    )

    structured_bot("Give me a comedian's name.")
    return sb_memory, structured_bot


@app.cell
def _(structured_bot):
    structured_bot("Give me a comedian's name that is a variant of the previous one.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""To prove the point, there are no items that are present in `sb_memory`."""
    )
    return


@app.cell
def _(sb_memory):
    sb_memory.retrieve("All names please.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""As such, any similarities between the two calls is purely coincidental."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Autonomous actions through agents

    LlamaBot also has a pedagogical Agent implementation that shows how an LLM can be used to take autonomous actions, such as searching the web and writing code to accomplish a goal.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Built-in tools

    Two tools built into llamabot are available:

    - `search_internet_and_summarize`, which searches the internet and writes a summary response to the query for each result page, and
    - `write_and_execute_script`, which the bot uses to execute a script that it writes within a secure Docker container locally.

    Let's see them in action.
    """
    )
    return


@app.cell
def _(lmb):
    from llamabot.components.tools import (
        search_internet_and_summarize,
        write_and_execute_script,
    )

    bot = lmb.AgentBot(
        system_prompt="You are a helpful assistant.",
        tools=[search_internet_and_summarize, write_and_execute_script],
        model_name="gpt-4.1",  # In vibe testing, this is the model that works best.
    )

    return bot, search_internet_and_summarize


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Every function that is passed in is decorated with the `lmb.tool` decorator that attaches a `.json_schema` to each function. Here is one example, for the `search_internet_and_summarize` function:"""
    )
    return


@app.cell
def _(search_internet_and_summarize):
    search_internet_and_summarize.json_schema
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Here's a task that we should be able to get an LLM to execute on. This first task _should_ use the `search_internet_and_summarize` tool to search the internet. I have intentionally included a few spelling errors to make the task a bit harder for the LLM."""
    )
    return


@app.cell
def _(bot):
    response = bot("What are the latest ratings on Taylor Swfit's latset album?")
    print(response)

    return


@app.cell
def _(bot):
    response2 = bot(
        "Download the red wine quality dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv. Use the provided column headers. Train a random forest classifier with 100 trees and a maximum depth of 5 to predict the ‘quality’ column using the other features. Perform 5-fold cross-validation and return the mean and standard deviatlitellmion of the accuracy."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## `__call__` defines the Bot's behaviour

    Essentially, the `__call__` method defines a Bot's behaviour. We can create new bots that exhibit different behaviours by varying how we compose together:

    - response generation
    - tool calling
    - accessing memory and external documents

    Armed with this knowledge, we can create custom bots that do stuff for us.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Custom Bots that Interact

    Let's take advantage of LlamaBot's customizability to build a custom _deep research_ agent that automatically does literature deep dives for us.

    For example, I may want to build a two agent system, in which one generates poetry on a topic that is provided and the other evaluates the poetry for certain criteria, and we allow the agents to continually interact until the evaluator believes the criteria has been satisfied. This (somewhat contrived) situation will allow us to illustrate the compositionality of bots, and how bots can be comprised of bots.
    """
    )
    return


@app.cell
def _(BaseModel, Field, lmb):
    class CriteriaEvaluation(BaseModel):
        passes: bool = Field(
            description="Whether or not the poem passes the evaluation or not."
        )
        reason: str = Field(
            description="Why the evaluation pass and score was given as such, and how to improve the score."
        )

    evaluation_bot = lmb.StructuredBot(
        # Put the criteria in the system prompt
        system_prompt=lmb.system(
            "You will evaluate a given poem for its ability to induce a high emotional reaction in a person. Return True if it will and False if it won't."
        ),
        pydantic_model=CriteriaEvaluation,
        model_name="ollama_chat/mistral-small3.1",
    )

    writer_bot = lmb.SimpleBot(
        system_prompt=lmb.system(
            "You will be given a keyword theme and optionally feedback on how to improve your previous iterations of the poem. Use it to generate an upgraded version of the poem."
        ),
        model_name="ollama_chat/mistral-small3.1",
    )

    @lmb.prompt("user")
    def write_poem_prompt(theme: str, previous_poem="", previous_evaluation=""):
        """Here is the theme keywordd: {{ theme }}.

        {% if previous_poem %}
        You previously wrote this poem:

        {{ previous_poem }}
        {% endif %}

        {% if previous_evaluation %}
        The evaluation of the previous version is as follows:

        Score: {{ previous_evaluation.score }}
        Reason: {{ previous_evaluation.reason}}
        {% endif %}

        Based on this, generate a new poem for me.
        """

    return evaluation_bot, write_poem_prompt, writer_bot


@app.cell
def _(mo):
    mo.md(r"""As always, make sure the individual components of the system work.""")
    return


@app.cell
def _(writer_bot):
    test_poem = writer_bot("losing a loved one")
    test_poem
    return (test_poem,)


@app.cell
def _(evaluation_bot, test_poem):
    test_evaluation = evaluation_bot(test_poem.content)

    print(test_evaluation.passes)
    print(test_evaluation.reason)
    return


@app.cell
def _(mo):
    mo.md(
        r"""Now, let's put this into an autonomous poem generation loop. Here, we will have a writer bot write a poem, and then have an evaluation bot score the poem based on the "emotional reaction" criteria."""
    )
    return


@app.cell
def _(evaluation_bot, write_poem_prompt, writer_bot):
    passes_criteria = False
    previous_evaluation = ""
    previous_poem = ""
    while not passes_criteria:
        poem = writer_bot(
            write_poem_prompt(
                theme="DNA Biotechnology",
                previous_poem=previous_poem,
                previous_evaluation=previous_evaluation,
            )
        )
        eval = evaluation_bot(poem)
        if eval.passes:
            passes_criteria = True
        previous_evaluation = eval
        previous_poem = poem

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""I won't go into this in too much detail, but this is an example of an "LLM-as-a-judge" in action."""
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
