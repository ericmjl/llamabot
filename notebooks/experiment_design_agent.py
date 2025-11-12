# Script metadata for marimo/llamabot local dev
# Requires llamabot to be installed locally (editable install recommended)
# Requires: pip: llamabot

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]>=0.17.1",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import llamabot as lmb

    return (lmb,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def experiment_design_critique_sysprompt():
        """You are an expert statistician that will help poke holes
        in proposed experiment designs to help make them robust.

        You will be provided with a proposed experiment design.
        Your task is to identify potential flaws, biases, or weaknesses in the design,
        and to ask questions back of the propose to clarify any ambiguous aspects.
        If you are given back a list of things to talk through, go through them one by one.
        """

    @lmb.nodeify(loopback_name="decide")
    @lmb.tool
    def critique_experiment_design(design: str) -> str:
        """Critique an experiment design and identify potential flaws, biases, or weaknesses.

        :param design: Description of the proposed experiment design
        :return: Critique of the experiment design with identified issues and questions, as well as suggestions for improvement.
        """
        bot = lmb.SimpleBot(system_prompt=experiment_design_critique_sysprompt())
        return bot(design)

    return (critique_experiment_design,)


@app.cell
def _(critique_experiment_design, lmb):
    bot = lmb.AgentBot(tools=[critique_experiment_design])
    return (bot,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(bot, mo):
    example_prompts = [
        "I want to test if compound X reduces cell viability. I'll treat cells with 3 doses of compound X and measure viability after 24 hours.",
        "I'm comparing two treatments: drug A vs drug B. I'll use 5 replicates per group and measure the response after 48 hours.",
        "I want to screen 10 compounds for activity. I'll test each at a single concentration with 3 replicates.",
    ]

    def chat_turn(messages, config):
        user_message = messages[-1].content
        print(user_message)
        result = bot(user_message)
        return result

    chat = mo.ui.chat(chat_turn, max_height=600, prompts=example_prompts)
    return (chat,)


@app.cell
def _(bot):
    bot
    return


@app.cell
def _(chat, mo):
    ui = mo.vstack(
        [
            mo.md("# Experiment Design Agent"),
            mo.md(
                "Chat with an expert statistician to critique and improve your experiment designs."
            ),
            chat,
        ]
    )
    ui
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
