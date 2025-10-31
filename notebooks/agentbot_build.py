# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import llamabot as lmb

    return (lmb,)


@app.cell
def _(lmb):
    bot = lmb.AgentBot(
        system_prompt="You are a helpful assistant.",
        model_name="ollama_chat/qwen2.5:0.5b",
    )
    bot("What's today's date?")
    return


@app.cell
def _(lmb):
    bot2 = lmb.AgentBot(
        system_prompt="You are a helpful assistant.", model_name="openai/gpt-4.1"
    )
    bot2("What's today's date?")
    return


@app.cell
def _(lmb):
    @lmb.tool
    def run_shell_command(command: str) -> str:
        """Run a shell command and return its output.

        :param command: The shell command to run.
        :return: The output of the command.
        """
        import subprocess

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr

    return (run_shell_command,)


@app.cell
def _(lmb, run_shell_command):
    bot3 = lmb.AgentBot(
        system_prompt="You are a helpful assistant.",
        model_name="ollama_chat/qwen2.5:0.5b",
        tools=[run_shell_command],
    )
    response = bot3(
        "What did I do on github recently? Use the run_shell_command tool to figure this out using the `gh` CLI."
    )
    print(response.content)
    return


@app.cell
def _(lmb, run_shell_command):
    bot4 = lmb.AgentBot(
        system_prompt="You are a helpful assistant.",
        model_name="ollama_chat/qwen2.5:0.5b",
        tools=[run_shell_command],
    )
    response4 = bot4(
        "What files are in my current working directory? Use the run_shell_command to figure it out."
    )
    print(response4.content)
    return


@app.cell
def _(lmb, run_shell_command):
    bot5 = lmb.AgentBot(
        system_prompt="You are a helpful assistant.",
        model_name="openai/gpt-4.1",
        tools=[run_shell_command],
    )
    response5 = bot5("What files are in here?")
    print(response5.content)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
