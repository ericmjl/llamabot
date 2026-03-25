# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot>=0.17.0",
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
    # ChatBots in a Jupyter Notebook
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's see how to use the ChatBot class to enable you to chat with Mistral inside a Jupyter notebook.
    """
    )
    return


@app.cell
def _():
    from llamabot import ChatBot

    code_tester = ChatBot(
        """
    You are a Python quality assurance developer who delivers high quality unit tests for code.
    You write tests using PyTest and not the built-in unittest library.
    Write the tests using test functions and not using classes and class methods
    Here is the code to write tests against:
    """,
        session_name="code-tested",
        model_name="mistral/mistral-medium",
        stream_target="stdout",
    )
    return (code_tester,)


@app.cell
def _(code_tester):
    code_tester(
        '''
    class ChatBot:
        """Chat Bot that is primed with a system prompt, accepts a human message.

        Automatic chat memory management happens.

        h/t Andrew Giessel/GPT4 for the idea.
        """

        def __init__(self, system_prompt, temperature=0.0, model_name="gpt-4"):
            """Initialize the ChatBot.

            :param system_prompt: The system prompt to use.
            :param temperature: The model temperature to use.
                See https://platform.openai.com/docs/api-reference/completions/create#completions/create-temperature
                for more information.
            :param model_name: The name of the OpenAI model to use.
            """
            self.model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=True,
                verbose=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
            self.chat_history = [
                SystemMessage(content="Always return Markdown-compatible text."),
                SystemMessage(content=system_prompt),
            ]

        def __call__(self, human_message) -> Response:
            """Call the ChatBot.

            :param human_message: The human message to use.
            :return: The response to the human message, primed by the system prompt.
            """
            self.chat_history.append(HumanMessage(content=human_message))
            response = self.model(self.chat_history)
            self.chat_history.append(response)
            return response
    '''
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see, ChatBot keeps track of conversation memory/history automatically.
    We can even access any item in the conversation by looking at the conversation history.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `__repr__` of a chatbot will simply print out the entire history:
    """
    )
    return


@app.cell
def _(code_tester):
    code_tester
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On the other hand, accessing the `.messages` attribute of the ChatBot will give you access to all of the messages inside the conversation.
    """
    )
    return


@app.cell
def _(code_tester):
    code_tester.messages
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You can even access any arbitrary message.
    """
    )
    return


@app.cell
def _(code_tester):
    print(code_tester.messages[-1].content)
    return


if __name__ == "__main__":
    app.run()
