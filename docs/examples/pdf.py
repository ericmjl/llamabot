# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot[rag]>=0.17.0",
#     "requests",
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
    # PDF Chatbot
    """
    )
    return


@app.cell
def _():
    # Download pre-built index.json file from Dropbox
    import requests

    headers = {"user-agent": "Wget/1.16 (linux-gnu)"}  # <-- the key is here!
    r = requests.get(
        "https://www.dropbox.com/s/wrixlu7e3noi43q/Ma%20et%20al.%20-%202021%20-%20Machine-Directed%20Evolution%20of%20an%20Imine%20Reductase%20f.pdf?dl=0",
        stream=True,
        headers=headers,
    )
    pdf_fname = "/tmp/machine-directed-evolution.pdf"
    with open(pdf_fname, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return


@app.cell
def _():
    from llamabot import QueryBot

    # If you're prototyping with your own PDF, uncomment the following code and use it instead of the saved index path:
    # bot = QueryBot(
    #     "You are a bot that reads a PDF book and responds to questions about that book.",
    #     document_paths=[pdf_fname],
    #     collection_name="machine-directed-evolution-paper",
    #     model_name="mistral/mistral-medium",
    # )

    bot = QueryBot(
        "You are a bot that reads a PDF book and responds to questions about that book.",
        collection_name="machine-directed-evolution-paper",
        model_name="mistral/mistral-medium",
    )
    return (bot,)


@app.cell
def _(bot):
    prompt = "I'd like to use the workflow of this paper to educate colleagues. What are the main talking points I should use?"
    bot(prompt)
    return


@app.cell
def _(bot):
    prompt_1 = "My colleagues are interested in evolving another enzyme. However, they may be unaware of how machine learning approaches will help them there. Based on this paper, what can I highlight that might overcome their lack of knowledge?"
    bot(prompt_1)
    return


@app.cell
def _(bot):
    prompt_2 = "What data from the paper helped show this point, 'Machine-directed evolution is an efficient strategy for enzyme engineering, as it can help navigate enzyme sequence space more effectively and reduce the number of enzyme variants to be measured en route to a desirable enzyme under realistic process conditions.'?"
    bot(prompt_2)
    return


@app.cell
def _(bot):
    prompt_3 = "How can I succinctly present the SGM vs. EPPCR results to my colleagues? Or in other words, how would Richard Feynman present these results?"
    bot(prompt_3)
    return (prompt_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Using SimpleBot below should prove that we are indeed querying a book
    and not just relying on the LLM's training set.
    """
    )
    return


@app.cell
def _(prompt_3):
    from llamabot import SimpleBot

    sbot = SimpleBot("You are a bot that responds to human questions.")
    sbot(prompt_3)
    return


if __name__ == "__main__":
    app.run()
