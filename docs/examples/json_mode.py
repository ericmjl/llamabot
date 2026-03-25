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
    from llamabot import SimpleBot

    bot = SimpleBot(
        "You are a bot proficient at returning JSON.",
        model_name="ollama/mistral",
        json_mode=True,
    )
    bot(
        "What is the weather like today? Return in JSON with the following structure: {'location': 'City Name', 'temperature': 'Temperature in Celsius', 'description': 'Weather Description'}"
    )
    return


if __name__ == "__main__":
    app.run()
