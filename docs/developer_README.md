# LlamaBot: A Pythonic bot interface to LLMs

This is a `developer` README for LlamaBot.

LlamaBot implements a Pythonic interface to LLMs,
making it much easier to experiment with LLMs in a Jupyter notebook
and build Python apps that utilize LLMs.
All models supported by [LiteLLM](https://github.com/BerriAI/litellm) are supported by LlamaBot.

## Setting up your developer environment

Make sure that you have followed the [README.md](../README.md) before jumping on the next steps.

## Creating a Conda virtual environment

It is recommended to have [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) available on your system.

```bash
conda env create -f environment.yml
```

After the conda environment creation, you may need to run the following commands.

```bash
conda activate llamabot
```

The following command will install all your pip dependencies

```bash
pip install -e .
```

## Installing pre-commit hook

It is recommended to install and use the [pre-commit](https://pre-commit.com/) hook if you plan to commit this project.
This will check for any issues before committing your code. It is also one of the recommended developers' best practices.

The following command will install pre-commit in your virtual environment.

```bash
pre-commit install
```

Now before you commit your changes, running the pre-commit command below will automatically run code checks for you.
If there are any issues that pre-commit finds, such as missing line breaks, lint errors or others,
then please make the appropriate changes, rerun pre-commit to ensure that the checks pass and then commit the files.

```bash
pre-commit run
```

## Configuring your downloaded model

If you are using a local model, you may need to update the `DEFAULT_LANGUAGE_MODEL` to your local model path by
running the following command:

Replace <your_model_name> wth your locally downloaded model.

For example:

- If you are using Mistral: replace `<your_model_name>` with `mistral`.
- If you are using Llama3: replace `<your_model_name>` with `llama3`.

```bash
llamabot configure default-model --model-name "ollama/<your_model_name>"
```

OR

You could also pass the language model as an argument when extending/creating your own model from the
[SimpleBot](../llamabot/bot/simplebot.py) api.

Example:

```python
feynman = SimpleBot(
    "You are Richard Feynman. You will be given a difficult concept, and your task is to explain it back.",
    model_name="ollama/llama3"
)
```
