# LlamaBot: A Pythonic bot interface to LLMs

This is a `developer` README for LlamaBot.

LlamaBot implements a Pythonic interface to LLMs,
making it much easier to experiment with LLMs in a Jupyter notebook
and build Python apps that utilize LLMs.
All models supported by [LiteLLM](https://github.com/BerriAI/litellm) are supported by LlamaBot.


## Setting up your developer environment

Make sure that you have followed the [README.md](../README.md) before jumping on the next steps.

## Create a Conda virtual environment

It is recommended to have [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) available on your system.

```bash
conda env create -f environment.yml
```

After the conda environment creation, you may need to do run the following commands.

```bash
conda activate llamabot 
```

The following command will install all your pip dependencies
```bash
pip install -e .
```

## Install LlamaBot

To install LlamaBot:

```bash
pip install llamabot
```

Please note: if you are using a local model, you may need to update the `DEFAULT_LANGUAGE_MODEL` to your local model path
at [config.py](../llamabot/config.py).

this could be something similar to "ollama/<your_model>" 

For Llama3: (ex. "ollama/llama3")  
For Mistral: (ex. "ollama/mistral")