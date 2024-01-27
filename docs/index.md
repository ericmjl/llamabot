# LlamaBot: A Pythonic bot interface to LLMs

LlamaBot implements a Pythonic interface to LLMs,
making it much easier to experiment with LLMs in a Jupyter notebook
and build Python apps that utilize LLMs.
All models supported by [LiteLLM](https://github.com/BerriAI/litellm) are supported by LlamaBot.

## Install LlamaBot

To install LlamaBot:

```python
pip install llamabot
```

## Get access to LLMs

### Option 1: Using local models with Ollama

LlamaBot supports using local models through Ollama.
To do so, head over to the [Ollama website](https://ollama.ai) and install Ollama.
Then follow the instructions below.

### Option 2: Use an API provider

#### OpenAI

If you have an OpenAI API key, then configure LlamaBot to use the API key by running:

```bash
export OPENAI_API_KEY="sk-your1api2key3goes4here"
```

#### Mistral

If you have a Mistral API key, then configure LlamaBot to use the API key by running:

```bash
export MISTRAL_API_KEY="your-api-key-goes-here"
```

#### Other API providers

Other API providers will usually specify an environment variable to set.
If you have an API key, then set the environment variable accordingly.

## How to use

### SimpleBot

The simplest use case of LlamaBot
is to create a `SimpleBot` that keeps no record of chat history.
This is effectively the same as a _stateless function_
that you program with natural language instructions rather than code.
This is useful for prompt experimentation,
or for creating simple bots that are preconditioned on an instruction to handle texts
and are then called upon repeatedly with different texts.
For example, to create a Bot that explains a given chunk of text
like Richard Feynman would:

```python
from llamabot import SimpleBot

feynman = SimpleBot("You are Richard Feynman. You will be given a difficult concept, and your task is to explain it back.", model_name="gpt-3.5-turbo")
```

Now, `feynman` is callable on any arbitrary chunk of text and will return a rephrasing of that text in Richard Feynman's style (or more accurately, according to the style prescribed by the prompt).
For example:

```python
feynman("Enzyme function annotation is a fundamental challenge, and numerous computational tools have been developed. However, most of these tools cannot accurately predict functional annotations, such as enzyme commission (EC) number, for less-studied proteins or those with previously uncharacterized functions or multiple activities. We present a machine learning algorithm named CLEAN (contrastive learning–enabled enzyme annotation) to assign EC numbers to enzymes with better accuracy, reliability, and sensitivity compared with the state-of-the-art tool BLASTp. The contrastive learning framework empowers CLEAN to confidently (i) annotate understudied enzymes, (ii) correct mislabeled enzymes, and (iii) identify promiscuous enzymes with two or more EC numbers—functions that we demonstrate by systematic in silico and in vitro experiments. We anticipate that this tool will be widely used for predicting the functions of uncharacterized enzymes, thereby advancing many fields, such as genomics, synthetic biology, and biocatalysis.")
```

This will return something that looks like:

```text
Alright, let's break this down.

Enzymes are like little biological machines that help speed up chemical reactions in our
bodies. Each enzyme has a specific job, or function, and we use something called an
Enzyme Commission (EC) number to categorize these functions.

Now, the problem is that we don't always know what function an enzyme has, especially if
it's a less-studied or new enzyme. This is where computational tools come in. They try
to predict the function of these enzymes, but they often struggle to do so accurately.

So, the folks here have developed a new tool called CLEAN, which stands for contrastive
learning–enabled enzyme annotation. This tool uses a machine learning algorithm, which
is a type of artificial intelligence that learns from data to make predictions or
decisions.

CLEAN uses a method called contrastive learning. Imagine you have a bunch of pictures of
cats and dogs, and you want to teach a machine to tell the difference. You'd show it
pairs of pictures, some of the same animal (two cats or two dogs) and some of different
animals (a cat and a dog). The machine would learn to tell the difference by contrasting
the features of the two pictures. That's the basic idea behind contrastive learning.

CLEAN uses this method to predict the EC numbers of enzymes more accurately than
previous tools. It can confidently annotate understudied enzymes, correct mislabeled
enzymes, and even identify enzymes that have more than one function.

The creators of CLEAN have tested it with both computer simulations and lab experiments,
and they believe it will be a valuable tool for predicting the functions of unknown
enzymes. This could have big implications for fields like genomics, synthetic biology,
and biocatalysis, which all rely on understanding how enzymes work.
```

If you want to use an Ollama model hosted locally,
then you would use the following syntax:

```python
from llamabot import SimpleBot
bot = SimpleBot(
    "You are Richard Feynman. You will be given a difficult concept, and your task is to explain it back.",
    model_name="ollama/llama2:13b"
)
```

Simply specify the `model_name` keyword argument
and provide a model name from the [Ollama library of models](https://ollama.ai/library)
prefixed by `ollama/`.
All you need to do is make sure Ollama is running locally;
see the [Ollama documentation](https://ollama.ai/) for more details.
(The same can be done for the `ChatBot` and `QueryBot` classes below!)

### Chat Bot

To experiment with a Chat Bot in the Jupyter Notebook,
we also provide the ChatBot interface.
This interface automagically keeps track of chat history
for as long as your Jupyter session is alive.
Doing so allows you to use your own local Jupyter Notebook as a chat interface.

For example:

```python
from llamabot import ChatBot

feynman = ChatBot("You are Richard Feynman. You will be given a difficult concept, and your task is to explain it back.", session_name="feynman_chat")
feynman("Enzyme function annotation is a fundamental challenge, and numerous computational tools have been developed. However, most of these tools cannot accurately predict functional annotations, such as enzyme commission (EC) number, for less-studied proteins or those with previously uncharacterized functions or multiple activities. We present a machine learning algorithm named CLEAN (contrastive learning–enabled enzyme annotation) to assign EC numbers to enzymes with better accuracy, reliability, and sensitivity compared with the state-of-the-art tool BLASTp. The contrastive learning framework empowers CLEAN to confidently (i) annotate understudied enzymes, (ii) correct mislabeled enzymes, and (iii) identify promiscuous enzymes with two or more EC numbers—functions that we demonstrate by systematic in silico and in vitro experiments. We anticipate that this tool will be widely used for predicting the functions of uncharacterized enzymes, thereby advancing many fields, such as genomics, synthetic biology, and biocatalysis.")
```

With the chat history available, you can ask a follow-up question:

```python
feynman("Is there a simpler way to rephrase the text such that a high schooler would understand it?")
```

And your bot will work with the chat history to respond.

### QueryBot

The final bot provided is a QueryBot.
This bot lets you query a collection of documents.
To use it, you have two options:

1. Pass in a list of paths to text files, or
2. Pass in a session name of a previously instantiated `QueryBot` that model. (This will load the previously-computed text index into memory.)

As an illustrative example:

```python
from llamabot import QueryBot
from pathlib import Path

blog_index = Path("/path/to/index.json")
bot = QueryBot(system_message="You are an expert on Eric Ma's blog.", session_name="eric_ma_blog")  # this loads my previously-embedded blog text.
# alternatively:
# bot = QueryBot(system_message="You are an expert on Eric Ma's blog.", session_name="eric_ma_blog", document_paths=[Path("/path/to/blog/post1.txt"), Path("/path/to/blog/post2.txt"), ...])
result = bot("Do you have any advice for me on career development?")
display(Markdown(result.response))
```

### ImageBot

With the release of the OpenAI API updates,
as long as you have an OpenAI API key,
you can generate images with LlamaBot:

```python
from llamabot import ImageBot

bot = ImageBot()
# Within a Jupyter notebook:
url = bot("A painting of a dog.")

# Or within a Python script
filepath = bot("A painting of a dog.")

# Now, you can do whatever you need with the url or file path.
```

If you're in a Jupyter Notebook,
you'll see the image show up magically as part of the output cell as well.

## CLI Demos

Llamabot comes with CLI demos of what can be built with it and a bit of supporting code.

Here is one where I expose a chatbot directly at the command line using `llamabot chat`:

<script async id="asciicast-594332" src="https://asciinema.org/a/594332.js"></script>

And here is another one where `llamabot` is used as part of the backend of a CLI app
to chat with one's Zotero library using `llamabot zotero chat`:

<script async id="asciicast-594326" src="https://asciinema.org/a/594326.js"></script>

And finally, here is one where I use `llamabot`'s `SimpleBot` to create a bot
that automatically writes commit messages for me.

<script async id="asciicast-594334" src="https://asciinema.org/a/594334.js"></script>

## Contributing

### New features

New features are welcome!
These are early and exciting days for users of large language models.
Our development goals are to keep the project as simple as possible.
Features requests that come with a pull request will be prioritized;
the simpler the implementation of a feature (in terms of maintenance burden),
the more likely it will be approved.

### Bug reports

Please submit a bug report using the issue tracker.

### Questions/Discussions

Please use the issue tracker on GitHub.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://ericmjl.github.io/"><img src="https://avatars.githubusercontent.com/u/2631566?v=4?s=100" width="100px;" alt="Eric Ma"/><br /><sub><b>Eric Ma</b></sub></a><br /><a href="https://github.com/modernatx/seqlike/commits?author=ericmjl" title="Code">💻</a> <a href="https://github.com/modernatx/seqlike/commits?author=ericmjl" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
