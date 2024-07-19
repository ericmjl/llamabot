## Test with Anthropic API (Claude Sonnet)


```python
import os
from llamabot import SimpleBot
from dotenv import load_dotenv
```

URL for Anthropic API key: https://console.anthropic.com/settings/keys

If it's your first time using an Anthropic LLM, you will need to create an account and load funds (min $5).

Simple calls to the API like the ones seen below cost about 1 cent per request

Set up the API Key:

Create a .env file (beware that your OS does not append another file extension). Explicitly:

export ANTHROPIC_API_KEY="YOUR KEY"

Save this file to the same directory as this notebook

We now load the .env file and store the API key as an environment variable


```python
load_dotenv(
    ".env"
)  # explicitly state that we are looking for .env in the same directory as notebook
```




    True



Setting up the SimpleBot. The API key will be automatically read from the environment variable


```python
system_prompt = """You are a bot that will respond to questions written by a human curious about using Anthropic's LLMs.
When applicable, you should give an unbiased comparison between the various LLMs.
The post should be written in professional English and in first-person tone for the human.
"""
claude = SimpleBot(
    system_prompt=system_prompt,
    stream_target="stdout",  # this is the default!,
    model_name="claude-3-5-sonnet-20240620",
)
```

Asking a question


```python
first_question = claude("What types of tasks does claude 3.5 sonnet excel at")
```

    As an AI assistant, I'll do my best to provide an informative and unbiased overview of Claude 3.5 Sonnet's capabilities:

    Claude 3.5 Sonnet excels at a wide range of natural language processing tasks. Some key areas where it performs particularly well include:

    1. Text analysis and comprehension: Sonnet has strong reading comprehension abilities and can analyze complex texts across many domains.

    2. Writing and content generation: It can produce high-quality written content in various styles and formats, from creative writing to technical documentation.

    3. Summarization: Sonnet is adept at condensing long texts into concise summaries while retaining key information.

    4. Question answering: It can provide detailed and accurate responses to questions on a broad range of topics.

    5. Language translation: Sonnet can translate between numerous languages with good accuracy.

    6. Code understanding and generation: It has capabilities in understanding and generating code in multiple programming languages.

    7. Task planning and problem-solving: Sonnet can break down complex problems and provide step-by-step solutions or action plans.

    8. Data analysis and interpretation: It can process and interpret structured data, offering insights and explanations.

    While Sonnet is highly capable, it's important to note that its performance can vary depending on the specific task and context. For the most up-to-date and task-specific comparisons with other LLMs, I'd recommend checking Anthropic's official documentation or conducting comparative tests for your particular use case.

    Additionally, Claude 3.5 Sonnet sits between Claude 3 Opus (the most capable model) and Claude 3 Haiku (the fastest model) in terms of capabilities and speed. So for tasks requiring the highest level of performance, Claude 3 Opus might be more suitable, while for tasks prioritizing speed, Claude 3 Haiku could be a better choice.


```python
unsafe_question = claude(
    "Why is it not secure to copy API keys directly into an example Jupyter notebook"
)
```

    As an AI assistant, I'll explain why copying API keys directly into an example Jupyter notebook is not secure:

    Copying API keys directly into a Jupyter notebook is generally considered insecure for several important reasons:

    1. Visibility: Jupyter notebooks are often shared or published, either intentionally or accidentally. If your API key is visible in the notebook, anyone who gains access to it can potentially use your credentials.

    2. Version control risks: If you use version control systems like Git for your notebooks, your API key could end up in your repository history, even if you later remove it. This makes it accessible to anyone with access to the repository.

    3. Lack of encryption: Jupyter notebooks typically store content in plain text, meaning your API key is not encrypted or protected in any way.

    4. Accidental exposure: You might inadvertently share your notebook with colleagues or publish it online without remembering to remove the API key.

    5. Security best practices: It goes against the principle of keeping sensitive information separate from code, which is a fundamental security best practice.

    Instead of directly copying API keys into notebooks, I would recommend using more secure methods such as:

    1. Environment variables: Store your API key as an environment variable and access it in your code.
    2. Configuration files: Use a separate, gitignored configuration file to store sensitive information.
    3. Secret management tools: Utilize dedicated secret management tools or services, especially in production environments.
    4. Jupyter extensions: Some extensions allow you to securely input and use secrets in notebooks without exposing them in the code.

    By using these methods, you can work with API keys more securely while still leveraging the power and convenience of Jupyter notebooks. This approach helps protect your credentials and aligns with best practices in software development and data science.
