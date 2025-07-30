# How to use the @prompt decorator

The `@prompt` decorator is LlamaBot's powerful tool for creating reusable, version-controlled prompt templates. This guide shows you how to use it effectively in your projects.

## What is the @prompt decorator?

The `@prompt` decorator transforms Python functions into Jinja2-templated prompts with automatic version control. Instead of writing prompts as static strings, you can create dynamic, reusable prompt templates that adapt to different inputs.

## Basic usage

### Creating your first prompt

```python
from llamabot import prompt

@prompt(role="user")
def explain_concept(concept, audience):
    """Please explain {{ concept }} to {{ audience }} in simple terms.

    Use examples and analogies that {{ audience }} would understand.
    Break down complex ideas into digestible pieces.
    """
```

The decorator uses your function's docstring as a Jinja2 template. Variables in double curly braces `{{ variable }}` are replaced with function arguments.

### Using the prompt

The `@prompt` decorated function is still a regular Python function that you can call directly. When you call it, it returns a message object with the rendered Jinja2 template:

```python
# Create a prompt message
message = explain_concept("machine learning", "high school students")

# The message object contains:
# - role: "user"
# - content: The rendered template
# - prompt_hash: A unique hash for version tracking
print(message.content)
```

**What happens when you call the function:**

1. The function arguments (`"machine learning"` and `"high school students"`) are passed to the Jinja2 template
2. The template variables `{{ concept }}` and `{{ audience }}` are replaced with the actual values
3. The docstring is rendered as a complete prompt string
4. A message object is returned containing the rendered content

**Example of the rendered output:**
```python
message = explain_concept("machine learning", "high school students")
print(message.content)
# Output:
# "Please explain machine learning to high school students in simple terms.
#
# Use examples and analogies that high school students would understand.
# Break down complex ideas into digestible pieces."
```

The key insight is that **the decorated function executes normally** - it takes your arguments, interpolates them into the docstring template, and returns the final rendered prompt as a message object.

## Understanding message roles

The `@prompt` decorator supports three message roles:

### System prompts
Use `role="system"` for instructions that define the AI's behavior:

```python
@prompt(role="system")
def helpful_assistant():
    """You are a helpful AI assistant. Always provide accurate,
    clear answers and ask for clarification when needed."""
```

### User prompts
Use `role="user"` for messages that simulate user input:

```python
@prompt(role="user")
def ask_question(topic):
    """I need help understanding {{ topic }}.
    Can you explain it step by step?"""
```

### Assistant prompts
Use `role="assistant"` for few-shot examples or assistant responses:

```python
@prompt(role="assistant")
def example_response(task, solution):
    """For the task "{{ task }}", here's how I approach it:

    {{ solution }}

    This demonstrates the structured thinking process."""
```

## Advanced templating

### Complex data structures

Your prompts can accept complex data types:

```python
@prompt(role="user")
def code_review(code_snippet, issues):
    """Please review this code:

    ```python
    {{ code_snippet }}
    ```

    Focus on these areas:
    {% for issue in issues %}
    - {{ issue }}
    {% endfor %}
    """

# Usage
issues = ["performance", "readability", "security"]
message = code_review("def hello(): print('world')", issues)
```

### Conditional content

Use Jinja2 conditionals for dynamic content:

```python
@prompt(role="user")
def write_documentation(function_name, include_examples=True):
    """Write documentation for the {{ function_name }} function.

    {% if include_examples %}
    Include practical examples showing how to use it.
    {% endif %}

    Follow standard docstring conventions."""
```

### Template inheritance patterns

Create base prompts and extend them:

```python
@prompt(role="system")
def base_coding_assistant(language):
    """You are an expert {{ language }} programmer.
    Always write clean, well-documented code."""

@prompt(role="user")
def debug_code(code, error_message, language="Python"):
    """I'm having trouble with this {{ language }} code:

    {{ code }}

    Error: {{ error_message }}

    Please help me fix it."""
```

## Integration with bots

### Using prompts with SimpleBot

```python
from llamabot import SimpleBot, prompt

@prompt(role="system")
def python_tutor():
    """You are a patient Python tutor who explains concepts clearly."""

@prompt(role="user")
def explain_topic(topic, skill_level):
    """Explain {{ topic }} to someone with {{ skill_level }} Python experience."""

# Create bot with system prompt
bot = SimpleBot(
    system_prompt=python_tutor(),
    model_name="gpt-3.5-turbo"
)

# Use user prompt
response = bot(explain_topic("decorators", "beginner"))
```

### Building prompt libraries

Organize related prompts in modules:

```python
# prompts/python_help.py
from llamabot import prompt

@prompt(role="user")
def explain_error(error_type, error_message):
    """I got this {{ error_type }} error:

    {{ error_message }}

    Can you explain what caused it and how to fix it?"""

@prompt(role="user")
def optimize_code(code, optimization_goal):
    """Please optimize this Python code for {{ optimization_goal }}:

    {{ code }}

    Explain the improvements you made."""
```

## Automatic version control

### How versioning works

LlamaBot automatically tracks changes to your prompts:

```python
@prompt(role="user")
def greeting(name):
    """Hello {{ name }}!"""  # Version 1

# Later, you modify the prompt:
@prompt(role="user")
def greeting(name):
    """Hello {{ name }}! How are you today?"""  # Version 2 (new hash)
```

Each unique prompt template gets a hash that's stored in a local SQLite database. When you modify a prompt, LlamaBot creates a new version while preserving the history.

### Accessing version information

```python
message = greeting("Alice")
print(f"Prompt hash: {message.prompt_hash}")
```

The prompt hash helps you track which version of a prompt generated specific results.

## Working with experiments

### Tracking prompt experiments

```python
from llamabot import Experiment, prompt, SimpleBot

@prompt(role="system")
def creative_writer(style):
    """You are a creative writer who writes in {{ style }} style."""

@prompt(role="user")
def write_story(theme, length):
    """Write a {{ length }} story about {{ theme }}."""

with Experiment(name="story_generation") as exp:
    # Different prompt versions are automatically tracked
    bot = SimpleBot(creative_writer("mystery"))
    story = bot(write_story("time travel", "short"))
```

### A/B testing prompts

```python
@prompt(role="user")
def prompt_v1(topic):
    """Explain {{ topic }} briefly."""

@prompt(role="user")
def prompt_v2(topic):
    """Provide a comprehensive explanation of {{ topic }} with examples."""

# Test both versions
with Experiment(name="explanation_styles") as exp:
    bot = SimpleBot("You are a helpful teacher.")

    # Version 1
    response_v1 = bot(prompt_v1("photosynthesis"))

    # Version 2
    response_v2 = bot(prompt_v2("photosynthesis"))

    # Compare results in log viewer
```

## Best practices

### Design effective prompts

1. **Be specific**: Include clear instructions about format, tone, and expectations
2. **Use examples**: Show the AI what good output looks like
3. **Test iterations**: Use the version control to experiment with different approaches
4. **Validate variables**: Ensure all template variables are provided as function parameters

### Template validation

The decorator automatically validates that all template variables have corresponding function parameters:

```python
@prompt(role="user")
def broken_prompt(name):
    """Hello {{ name }}! Your {{ age }} is showing."""  # Error: 'age' not in parameters

# Fix by adding the parameter:
@prompt(role="user")
def fixed_prompt(name, age):
    """Hello {{ name }}! Your {{ age }} is showing."""
```

### Error handling

```python
@prompt(role="user")
def safe_prompt(required_param, optional_param=None):
    """Required: {{ required_param }}

    {% if optional_param %}
    Optional: {{ optional_param }}
    {% endif %}
    """

# This works
message = safe_prompt("value1")

# This also works
message = safe_prompt("value1", "value2")
```

## Common patterns

### Few-shot examples

```python
@prompt(role="user")
def classification_with_examples(text, categories):
    """Classify the following text into one of these categories:
    {% for category in categories %}
    - {{ category }}
    {% endfor %}

    Examples:
    Text: "The weather is sunny today"
    Category: Weather

    Text: "I love this new restaurant"
    Category: Food

    Text: "{{ text }}"
    Category:"""
```

### Chain of thought prompting

```python
@prompt(role="user")
def solve_step_by_step(problem):
    """Solve this problem step by step:

    {{ problem }}

    Think through this carefully:
    1. What information do I have?
    2. What do I need to find?
    3. What steps should I take?
    4. What is the final answer?
    """
```

### Multi-turn conversations

```python
@prompt(role="system")
def conversation_system():
    """You are a helpful assistant in a multi-turn conversation.
    Remember context from previous messages."""

@prompt(role="user")
def follow_up(previous_topic, new_question):
    """Earlier we discussed {{ previous_topic }}.
    Now I want to ask: {{ new_question }}"""
```

## Troubleshooting

### Common issues

**Missing template variables**: Ensure all `{{ variable }}` references have corresponding function parameters.

**Jinja2 syntax errors**: Check your template syntax, especially with loops and conditionals.

**Import errors**: Make sure to import `prompt` from `llamabot`:
```python
from llamabot import prompt
```

**Role validation**: Use only "system", "user", or "assistant" as role values.

### Debugging prompts

```python
@prompt(role="user")
def debug_prompt(data):
    """Debug data: {{ data }}"""

# Check the rendered content
message = debug_prompt("test value")
print(message.content)  # See exactly what was generated
print(message.prompt_hash)  # Track the version
```

## Next steps

- Explore the built-in prompt library in `llamabot.prompt_library`
- Use the log viewer to analyze your prompt experiments
- Integrate prompts with QueryBot for document-based interactions
- Build custom prompt libraries for your specific use cases

The `@prompt` decorator is your gateway to systematic prompt engineering. Start with simple templates and gradually build more sophisticated prompt libraries as you discover what works best for your applications.
