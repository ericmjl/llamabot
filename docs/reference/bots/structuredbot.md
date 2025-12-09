# StructuredBot API Reference

StructuredBot is designed for getting structured, validated outputs from LLMs. It enforces Pydantic schema validation and provides automatic retry logic when the LLM doesn't produce valid output.

## Class Definition

```python
class StructuredBot(SimpleBot):
    """StructuredBot is given a Pydantic Model and expects the LLM to return
    a JSON structure that conforms to the model schema.
    It will validate the returned json against the pydantic model,
    prompting the LLM to fix any of the validation errors if it does not validate,
    and then explicitly return an instance of that model.
    """
```

## Constructor

```python
def __init__(
    self,
    system_prompt: Union[str, SystemMessage],
    pydantic_model: BaseModel,
    model_name: str = default_language_model(),
    stream_target: str = "stdout",
    allow_failed_validation: bool = False,
    **completion_kwargs,
)
```

### Constructor Parameters

- **system_prompt** (`Union[str, SystemMessage]`): The system prompt to use
  for the bot. Should instruct the LLM on how to extract or generate
  structured data.

- **pydantic_model** (`BaseModel`): The Pydantic model that defines the
  expected output schema. The LLM must return JSON that validates against
  this model.

- **model_name** (`str`, default: `default_language_model()`): The name of
  the model to use. Must support structured outputs (e.g., `gpt-4o`,
  `anthropic/claude-3-5-sonnet`, `gemini/gemini-1.5-pro-latest`). See
  [model support](#model-support) for details.

- **stream_target** (`str`, default: `"stdout"`): The target to stream the
  response to. StructuredBot streams only to stdout; other modes may not
  work correctly.

- **allow_failed_validation** (`bool`, default: `False`): Whether to allow
  returning invalid data if validation fails after retries. If `False`,
  raises `ValidationError` on failure.

- **completion_kwargs**: Additional keyword arguments to pass to the completion function.

### Model Support

StructuredBot requires models that support both `response_format` and
`response_schema` parameters. Supported models include:

- `gpt-4o`, `gpt-4-turbo`, `gpt-4`
- `anthropic/claude-3-5-sonnet`, `anthropic/claude-3-opus`
- `gemini/gemini-1.5-pro-latest`
- `ollama_chat/*` (with structured output support)

If a model doesn't support structured outputs, `StructuredBot` will raise a `ValueError` at initialization.

## Methods

### `__call__`

```python
def __call__(
    self,
    *messages: Union[str, BaseMessage],
) -> BaseModel
```

Process messages and return a validated Pydantic model instance.

#### Parameters

- **messages**: One or more messages to process. Can be strings or `BaseMessage` objects.

#### Returns

- **BaseModel**: An instance of the provided `pydantic_model` with validated data.

#### Raises

- **ValidationError**: If validation fails after retries and `allow_failed_validation=False`.

#### Example

```python
import llamabot as lmb
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

bot = lmb.StructuredBot(
    system_prompt="Extract person information from text.",
    pydantic_model=Person,
    model_name="gpt-4o"
)

person = bot("John is 25 years old. Email: john@example.com")
print(person.name)  # "John"
print(person.age)   # 25
```

## Retry Logic

StructuredBot automatically retries when validation fails:

1. LLM generates JSON response
2. JSON is validated against Pydantic model
3. If validation fails, error details are sent back to LLM
4. LLM attempts to fix the JSON
5. Process repeats up to a maximum number of retries
6. If still invalid and `allow_failed_validation=False`, raises `ValidationError`

## Attributes

- **pydantic_model** (`BaseModel`): The Pydantic model for validation
- **allow_failed_validation** (`bool`): Whether to allow failed validation

## Usage Examples

### Basic Data Extraction

```python
import llamabot as lmb
from pydantic import BaseModel
from typing import List

class Person(BaseModel):
    name: str
    age: int
    hobbies: List[str]

bot = lmb.StructuredBot(
    system_prompt="Extract person information from text.",
    pydantic_model=Person,
    model_name="gpt-4o"
)

person = bot("John is 25 years old and enjoys hiking and photography.")
print(person.name)      # "John"
print(person.age)       # 25
print(person.hobbies)   # ["hiking", "photography"]
```

### Complex Nested Models

```python
import llamabot as lmb
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    address: Address
    created_at: datetime

bot = lmb.StructuredBot(
    system_prompt="Extract person information with address.",
    pydantic_model=Person,
    model_name="gpt-4o"
)

person = bot("John, 25, lives at 123 Main St, New York, 10001")
print(person.address.city)  # "New York"
```

### With Optional Fields

```python
import llamabot as lmb
from pydantic import BaseModel
from typing import Optional

class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

bot = lmb.StructuredBot(
    system_prompt="Extract person information.",
    pydantic_model=Person,
    model_name="gpt-4o"
)

# Works even if email is missing
person = bot("John is 25 years old")
print(person.email)  # None
```

### Allowing Failed Validation

```python
import llamabot as lmb
from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    name: str
    age: int

bot = lmb.StructuredBot(
    system_prompt="Extract person information.",
    pydantic_model=Person,
    model_name="gpt-4o",
    allow_failed_validation=True
)

# If validation fails, returns partial data instead of raising
try:
    person = bot("Invalid text")
except ValidationError:
    # Handle validation error
    pass
```

## Differences from SimpleBot JSON Mode

- **StructuredBot**: Guarantees schema validation, returns Pydantic objects, automatic retries
- **SimpleBot JSON Mode**: Ensures valid JSON only, no schema validation, returns strings

## Best Practices

1. **Use descriptive system prompts**: Clearly explain what data to extract
2. **Define clear schemas**: Use Pydantic's validation features (validators, constraints)
3. **Handle optional fields**: Use `Optional` for fields that may be missing
4. **Test with edge cases**: Ensure your schema handles various input formats

## Related Classes

- **SimpleBot**: Base class that StructuredBot extends
- **Pydantic BaseModel**: Schema definition class

## See Also

- [StructuredBot Tutorial](../tutorials/structuredbot.md)
- [Which Bot Should I Use?](../getting-started/which-bot.md)
- [Pydantic Documentation](https://docs.pydantic.dev/)
