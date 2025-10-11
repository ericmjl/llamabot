# StructuredBot Tutorial

Welcome to the StructuredBot tutorial! In this tutorial, we will learn how to use the `StructuredBot` class to get validated, structured outputs from LLMs using Pydantic models.

## What is StructuredBot?

StructuredBot is designed for scenarios where you need **guaranteed structured outputs** from LLMs. Unlike SimpleBot, StructuredBot:

- **Enforces Pydantic schema validation** on all responses
- **Automatically retries** when the LLM produces invalid output
- **Returns validated Pydantic objects** instead of raw text
- **Provides clear error messages** when validation fails

This makes StructuredBot perfect for data extraction, API responses, form processing, and any scenario where you need reliable structured data.

## Prerequisites

Before you begin, ensure you have the following:

- Basic knowledge of Python programming
- Familiarity with Pydantic models
- Access to a Python environment with the necessary libraries installed

## Installation

First, ensure you have the `llamabot` library installed:

```bash
pip install llamabot
```

## Basic Usage

### Step 1: Define Your Pydantic Model

Start by creating a Pydantic model that defines the structure you want:

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None
    hobbies: List[str] = []
    created_at: datetime
```

### Step 2: Create a StructuredBot

```python
import llamabot as lmb
from datetime import datetime

# Create a StructuredBot with your Pydantic model
bot = lmb.StructuredBot(
    system_prompt="Extract person information from the given text. Always include a created_at timestamp.",
    pydantic_model=Person,
    model_name="gpt-4o"
)
```

### Step 3: Use the Bot

```python
# The bot will return a validated Person object
person = bot("John Smith is 25 years old and enjoys hiking, photography, and cooking. His email is john@example.com.")

print(person.name)        # "John Smith"
print(person.age)         # 25
print(person.email)       # "john@example.com"
print(person.hobbies)     # ["hiking", "photography", "cooking"]
print(person.created_at)  # datetime object
```

## Advanced Features

### Validation and Retry Logic

StructuredBot automatically handles validation failures by retrying with the LLM:

```python
# If the LLM produces invalid output, StructuredBot will:
# 1. Show the validation error to the LLM
# 2. Ask it to fix the output
# 3. Retry up to the maximum number of attempts
# 4. Raise an error if all attempts fail

try:
    person = bot("Invalid input that might confuse the model")
except ValidationError as e:
    print(f"Validation failed after retries: {e}")
```

### Custom Validation Rules

You can add custom validation to your Pydantic models:

```python
from pydantic import BaseModel, validator
from typing import List

class Product(BaseModel):
    name: str
    price: float
    category: str
    tags: List[str]

    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

    @validator('category')
    def category_must_be_valid(cls, v):
        valid_categories = ['electronics', 'clothing', 'books', 'home']
        if v.lower() not in valid_categories:
            raise ValueError(f'Category must be one of: {valid_categories}')
        return v.lower()

# Create bot with custom validation
bot = lmb.StructuredBot(
    system_prompt="Extract product information from text.",
    pydantic_model=Product,
    model_name="gpt-4o"
)
```

### Complex Nested Models

StructuredBot works with complex nested structures:

```python
from pydantic import BaseModel
from typing import List, Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Company(BaseModel):
    name: str
    industry: str
    address: Address
    employees: int

class Employee(BaseModel):
    name: str
    position: str
    salary: float
    company: Company
    skills: List[str]

# Create bot for complex nested data
bot = lmb.StructuredBot(
    system_prompt="Extract employee information including company details.",
    pydantic_model=Employee,
    model_name="gpt-4o"
)

employee = bot("""
Sarah Johnson works as a Senior Software Engineer at TechCorp,
a technology company in the software industry.
She earns $95,000 per year and has skills in Python, JavaScript, and React.
The company is located at 123 Tech Street, San Francisco, CA 94105
and has 150 employees.
""")
```

## Configuration Options

### Retry Behavior

```python
bot = lmb.StructuredBot(
    system_prompt="Extract data from text.",
    pydantic_model=YourModel,
    model_name="gpt-4o",
    allow_failed_validation=False,  # Default: False (retry on validation failure)
    max_retries=3,                  # Default: 3 retries
    temperature=0.1                 # Lower temperature for more consistent outputs
)
```

### Streaming

StructuredBot supports streaming for real-time feedback:

```python
bot = lmb.StructuredBot(
    system_prompt="Extract data from text.",
    pydantic_model=YourModel,
    stream_target="stdout"  # Stream to console
)
```

## Common Use Cases

### 1. Data Extraction from Documents

```python
class Invoice(BaseModel):
    invoice_number: str
    date: datetime
    total_amount: float
    vendor: str
    line_items: List[dict]

bot = lmb.StructuredBot(
    system_prompt="Extract invoice information from the document.",
    pydantic_model=Invoice,
    model_name="gpt-4o"
)

invoice = bot(invoice_document_text)
```

### 2. API Response Processing

```python
class APIResponse(BaseModel):
    status: str
    data: dict
    error_message: Optional[str] = None
    timestamp: datetime

bot = lmb.StructuredBot(
    system_prompt="Parse API response and extract structured data.",
    pydantic_model=APIResponse,
    model_name="gpt-4o"
)

response = bot(api_response_text)
```

### 3. Form Data Validation

```python
class ContactForm(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    message: str
    urgency: str  # "low", "medium", "high"

bot = lmb.StructuredBot(
    system_prompt="Extract and validate contact form information.",
    pydantic_model=ContactForm,
    model_name="gpt-4o"
)

form_data = bot(user_submitted_text)
```

## Best Practices

### 1. Design Clear Schemas

```python
# Good: Clear, specific fields
class UserProfile(BaseModel):
    full_name: str
    email: str
    age: int
    interests: List[str]

# Avoid: Vague or overly complex schemas
class BadProfile(BaseModel):
    info: dict  # Too vague
    data: Any   # Too flexible
```

### 2. Use Appropriate Types

```python
from typing import Optional, List, Union
from datetime import datetime

class Event(BaseModel):
    title: str
    start_time: datetime
    duration_minutes: int
    attendees: List[str]
    is_online: bool
    location: Optional[str] = None
```

### 3. Add Helpful Validation

```python
from pydantic import BaseModel, validator

class Product(BaseModel):
    name: str
    price: float
    category: str

    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
```

### 4. Handle Edge Cases

```python
# Use Optional fields for data that might not be present
class Article(BaseModel):
    title: str
    content: str
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    tags: List[str] = []
```

## Troubleshooting

### Common Issues

1. **Validation Errors**: Check your Pydantic model for type mismatches
2. **Retry Failures**: Ensure your system prompt is clear about the expected format
3. **Complex Nested Data**: Start with simpler models and gradually add complexity

### Debug Mode

```python
import llamabot as lmb

# Enable debug mode to see validation attempts
lmb.set_debug_mode(True)

bot = lmb.StructuredBot(
    system_prompt="Extract data from text.",
    pydantic_model=YourModel,
    model_name="gpt-4o"
)
```

## Comparison with SimpleBot

| Feature | SimpleBot | StructuredBot |
|---------|-----------|---------------|
| **Output Type** | Raw text | Validated Pydantic objects |
| **Validation** | None | Automatic Pydantic validation |
| **Retry Logic** | None | Automatic retry on validation failure |
| **Type Safety** | No | Yes (Pydantic models) |
| **Use Case** | General conversation | Structured data extraction |

## Conclusion

StructuredBot provides a powerful way to get reliable, validated structured outputs from LLMs. By combining Pydantic models with automatic validation and retry logic, StructuredBot ensures that your applications receive data in the exact format you expect.

Key takeaways:
- Use StructuredBot when you need guaranteed structured outputs
- Design clear, well-validated Pydantic models
- Leverage automatic retry logic for robust data extraction
- Combine with appropriate system prompts for best results

For more advanced usage patterns and examples, check out the other bot tutorials in the LlamaBot documentation.
