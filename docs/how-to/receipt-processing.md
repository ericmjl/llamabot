---
title: Receipt Processing
marimo-version: 0.18.4
width: full
header: |-
  # /// script
  # requires-python = ">=3.10"
  # dependencies = [
  #     "llamabot[all]",
  #     "marimo>=0.17.0",
  #     "pydantic",
  #     "pdf2image",
  # ]
  # ///
---

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ericmjl/llamabot/blob/main/docs/how-to/receipt-processing.py)

To run this notebook, click on the molab shield above or run the following command at the terminal:

```bash
uvx marimo edit --sandbox --mcp --no-token --watch https://github.com/ericmjl/llamabot/blob/main/docs/how-to/receipt-processing.py
```

```python {.marimo}
import marimo as mo
```

## How to Process Receipts with LLM Agents

Learn how to extract structured data from receipt PDFs and images using a two-step
OCR and structuring pattern with llamabot's SimpleBot and StructuredBot.
<!---->
## Prerequisites

Before you begin, ensure you have:

- **Ollama installed and running locally**: Visit [ollama.ai](https://ollama.ai) to install
- **Required Ollama models**:
  - `ollama pull deepseek-ocr` (for OCR text extraction)
  - `ollama pull gemma3n:latest` (for structured output, or another model that supports structured outputs)
- **Python 3.10+** with llamabot and pdf2image installed
- **A receipt PDF or image** to process (or use the example provided)

All llamabot models in this guide use the `ollama/` or `ollama_chat/` prefix for local execution.
<!---->
## Goal

By the end of this guide, you'll have built a receipt processing system that:

- Converts receipt PDFs to images
- Extracts text using vision models (OCR)
- Structures the extracted data into a validated Pydantic model
- Provides observability through spans

```python {.marimo}
from pathlib import Path
import tempfile

from pdf2image import convert_from_path
from pydantic import BaseModel, Field

import llamabot as lmb
from llamabot import get_current_span, span
from llamabot.bot.structuredbot import StructuredBot
from llamabot.components.messages import user
from llamabot.prompt_manager import prompt
```

## Step 1: Define Your Receipt Data Schema

First, define the Pydantic model that represents the structured receipt data.
This schema must be defined before building the extraction agent.

```python {.marimo}
class ReceiptData(BaseModel):
    """Receipt data schema - must be defined BEFORE building extraction agent."""

    vendor: str = Field(..., description="The name of the vendor/merchant")
    date: str = Field(..., description="The transaction date in YYYY-MM-DD format")
    amount: float = Field(
        ..., description="The total amount as a number (without currency symbols)"
    )
    category: str = Field(
        ...,
        description="Business category (e.g., 'Office Supplies', 'Travel', 'Meals', 'Software', 'Equipment')",
    )
    description: str = Field(
        ..., description="Brief description of what was purchased"
    )
```

## Step 2: Create the Two-Step Processing Bots

We use a two-step pattern because vision models like DeepSeek-OCR excel at OCR
but don't necessarily support structured outputs. The solution:

1. **OCR Step** (SimpleBot): Extract text from images using vision models
2. **Structuring Step** (StructuredBot): Convert unstructured text to validated Pydantic models

```python {.marimo}
@prompt("system")
def receipt_extraction_sysprompt() -> str:
    """You are an expert at extracting financial information from receipt and invoice documents.

    Extract the following information accurately:

    - vendor: The name of the vendor/merchant
    - date: The transaction date in YYYY-MM-DD format
    - amount: The total amount as a number (without currency symbols)
    - category: Business category (e.g., "Office Supplies", "Travel", "Meals", "Software", "Equipment")
    - description: Brief description of what was purchased

    If any field is unclear or missing, use your best judgment based on the context.
    For dates, convert any format to YYYY-MM-DD. For amounts, extract only the numerical value.
    """

# Step 1: OCR extraction with DeepSeek-OCR (SimpleBot)
# DeepSeek-OCR doesn't support structured outputs, so we use SimpleBot
ocr_bot = lmb.SimpleBot(
    system_prompt="Extract all text from receipts accurately. "
    "Preserve the structure and include all numbers, dates, and vendor names.",
    model_name="ollama/deepseek-ocr",
    stream_target="none",
)

# Step 2: Structure the data (using a model that supports structured outputs)
receipt_structuring_bot = StructuredBot(
    system_prompt=receipt_extraction_sysprompt(),
    pydantic_model=ReceiptData,
    model_name="ollama_chat/gemma3n:latest",
    stream_target="none",
)
```

## Step 3: Create PDF to Image Converter with Spans

Let's create a function that converts PDFs to images, using spans for observability.

```python {.marimo}
@span
def convert_pdf_to_images(file_path: str):
    """Convert PDF to list of image paths."""
    s = get_current_span()
    s["file_path"] = file_path
    file_extension = Path(file_path).suffix.lower()
    s["file_extension"] = file_extension

    if file_extension == ".pdf":
        images = convert_from_path(file_path, dpi=200)
        image_paths = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_page_{i + 1}.png"
            ) as temp_img:
                image.save(temp_img.name, "PNG")
                image_paths.append(temp_img.name)
        s["page_count"] = len(image_paths)
        s["conversion_success"] = True
        return image_paths
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        s["page_count"] = 1
        s["conversion_success"] = True
        return [file_path]
    else:
        s["conversion_success"] = False
        raise ValueError(f"Unsupported file type: {file_extension}")
```

## Step 4: Process a Receipt

Now let's process a receipt through the complete workflow:

1. Convert PDF to images
2. Extract text with OCR
3. Structure the data

```python {.marimo}
# Example: Process a receipt
# Replace with your own receipt file path
receipt_path = "./receipt_lunch.pdf"  # Or use your own: "/path/to/your/receipt.pdf"

# Step 1: Convert PDF to images
image_paths = convert_pdf_to_images(receipt_path)
print(f"Converted to {len(image_paths)} image(s)")

# Step 2: Extract text with OCR
ocr_texts = []
for image_path in image_paths:
    ocr_response = ocr_bot(
        user("Extract all text from this receipt image.", image_path)
    )
    ocr_texts.append(ocr_response.content)
print(f"Extracted text from {len(ocr_texts)} page(s)")

# Step 3: Structure the extracted text
combined_ocr_text = "\n\n--- Page Break ---\n\n".join(ocr_texts)
receipt_data = receipt_structuring_bot(combined_ocr_text)
print(f"Structured data: {receipt_data.model_dump_json(indent=2)}")
receipt_data
```

## Step 5: View Observability with Spans

Both bots automatically create spans for observability. Let's see what information is tracked.

```python {.marimo}
# Display spans from both bots
print("OCR Bot Spans:")
ocr_bot.spans

print("\n\nReceipt Structuring Bot Spans:")
receipt_structuring_bot.spans
```

The spans show:

- **OCR Bot**: query, model, input_message_count, duration_ms
- **Structuring Bot**: query, model, validation_attempts, validation_success, schema_fields, duration_ms

You can also see nested spans from the `convert_pdf_to_images` function showing:

- file_path, file_extension, page_count, conversion_success

This observability helps you debug issues and understand the workflow execution.
<!---->
## Step 6: Create a Complete Receipt Processing Function

Let's combine everything into a single function that can be used as a tool.

```python {.marimo}
from llamabot.components.tools import tool
```

```python {.marimo}
@tool
def process_receipt(file_path: str, _globals_dict: dict = None) -> str:
    """Process a receipt PDF or image and extract structured data.

    This tool demonstrates that agents can have read access to the local file system.
    Simply provide a file path and the tool will read it from disk.

    :param file_path: Path to the receipt file (PDF, PNG, JPG, or JPEG)
    :param _globals_dict: Internal parameter - automatically injected by AgentBot
    :return: JSON string of extracted receipt data
    """
    # Access current span to add attributes
    s = get_current_span()
    s["file_path"] = file_path

    # Verify the file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Receipt file not found: {file_path}")

    # PDF to image conversion
    image_paths = convert_pdf_to_images(file_path)
    s["page_count"] = len(image_paths)

    if len(image_paths) == 1:
        prompt_text = "Extract all text from this receipt image."
    else:
        prompt_text = (
            f"Extract all text from this {len(image_paths)}-page receipt document."
        )

    # Step 1: OCR extraction - extract text from images
    ocr_texts = []
    for image_path in image_paths:
        ocr_response = ocr_bot(user(prompt_text, image_path))
        ocr_texts.append(ocr_response.content)
    s.log("ocr_completed", pages=len(image_paths))

    # Combine OCR results from all pages
    combined_ocr_text = "\n\n--- Page Break ---\n\n".join(ocr_texts)

    # Step 2: Structure the extracted text according to ReceiptData schema
    result = receipt_structuring_bot(combined_ocr_text)
    s.log("structuring_completed")
    s["vendor"] = result.vendor
    s["amount"] = result.amount

    # Store ReceiptData object in globals for returning to user
    if _globals_dict is not None:
        _globals_dict["receipt_data"] = result

    return result.model_dump_json()
```

## Summary

You've built a receipt processing system that:

- Uses a two-step OCR + structuring pattern
- Leverages vision models for text extraction
- Validates output with Pydantic schemas
- Provides observability through spans
- Can be used as a tool in agent workflows

**Key Takeaways:**

- Define your Pydantic schema first
- Use SimpleBot for vision/OCR tasks
- Use StructuredBot for validated structured outputs
- Use `@span` decorator and `get_current_span()` for manual observability
- Spans automatically track bot operations
- The `@tool` decorator makes functions agent-callable
