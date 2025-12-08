# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]",
#     "marimo>=0.17.0",
#     "pydantic",
#     "pdf2image",
#     "pyzmq",
#     "anthropic",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    return Path, mo


@app.cell
def _():
    import os

    os.environ["TUTORIAL_API_BASE"] = (
        "https://ericmjl--ollama-service-ollamaservice-server.modal.run"
    )
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Building LLM Agents: Workflow-First Approach

    Learn to build practical LLM agents using LlamaBot and Marimo notebooks.
    The most important lesson: **start with workflows, not technology**.

    We'll build a complete back-office automation system through three agents:
    - Receipt processor (extracts data from PDFs)
    - Invoice writer (generates documents)
    - Coordinator (orchestrates both)

    This demonstrates the fundamental pattern: map your boring workflows first,
    build focused agents for specific tasks, then compose them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Setup Verification

    This tutorial uses a modal-hosted Ollama endpoint.
    Make sure you have access to the endpoint URL.
    """)
    return


@app.cell
def _():
    from llamabot import (
        get_current_span,
        span,
    )
    return get_current_span, span


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Marimo Notebooks

    Marimo uses reactive execution:

    - Cells execute automatically when dependencies change
    - Variables cannot be redeclared across cells
    - The notebook forms a directed acyclic graph (DAG)
    - Last expression in a cell is automatically displayed
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 1: Workflow Mapping

    Back-office workflow:
    - Receipt → Extract Data (for expense tracking)
    - Natural Language → Generate Invoice (for billing clients)

    Decision points:
    - What type of document? (receipt, invoice, etc.)
    - What data to extract? (vendor, date, amount, etc.)
    - What format for output? (structured data, formatted invoice)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Workflow Diagram
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
    flowchart TD
    A[PDF Receipt] --> B[Receipt Processor Agent]
    B --> C[Structured Data:<br/>vendor, date, amount,<br/>category, description]
    D[Natural Language<br/>Invoice Request] --> E[Invoice Writer Agent]
    E --> F[Formatted Invoice HTML]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Agent Breakdown

    1. **Receipt Processor**: PDF → structured data (for expense tracking)
    2. **Invoice Writer**: natural language → formatted invoice (for billing clients)
    3. **Coordinator**: orchestrates both agents

    Note that each corresponds to a **testable workflow that we can build first**, upon which we add agentic orchestration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Prerequisites (Critical!)

    Before building agents, you must have:

    1. **Schema definition**: Define your data schema (Pydantic model) BEFORE building extraction agents
    2. **API/storage access**: Ensure access to storage APIs (Notion, database, etc.) BEFORE building agents that store data
    3. **Template/form definition**: Invoice generation requires a template/form structure that AI fills out

    In the interest of time, within this tutorial, we will skip these definitions, but I am happy to show an example from my own personal life (expenses tracking) afterwards.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 2: Receipt Processor Agent

    Extract structured data from receipt PDFs using vision models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Two-Step Process: OCR + Structuring

    **Why two steps?** Vision models like DeepSeek-OCR are excellent at OCR (extracting text from images),
    but they don't support structured outputs (JSON mode with Pydantic schemas).

    **The solution:** Use a two-step process:

    1. **OCR Step** (`SimpleBot` + `ollama/deepseek-ocr`):
       - Extracts all text from receipt images
       - Returns unstructured text
       - Works with vision models that don't support structured outputs

    2. **Structuring Step** (`StructuredBot` + `ollama_chat/gemma3n:latest`):
       - Takes the extracted text
       - Structures it according to the `ReceiptData` schema
       - Returns validated Pydantic model
       - Requires a model that supports structured outputs

    This pattern lets you use specialized OCR models (like DeepSeek-OCR) while still getting
    structured, validated output.
    """)
    return


@app.cell
def _():
    from pydantic import BaseModel
    return (BaseModel,)


@app.cell
def _():
    return


@app.cell
def _():
    import llamabot as lmb
    return (lmb,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def receipt_extraction_sysprompt():
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
    return (receipt_extraction_sysprompt,)


@app.cell
def _():
    from pdf2image import convert_from_path
    return (convert_from_path,)


@app.cell
def _():
    import tempfile
    return (tempfile,)


@app.cell(hide_code=True)
def _(Path, convert_from_path, span, tempfile):
    @span
    def convert_pdf_to_images(file_path: str):
        """Convert PDF to list of image paths."""
        from llamabot import get_current_span

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
    return (convert_pdf_to_images,)


@app.cell
def _():
    from llamabot.components.messages import user
    return (user,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Implementation: Two Bots

    We create two separate bots for the two-step process:
    """)
    return


@app.cell
def _(lmb, os):
    # Step 1: OCR extraction with DeepSeek-OCR (SimpleBot)
    # DeepSeek-OCR doesn't support structured outputs, so we use SimpleBot
    ocr_bot = lmb.SimpleBot(
        system_prompt="Extract all text from receipts accurately. "
        "Preserve the structure and include all numbers, dates, and vendor names.",
        model_name="ollama/deepseek-ocr",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (ocr_bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: Define ReceiptData Schema

    Your task is to add the following fields to this model:
    - vendor: The name of the vendor/merchant (string)
    - date: The transaction date in YYYY-MM-DD format (string)
    - amount: The total amount as a number without currency symbols (float)
    - category: Business category like "Office Supplies", "Travel", "Meals", etc. (string)
    - description: Brief description of what was purchased (string)

    Once you define these fields, the receipt processing agent will be able to
    extract structured data from receipt images and PDFs.
    """)
    return


@app.cell
def _(BaseModel, render_receipt_html):
    class ReceiptDataExercise(BaseModel):
        """Receipt data schema - must be defined BEFORE building extraction agent."""

        pass

        def _repr_html_(self) -> str:
            """Return HTML representation for marimo display."""
            # `render_receipt_html` is defined at the bottom of the notebook!
            return render_receipt_html(
                vendor=self.vendor,
                date=self.date,
                amount=self.amount,
                category=self.category,
                description=self.description,
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: Use ReceiptDataExercise

    Change `ReceiptData` to `ReceiptDataExercise` that you filled out.
    """)
    return


@app.cell
def _(ReceiptData, lmb, os, receipt_extraction_sysprompt):
    # Step 2: Structure the data (using a model that supports structured outputs)
    # This bot takes the OCR text and structures it according to ReceiptData schema
    receipt_structuring_bot = lmb.StructuredBot(
        system_prompt=receipt_extraction_sysprompt(),
        pydantic_model=ReceiptData,
        model_name="ollama_chat/gemma3n:latest",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (receipt_structuring_bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Breaking down receipt processing: Step by step

    Before we create the full tool, let's see how each piece works independently.
    This helps understand what's happening under the hood.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 1 - Convert PDF to images

    First, we need to convert the receipt PDF into images that vision models can process.

    Try this with the provided receipt in the repo (`./receipt_lunch.pdf`), then feel free to experiment with your own receipts by pointing to the right file path on disk.
    """)
    return


@app.cell
def _(convert_pdf_to_images):
    # Exercise: Convert a receipt PDF to images
    # Start with the provided receipt:
    example_file_path = "./receipt_lunch.pdf"

    # After trying with the provided receipt, try your own:
    # example_file_path = "/path/to/your/receipt.pdf"

    image_paths = convert_pdf_to_images(example_file_path)
    print(f"Converted to {len(image_paths)} image(s)")
    return (image_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Exercise: Step 2 - Extract text with OCR

    Next, we use the OCR bot to extract all text from each image.
    The `ocr_bot` uses a vision model to read the receipt.

    Try experimenting with different prompts to see how the results change. Does more specific instruction lead to better extraction? Does a simpler prompt work just as well?
    """)
    return


@app.cell
def _(image_paths, ocr_bot, user):
    # Exercise: Extract text from receipt images using OCR
    # Try different prompts to see what works best:
    ocr_texts = []
    for image_path in image_paths:
        # Try this prompt first:
        prompt_text = "Extract all text from this image."

        # Then experiment with alternatives like:
        # prompt_text = "Read all the text from this receipt, preserving the layout."
        # prompt_text = "Extract the text content from this receipt image."
        # prompt_text = "What text do you see in this image?"

        ocr_response = ocr_bot(user(prompt_text, image_path))
        ocr_texts.append(ocr_response.content)
    print(f"Extracted text from {len(ocr_texts)} page(s)")
    return (ocr_texts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The use of `user(text1, text2)` here is to cast text into a UserMessage, which underneath the hood affords simpler code writing compared to using the raw OpenAI-compatible API.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Step 3: Structure the extracted text

    Finally, we combine the OCR text and use the structuring bot to convert
    it into our `ReceiptData` schema with validated fields.
    """)
    return


@app.cell
def _(ocr_texts, receipt_structuring_bot):
    # Example: Structure the extracted OCR text
    combined_ocr_text = "\n\n--- Page Break ---\n\n".join(ocr_texts)
    result = receipt_structuring_bot(combined_ocr_text)
    print(f"Structured data: {result.model_dump_json(indent=2)}")
    result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4: Store the structured data

    At this point, you have a validated `ReceiptData` object with all the extracted information.
    In a production system, this is where you'd persist the data to your backend storage.

    Common storage patterns include:

    - **Database**: Insert into PostgreSQL, SQLite, or your preferred database
    - **API**: POST to a REST API endpoint (e.g., Notion, Airtable, custom backend)
    - **File system**: Append to CSV, JSONL, or other structured files
    - **Data warehouse**: Stream to BigQuery, Snowflake, or similar systems

    Your schema (`ReceiptData`) is already defined, so storage becomes
    a straightforward serialization step. The Pydantic model ensures data consistency
    before it reaches your storage layer.

    For this tutorial, we'll skip the storage step and focus on the agent orchestration,
    but remember: **workflow-first means thinking about where data flows next**.
    """)
    return


@app.cell
def _():
    from llamabot.components.pocketflow import nodeify
    from llamabot.components.tools import tool
    return nodeify, tool


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Now let's tie it all together: The receipt processing tool

    The `process_receipt` tool combines all three steps into a single function
    that agents can call. It handles file validation, PDF conversion, OCR extraction,
    and data structuring automatically.

    **Note**: This tool demonstrates that agents can have read access to the local file system.
    Simply provide a file path and the tool will read it directly from disk.

    You'll notice the function includes span-based logging, a newer feature in llamabot. Think of spans as breadcrumbs that track what your code is doing - they record when operations start and finish, what data flows through them, and any important details along the way. You can mostly ignore them while learning, but they become incredibly handy when debugging complex agent workflows. When something goes wrong, instead of wondering "what just happened?", you can trace back through the spans to see exactly what your agent was doing at each step.

    The three decorators stacked on this function each serve a specific purpose. The `@span` decorator handles that logging I just mentioned - it automatically creates a trace record whenever the function runs, capturing timing and any metadata you want to attach. The `@tool` decorator transforms the function into something agents can understand and call autonomously - it reads the docstring and type hints to generate a tool description that the LLM uses to decide when to invoke it. Finally, `@nodeify(loopback_name="decide")` integrates the function into the agent's workflow graph, specifying that after this tool completes, control should loop back to a node called "decide" where the agent chooses its next action. Together, these decorators turn a regular Python function into an observable, agent-callable workflow step.
    """)
    return


@app.cell
def _(
    Path,
    convert_pdf_to_images,
    get_current_span,
    nodeify,
    ocr_bot,
    receipt_structuring_bot,
    span,
    tool,
    user,
):
    @nodeify(loopback_name="decide")
    @tool
    @span
    def process_receipt(file_path: str) -> str:
        """Process a receipt PDF or image and extract structured data.

        This tool demonstrates that agents can have read access to the local file system.
        Simply provide a file path and the tool will read it from disk.

        :param file_path: Path to the receipt file (PDF, PNG, JPG, or JPEG)
        :return: JSON string of extracted receipt data
        """
        # This is a LlamaBot feature for logging, ignore this!
        s = get_current_span()
        s["file_path"] = file_path

        # Verify the file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Receipt file not found: {file_path}")

        # PDF to image conversion (convert_pdf_to_images is @span decorated)
        image_paths = convert_pdf_to_images(file_path)
        s["page_count"] = len(image_paths)

        if len(image_paths) == 1:
            prompt_text = "Extract all text from this receipt image."
        else:
            prompt_text = f"Extract all text from this {len(image_paths)}-page receipt document."

        # Step 1: OCR extraction - extract text from images
        # Process each image and combine the results (ocr_bot creates spans automatically)
        ocr_texts = []
        for image_path in image_paths:
            ocr_response = ocr_bot(user(prompt_text, image_path))
            ocr_texts.append(ocr_response.content)
        s.log("ocr_completed", pages=len(image_paths))

        # Combine OCR results from all pages
        combined_ocr_text = "\n\n--- Page Break ---\n\n".join(ocr_texts)

        # Step 2: Structure the extracted text according to ReceiptData schema
        # (receipt_structuring_bot creates spans automatically)
        result = receipt_structuring_bot(combined_ocr_text)
        s.log("structuring_completed")
        s["vendor"] = result.vendor
        s["amount"] = result.amount

        return result.model_dump_json()
    return (process_receipt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Test: Receipt Processor

    Now, let's test the LLM tool `process_receipt`, so that we can verify that it is working.
    """)
    return


@app.cell
def _(ReceiptData, process_receipt):
    # Test receipt processor with uploaded file
    test_receipt_json = process_receipt("./receipt_lunch.pdf")
    # Parse JSON back to ReceiptData object for display
    test_receipt_data = ReceiptData.model_validate_json(test_receipt_json)
    test_receipt_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: Display Bot Spans

    Display spans from the bot using the .spans property of ocr_bot:
    """)
    return


@app.cell
def _(ocr_bot):
    # Your answer goes here
    ocr_bot.spans
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 3: Invoice Writer Agent

    Generate formatted invoices from structured data.
    Invoice generation is like filling out a form - you need the form structure first.
    """)
    return


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def invoice_generation_sysprompt():
        """You are a professional invoice generation assistant.
        Fill out invoice forms with structured data provided.
        Ensure all fields are professional and business-appropriate.
        """
    return (invoice_generation_sysprompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: Define InvoiceData Schema

    Just as we defined the structure for receipt data, now we need to define what an invoice looks like. Your task is to create the `InvoiceDataExercise` model by adding these fields:

    - `invoice_number`: The invoice identifier like "INV-2025-001" (string)
    - `client_name`: The name of the client being billed (string)
    - `client_address`: The client's full mailing address (string)
    - `issue_date`: The date the invoice is issued in YYYY-MM-DD format (string)
    - `due_date`: The payment due date in YYYY-MM-DD format (string)
    - `project_description`: Description of the work or services provided (string)
    - `amount`: The total amount to be paid (float)
    - `notes`: Optional additional notes or payment instructions (string, optional with default empty string)

    Think of this as defining the form template that the invoice generation agent will fill out.
    """)
    return


@app.cell
def _(BaseModel, render_invoice_html):
    class InvoiceDataExercise(BaseModel):
        """Invoice data schema - form structure for invoice generation."""

        pass

        def _repr_html_(self) -> str:
            """Return HTML representation for marimo display."""
            # `render_invoice_html` is defined at the bottom of the notebook!
            return render_invoice_html(
                invoice_number=self.invoice_number,
                client_name=self.client_name,
                client_address=self.client_address,
                issue_date=self.issue_date,
                due_date=self.due_date,
                project_description=self.project_description,
                amount=self.amount,
                notes=self.notes,
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: Use InvoiceDataExercise

    Now that you've filled out the `InvoiceDataExercise` model, update the `invoice_writer_bot` to use your exercise version instead of the pre-built `InvoiceData`. Replace `InvoiceData` with `InvoiceDataExercise` in the pydantic_model parameter below.
    """)
    return


@app.cell
def _(InvoiceData, invoice_generation_sysprompt, lmb, os):
    invoice_writer_bot = lmb.StructuredBot(
        system_prompt=invoice_generation_sysprompt(),
        pydantic_model=InvoiceData,
        model_name="ollama_chat/gemma3n:latest",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (invoice_writer_bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tying it all together: The invoice generation workflow

    Now that we have our invoice writer bot configured, let's combine the pieces into a complete workflow. The `generate_invoice` function takes a natural language description of what to invoice and returns a structured `InvoiceData` object. It handles the date calculations automatically, so you don't need to worry about computing business days or formatting dates correctly. The invoice writer bot does the heavy lifting of extracting client information, project details, and amounts from your description.
    """)
    return


@app.cell
def _(InvoiceData, get_current_span, invoice_writer_bot, span):
    @span
    def generate_invoice(invoice_description: str) -> InvoiceData:
        """Generate invoice from natural language description.

        :param invoice_description: Natural language description of the invoice to generate.
            Should include client name, project description, amount, and any other relevant details.
        """
        s = get_current_span()
        s["invoice_description"] = invoice_description[
            :200
        ]  # Truncate for storage

        from datetime import datetime, timedelta

        # Calculate today's date and 30 business days from today
        today = datetime.now().date()

        # Calculate 30 business days (excluding weekends)
        business_days_added = 0
        due_date = today
        while business_days_added < 30:
            due_date += timedelta(days=1)
            # Check if it's a weekday (Monday=0, Sunday=6)
            if due_date.weekday() < 5:  # Monday-Friday
                business_days_added += 1

        prompt = f"""Generate an invoice based on this description:
        {invoice_description}

        IMPORTANT DATE RULES:
        - issue_date: MUST be {today.strftime("%Y-%m-%d")} (today's date)
        - due_date: MUST be {due_date.strftime("%Y-%m-%d")} (30 business days from today, which is {today.strftime("%Y-%m-%d")})

        Generate a professional invoice number in format INV-YYYY-XXX.
        Extract client information, project details, and amount from the description.
        """

        invoice = invoice_writer_bot(prompt)
        s["invoice_number"] = invoice.invoice_number
        s["amount"] = invoice.amount
        return invoice
    return (generate_invoice,)


@app.cell(hide_code=True)
def _(InvoiceData, get_current_span, span):
    @span
    def format_invoice_html(invoice: InvoiceData) -> str:
        """Render invoice as beautiful HTML.

        This function delegates to InvoiceData._repr_html_() for consistency.
        """
        s = get_current_span()
        html = invoice._repr_html_()
        s["invoice_number"] = invoice.invoice_number
        s["html_length"] = len(html)
        return html
    return (format_invoice_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Creating the invoice writing tool

    With our invoice generation workflow ready, we wrap it in a `write_invoice` tool that agents can call. You'll notice a few things that might look unusual at first. The `_globals_dict` parameter is automatically injected by AgentBot and allows the tool to store generated data (like the invoice HTML) in a shared space where other tools or the agent can access it later. This is how we pass complex objects between tool calls without serializing everything to strings.

    The docstring here is deliberately verbose and detailed because it serves double duty. When the agent decides which tool to call, it reads this docstring as the tool's description. The more context you provide about when to use the tool, what information it needs, and what it returns, the better the agent can decide whether this is the right tool for the user's request. Think of it as writing instructions for the agent, not just documentation for developers.
    """)
    return


@app.cell
def _(format_invoice_html, generate_invoice, nodeify, span, tool):
    @nodeify(loopback_name="decide")
    @tool
    @span
    def write_invoice(invoice_description: str, _globals_dict: dict = None) -> str:
        """Generate and render an invoice from natural language description.

        IMPORTANT: Only call this tool when you have sufficient information from the user.
        The invoice_description should include:
        - Client name and address
        - Project description
        - Amount
        - Due date (optional, defaults to 30 days from today)

        If the user's request is vague (e.g., "create an invoice"), ask for clarification first
        before calling this tool.

        :param invoice_description: Natural language description of the invoice to generate.
            Must include client name, project description, and amount at minimum.
            Example: "Invoice for Acme Corp at 123 Main St, Boston MA 02101, web development project, $5000, due in 30 days"
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Confirmation message indicating invoice was generated
        """
        # Generate invoice (generate_invoice and format_invoice_html are @span decorated)
        invoice = generate_invoice(invoice_description)
        html = format_invoice_html(invoice)

        # Store invoice HTML in globals so it can be returned to user
        if _globals_dict is not None:
            _globals_dict["invoice_html"] = html

        return (
            "Invoice generated successfully. "
            "**YOU MUST NOW**: Call return_object_to_user('invoice_html') immediately to return it to the user."
        )
    return (write_invoice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Test: Invoice Writer

    Test invoice generation from natural language:
    """)
    return


@app.cell
def _(format_invoice_html, generate_invoice):
    # Test invoice writer with natural language description
    # mo.stop(True)
    # Uncomment to test:
    test_description = "Invoice for Acme Corporation, web development project completed in January 2025, amount $5000, client address: 123 Main St, Boston MA 02101"
    test_invoice = generate_invoice(test_description)
    format_invoice_html(test_invoice)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 4: Back-Office Coordinator Agent

    Compose agents by making them tools for other agents.
    The coordinator decides when to call each specialized agent.
    """)
    return


@app.cell(hide_code=True)
def _(lmb):
    @lmb.prompt("system")
    def coordinator_sysprompt():
        """You are a back-office coordinator agent.
        You help process receipts and generate invoices for clients.

        **CRITICAL**: You MUST always select a tool. Never return empty tool calls.
        Every user query requires a tool to be executed.

        **CRITICAL**: After ANY tool executes, you MUST immediately call another tool (or respond_to_user if truly done).
        Tool results are instructions to continue, NOT completion signals.
        If a tool result says "Use X()" or "Call X()", you MUST call X() immediately - do not stop.
        If a tool stores data in globals, you MUST call another tool to return it - do not stop.

        **CRITICAL**: Most user requests require MULTIPLE tool calls to complete.
        Do NOT stop after a single tool call - continue calling tools until the request is fully satisfied.
        Only use `respond_to_user()` after you have completed ALL necessary tool calls.

        ## Multi-Step Tool Execution:

        **When to call multiple tools sequentially:**
        - Most requests require 2-3+ tool calls to complete (e.g., check files → process → respond)
        - When you need to gather information before acting (e.g., list files → process receipt → respond)
        - When you need to perform an action then return results (e.g., query database → format results → respond)
        - When a tool stores data in globals that needs to be returned (e.g., generate invoice → return HTML)

        **When NOT to use `respond_to_user()` immediately:**
        - If you just called a tool that stores results in globals → Call another tool to return those results
        - If you need to check resources first → Call checking tool, then action tool, then respond
        - If the user's request has multiple parts → Complete ALL parts before responding
        - If a tool tells you to call another tool → Do it immediately, don't respond yet

        **When to use `respond_to_user()`:**
        - After completing ALL steps in a multi-step workflow
        - After a tool stores data in globals and you've returned it to the user
        - When you need to ask the user for clarification (but only after checking what's available first)
        - After a single tool fully satisfies a simple request (rare)

        **Examples of multi-step workflows:**
        - "Process this receipt" → process_receipt("/path/to/file.pdf") → respond_to_user()
        - "Create an invoice" → write_invoice() → return_object_to_user('invoice_html') → respond_to_user()

        CRITICAL: Use as many tool calls as needed to support the user's request excellently.
        Do NOT assume information - always check what's available first, then ask for clarification if needed.

        Workflow Guidelines:

        1. **Receipt Processing**:
           - When user asks to "process this receipt" or provides a file path:
             STEP 1: Call process_receipt(file_path) with the file path provided by the user
             STEP 2: After processing completes, use respond_to_user() to summarize the extracted data
           - The tool reads files directly from disk - provide the full file path as a string
           - If the user doesn't provide a file path, ask them for it using respond_to_user()

        2. **Invoice Generation** (MULTI-STEP REQUIRED):
           **CRITICAL**: Each invoice request is INDEPENDENT. NEVER reuse information from previous requests unless explicitly stated by the user.

           **REQUIRED Information Checklist** - Before calling write_invoice(), you MUST verify ALL of these are present:
           - ✅ Client name (specific name, NOT "my client", "the client", or vague references)
           - ✅ Client address (full address with street, city, state, zip)
           - ✅ Project description (specific description of work/services)
           - ✅ Amount (specific dollar amount, NOT ranges or estimates)

           **Date Handling (AUTOMATIC - Do NOT ask user for dates):**
           - **Issuance date**: ALWAYS today's date (calculated automatically by the tool)
           - **Due date**: ALWAYS 30 BUSINESS DAYS (not calendar days) from the issuance date (calculated automatically by the tool)
           - The write_invoice tool will automatically handle ALL date calculations - you do NOT need to provide dates

           **If ANY required information is missing or vague:**
           STEP 1: Use respond_to_user() to ask SPECIFICALLY for the missing information.
           DO NOT make assumptions or use information from previous requests.
           Example: "I need a few details to create this invoice:
           - What is the client's full name?
           - What is the client's complete address (street, city, state, zip)?
           - What is the specific amount for this invoice?
           - What is the project description or services provided?"
           STEP 2: Wait for user's response with ALL required details

           **Only if ALL required information is explicitly provided:**
           STEP 1: Call write_invoice() with the complete invoice description including ALL required fields (client name, address, project description, amount).
           STEP 2: IMMEDIATELY call return_object_to_user('invoice_html') to return the HTML to the user
           STEP 3: Call respond_to_user() to confirm completion

           **Examples of INSUFFICIENT requests that require clarification:**
           - "Create an invoice for my client" → Missing: client name, address, amount, project description
           - "Create an invoice for consulting services" → Missing: client name, address, amount
           - "Generate an invoice for $5000" → Missing: client name, address, project description
           - "Create an invoice for Acme Corp" → Missing: address, amount, project description

           **Example of SUFFICIENT request:**
           - "Create an invoice for Acme Corp at 123 Main St, Boston MA 02101, web development project, $5000"

           **NEVER:**
           - Reuse client names, amounts, or addresses from previous requests
           - Assume "my client" refers to a previous client
           - Generate invoices with partial information
           - Make up or estimate missing information

        3. **General Principles**:
           - Break complex requests into multiple tool calls
           - Check available resources before assuming
           - Ask clarifying questions when information is missing
           - Explain what you're doing at each step
           - Use return_object_to_user() to return generated content (like invoice_html)

        Remember: It's better to ask for clarification than to assume and generate incorrect information.
        """
    return (coordinator_sysprompt,)


@app.cell
def _():
    from llamabot import AgentBot
    return (AgentBot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Introducing AgentBot

    AgentBot is LlamaBot's implementation of the agentic workflow pattern. Unlike SimpleBot or StructuredBot that just respond to prompts, AgentBot can make decisions about which tools to use and orchestrate multi-step workflows. You give it a set of tools (like our `process_receipt` and `write_invoice` functions) and a system prompt that explains its role, and it figures out which tool to call based on what the user asks for.

    Under the hood, AgentBot uses a loop-decide-execute pattern. It reads the user's request, decides which tool (if any) to call, executes that tool, sees the result, and then decides what to do next. This continues until it determines the user's request is fully satisfied. You can read more about AgentBot's architecture and capabilities in the LlamaBot documentation at https://llamabot.readthedocs.io/, though be aware the docs are still catching up with recent features.
    """)
    return


@app.cell
def _(AgentBot, coordinator_sysprompt, os, process_receipt, write_invoice):
    coordinator_bot = AgentBot(
        tools=[write_invoice, process_receipt],
        system_prompt=coordinator_sysprompt(),
        model_name="ollama_chat/deepseek-r1:32b",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (coordinator_bot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing the agent workflow

    AgentBots have a special `_repr_html_` method that renders a different view than other bots. When you display an AgentBot in a marimo notebook, you'll see a mermaid diagram showing the workflow graph. The blue nodes represent tools that yield control back to the agent for further decision-making (like `process_receipt` and `write_invoice`), while green nodes represent terminal tools that send responses directly back to the user (like `respond_to_user`). This visualization helps you understand the agent's decision flow at a glance.
    """)
    return


@app.cell
def _(coordinator_bot):
    coordinator_bot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Enabling file uploads

    The `files` widget allows users to upload receipts through the marimo GUI. When someone uploads a file, it gets loaded into memory and saved to a temporary location on disk. The agent can then access these files by their file paths, which we make available in the notebook's globals. This is essential for the receipt processing workflow, since users need a way to provide their receipt PDFs to the agent without manually typing file paths.
    """)
    return


@app.cell
def _(mo):
    files = mo.ui.file(filetypes=[".pdf", ".png", ".jpg", ".jpeg"])
    return (files,)


@app.cell
def _(files):
    files
    return


@app.cell
def _(files):
    display_file = None
    if files.value:
        display_file = files.value[0]
    display_file
    return


@app.cell
def _(Path, files, tempfile):
    if files.value:
        for file in files.value:
            print(f"## Processing: {file.name}")
            # Save file temporarily
            file_extension = Path(file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as temp_file:
                temp_file.write(file.contents)
                temp_file_path = temp_file.name

            # Make file available in globals
            variable_name = (
                Path(file.name).stem.replace(" ", "_").replace("-", "_")
            )
            globals()[variable_name] = temp_file_path
            print(f"File available as: {variable_name}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The chat turn pattern

    The `chat_turn` function follows marimo's standard pattern for `mo.ui.chat`. Marimo expects a function that takes a list of messages and a config object, then returns the next response. Inside the function, we extract the user's latest message from `messages[-1].content`, pass it to our coordinator bot, and return the result. The bot automatically handles the conversation state.

    We wrap the coordinator bot call in a `span` context manager here to track each chat interaction. This creates a trace record that captures the user's message and the bot's response, making it easier to debug issues or analyze conversation patterns later. The span's category label helps filter traces when you're looking at logs from multiple sources.
    """)
    return


@app.cell
def _(coordinator_bot, datetime, span):
    def chat_turn(messages, config):
        user_message = messages[-1].content
        with span(
            "coordinator_chat_turn",
            user_message=user_message[:100],
            category="chat_interaction",
        ):
            result = coordinator_bot(
                [f"Today's date: {datetime.now()}", user_message], globals()
            )
        return result
    return (chat_turn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example prompts

    The `example_prompts` list provides suggested starting points for users interacting with the chat interface. Marimo displays these as clickable suggestions when the chat is empty, helping users understand what the agent can do and how to phrase their requests. Think of them as both a user guide and a way to jumpstart conversations when someone isn't sure where to begin.
    """)
    return


@app.cell
def _(chat_turn, mo):
    example_prompts = [
        "Process this receipt and extract the data",
        "Generate an invoice for Acme Corp, web development project, $5000. The company's located at 123 Boylston Street, Boston, MA, 01234.",
        "What information did you extract from the receipt?",
        "Create an invoice for my client for consulting services",
    ]

    chat = mo.ui.chat(chat_turn, max_height=600, prompts=example_prompts)
    return (chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: Testing the limits

    Now comes the important part. Try to break the back-office coordinator. Give it ambiguous requests, edge cases, or instructions that don't quite fit the expected patterns. What happens when you ask for an invoice without providing all the required information? What if you upload a receipt that's hard to read or in an unexpected format? What if you chain multiple requests together in confusing ways?

    The failure modes you discover aren't bugs to be ashamed of - they're valuable data points. Each one tells you something about where your agent needs better guardrails, clearer instructions, or more sophisticated error handling. In a real deployment, you'd want to collect these failures systematically and build evaluations around them, but that's challenging when you're building something new and don't yet know what the common failure patterns will be. For now, just observe what breaks and think about whether each failure represents a fixable prompt engineering issue, a missing tool capability, or an inherent limitation of the approach.
    """)
    return


@app.cell
def _(chat, mo):
    mo.vstack(
        [
            mo.md("### Back-Office Coordinator"),
            mo.md("Upload receipts and chat with the coordinator agent."),
            chat,
        ]
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: View Coordinator Spans

    View the `coordinator_bot.spans` to see what the bot chose to do.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Test: Coordinator Agent

    Test the coordinator with a sample query:
    """)
    return


@app.cell
def _(coordinator_bot):
    # Test coordinator agent
    test_response = coordinator_bot(
        "Process a receipt and generate an invoice", globals()
    )
    test_response
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we get something like "Please provide the file path of the receipt you'd like me to process.", that's actually desired behaviour!
    """)
    return


@app.cell
def _(coordinator_bot):
    coordinator_bot.spans
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### How the Coordinator Works

    The coordinator uses AgentBot which:
    1. Analyzes your request
    2. Decides which tool to use (process_receipt or write_invoice)
    3. Executes the tool
    4. Loops back to decide next action (via `loopback_name="decide"`)
    5. Continues until task is complete

    This is the agent-as-tool pattern: agents use other agents as tools.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 6: Discussion & Next Steps

    ### Moving from Notebook to Production

    - **Local vs Cloud**: Use modal endpoints for production, local Ollama for development
    - **Error Handling**: Add try/except blocks and validation
    - **Storage**: Connect to real databases (Notion, PostgreSQL, etc.)
    - **Templates**: Store invoice templates in files or databases
    - **Monitoring**: Add logging and observability
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Deploying LLMs on Modal

    This tutorial uses a Modal-hosted Ollama endpoint for demonstration.
    To deploy your own LLM hosting infrastructure on Modal:

    **Repository**: [ollama-on-modal](https://github.com/ericmjl/ollama-on-modal)

    This repository contains:
    - Complete setup instructions for hosting Ollama models on Modal
    - Pre-configured deployments for popular models
    - Cost-effective cloud hosting for LLM inference
    - Production-ready infrastructure patterns

    **Why Modal?**
    - Pay-per-use pricing (no idle costs)
    - Automatic scaling
    - Easy deployment with Python
    - GPU access for fast inference

    Follow the repository's README to set up your own Modal endpoint and update
    the `TUTORIAL_API_BASE` environment variable in this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Best Practices

    1. **Workflow-first**: Always map workflows before coding
    2. **Schema-first**: Define data models before extraction
    3. **Template-first**: Design forms/templates before generation
    4. **Compose agents**: Build focused agents, then compose them
    5. **Test incrementally**: Test each agent before composing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Q&A

    Common questions:
    - How do I add new agents? → Build them as tools, add to AgentBot
    - How do I debug agent decisions? → Check AgentBot's decision logs
    - How do I handle errors? → Wrap tools in try/except, return error messages
    - How do I scale? → Use modal endpoints, batch processing, async operations
    """)
    return


@app.cell
def _():
    import textwrap


    def render_invoice_html(
        invoice_number: str,
        client_name: str,
        client_address: str,
        issue_date: str,
        due_date: str,
        project_description: str,
        amount: float,
        notes: str = "",
    ) -> str:
        """Render invoice data as HTML.

        :param invoice_number: Invoice identifier
        :param client_name: Client name
        :param client_address: Client address
        :param issue_date: Issue date
        :param due_date: Due date
        :param project_description: Project description
        :param amount: Invoice amount
        :param notes: Optional notes
        :return: HTML string
        """
        return textwrap.dedent(f"""
            <div style="max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3); overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center;">
                    <div style="font-size: 14px; opacity: 0.9; text-transform: uppercase; letter-spacing: 2px;">Invoice</div>
                    <div style="font-size: 36px; font-weight: 700; letter-spacing: -1px; margin-bottom: 8px;">{invoice_number}</div>
                </div>
                <div style="padding: 40px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea;">
                            <h3 style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #6c757d; margin-bottom: 12px; font-weight: 600; margin: 0 0 12px 0;">Issue Date</h3>
                            <p style="font-size: 18px; font-weight: 600; color: #495057; margin: 0;">{issue_date}</p>
                        </div>
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea;">
                            <h3 style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #6c757d; margin-bottom: 12px; font-weight: 600; margin: 0 0 12px 0;">Due Date</h3>
                            <p style="font-size: 18px; font-weight: 600; color: #495057; margin: 0;">{due_date}</p>
                        </div>
                    </div>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 30px;">
                        <h3 style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #6c757d; margin-bottom: 12px; font-weight: 600; margin: 0 0 12px 0;">Bill To</h3>
                        <p style="font-size: 16px; color: #212529; line-height: 1.6; margin: 0;">{client_name}<br>{client_address}</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 30px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #28a745;">
                        <h3 style="font-size: 18px; font-weight: 600; color: #212529; margin-bottom: 12px; margin: 0 0 12px 0;">Project Description</h3>
                        <p style="font-size: 16px; color: #495057; line-height: 1.7; margin: 0;">{project_description}</p>
                    </div>
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; text-align: center; margin-bottom: 30px;">
                        <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 2px; opacity: 0.9; margin-bottom: 8px;">Amount Due</div>
                        <div style="font-size: 48px; font-weight: 700; letter-spacing: -2px;">${amount:,.2f}</div>
                    </div>
                    {f'<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; border-radius: 8px; margin-top: 20px;"><strong style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #856404; display: block; margin-bottom: 8px;">Notes</strong><p style="color: #856404; line-height: 1.6; margin: 0;">{notes}</p></div>' if notes else ""}
                </div>
            </div>
            """).strip()


    def render_receipt_html(
        vendor: str,
        date: str,
        amount: float,
        category: str,
        description: str,
    ) -> str:
        """Render receipt data as HTML.

        :param vendor: Vendor name
        :param date: Receipt date
        :param amount: Receipt amount
        :param category: Expense category
        :param description: Receipt description
        :return: HTML string
        """
        return textwrap.dedent(f"""
            <div style="max-width: 700px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; border-left: 4px solid #10b981;">
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 20px 24px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;">Receipt Entry</div>
                            <div style="font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">{vendor}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;">Amount</div>
                            <div style="font-size: 28px; font-weight: 700; letter-spacing: -1px;">${amount:,.2f}</div>
                        </div>
                    </div>
                </div>
                <div style="padding: 24px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;">
                        <div style="background: #f8f9fa; padding: 16px; border-radius: 6px; border-left: 3px solid #10b981;">
                            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #6c757d; margin-bottom: 6px; font-weight: 600;">Date</div>
                            <div style="font-size: 16px; font-weight: 600; color: #212529;">{date}</div>
                        </div>
                        <div style="background: #f8f9fa; padding: 16px; border-radius: 6px; border-left: 3px solid #10b981;">
                            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #6c757d; margin-bottom: 6px; font-weight: 600;">Category</div>
                            <div style="font-size: 16px; font-weight: 600; color: #212529;">{category}</div>
                        </div>
                    </div>
                    <div style="background: #f8f9fa; padding: 16px; border-radius: 6px; border-left: 3px solid #10b981;">
                        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #6c757d; margin-bottom: 8px; font-weight: 600;">Description</div>
                        <div style="font-size: 15px; color: #495057; line-height: 1.5;">{description}</div>
                    </div>
                </div>
            </div>
            """).strip()
    return render_invoice_html, render_receipt_html


@app.cell
def _(BaseModel, render_invoice_html):
    class InvoiceData(BaseModel):
        """Invoice data schema - form structure for invoice generation."""

        invoice_number: str
        client_name: str
        client_address: str
        issue_date: str
        due_date: str
        project_description: str
        amount: float
        notes: str = ""

        def _repr_html_(self) -> str:
            """Return HTML representation for marimo display."""
            return render_invoice_html(
                invoice_number=self.invoice_number,
                client_name=self.client_name,
                client_address=self.client_address,
                issue_date=self.issue_date,
                due_date=self.due_date,
                project_description=self.project_description,
                amount=self.amount,
                notes=self.notes,
            )


    # Test instance to verify display
    example_invoice = InvoiceData(
        invoice_number="INV-2025-001",
        client_name="Acme Corporation",
        client_address="123 Main St, Boston MA 02101",
        issue_date="2025-01-15",
        due_date="2025-02-14",
        project_description="Web development and consulting services",
        amount=5000.00,
        notes="Payment due within 30 days of invoice date.",
    )
    example_invoice
    return (InvoiceData,)


@app.cell
def _(BaseModel, render_receipt_html):
    class ReceiptData(BaseModel):
        """Receipt data schema - must be defined BEFORE building extraction agent."""

        vendor: str
        date: str
        amount: float
        category: str
        description: str

        def _repr_html_(self) -> str:
            """Return HTML representation for marimo display."""
            return render_receipt_html(
                vendor=self.vendor,
                date=self.date,
                amount=self.amount,
                category=self.category,
                description=self.description,
            )


    # Test instance to verify display
    example_receipt = ReceiptData(
        vendor="Starbucks Coffee",
        date="2025-01-15",
        amount=12.50,
        category="Meals",
        description="Team lunch meeting - coffee and pastries",
    )
    example_receipt
    return (ReceiptData,)


if __name__ == "__main__":
    app.run()
