# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]",
#     "marimo>=0.17.0",
#     "pydantic",
#     "pdf2image",
#     "pyzmq",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


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


@app.cell
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


@app.cell
def _(mo):
    mo.md("""
    ## Setup Verification

    This tutorial uses a modal-hosted Ollama endpoint.
    Make sure you have access to the endpoint URL.

    For local development, you can use `gpt-4.1` or another vision-capable model.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Span-Based Observability

    This tutorial demonstrates span-based logging for observability.
    Spans automatically track bot operations, tool executions, and decision-making.
    We'll also add manual spans to custom workflows for complete trace visibility.
    """)
    return


@app.cell
def _():
    from llamabot import (
        get_current_span,
        get_span_tree,
        get_spans,
        span,
    )
    return get_current_span, get_span_tree, get_spans, span


@app.cell
def _(mo):
    mo.md("""
    ## Marimo Notebooks

    Marimo uses reactive execution:

    - Cells execute automatically when dependencies change
    - Variables cannot be redeclared across cells
    - The notebook forms a directed acyclic graph (DAG)
    - Last expression in a cell is automatically displayed
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


@app.cell
def _(mo):
    mo.md("""
    ### Workflow Diagram
    """)
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md("""
    ### Agent Breakdown

    1. **Receipt Processor**: PDF → structured data (for expense tracking)
    2. **Invoice Writer**: natural language → formatted invoice (for billing clients)
    3. **Coordinator**: orchestrates both agents
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Prerequisites (Critical!)

    Before building agents, you must have:

    1. **Schema definition**: Define your data schema (Pydantic model) BEFORE building extraction agents
    2. **API/storage access**: Ensure access to storage APIs (Notion, database, etc.) BEFORE building agents that store data
    3. **Template/form definition**: Invoice generation requires a template/form structure that AI fills out

    **Workflow-first means requirements-first.**
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
    ## Part 2: Receipt Processor Agent

    Extract structured data from receipt PDFs using vision models.
    """)
    return


@app.cell
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
def _(BaseModel):
    class ReceiptData(BaseModel):
        """Receipt data schema - must be defined BEFORE building extraction agent."""

        vendor: str
        date: str
        amount: float
        category: str
        description: str
    return (ReceiptData,)


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
        model_name="ollama/deepseek-ocr",  # Fixed: ollama/deepseek-ocr
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (ocr_bot,)


@app.cell
def _(ReceiptData, lmb, os, receipt_extraction_sysprompt):
    # Step 2: Structure the data (using a model that supports structured outputs)
    # This bot takes the OCR text and structures it according to ReceiptData schema
    receipt_structuring_bot = lmb.StructuredBot(
        system_prompt=receipt_extraction_sysprompt(),
        pydantic_model=ReceiptData,
        model_name="ollama_chat/gemma3n:latest",  # Or use "gpt-4.1" if available
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (receipt_structuring_bot,)


@app.cell
def _(mo):
    mo.md("""
    ### Putting It Together: The Receipt Processing Tool

    The `process_receipt` tool orchestrates the two-step process:
    - Converts PDFs to images (if needed)
    - Runs OCR on each image using `ocr_bot`
    - Combines OCR results
    - Structures the combined text using `receipt_structuring_bot`

    **Note**: This tool demonstrates that agents can have read access to the local file system.
    Simply provide a file path and the tool will read it directly from disk.
    """)
    return


@app.cell
def _():
    from llamabot.components.pocketflow import nodeify
    from llamabot.components.tools import tool
    return nodeify, tool


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


@app.cell
def _(mo):
    mo.md("""
    ### Test: Receipt Processor

    Upload a receipt PDF or image above, then test the receipt processor:
    """)
    return


@app.cell
def _(ReceiptData, process_receipt):
    # Test receipt processor with uploaded file
    # mo.stop(True)
    # Uncomment and modify to test:
    test_receipt_json = process_receipt(
        "/Users/ericmjl/github/llamabot/tutorials/pydata-boston-2025/receipt_lunch.pdf"
    )
    # Parse JSON back to ReceiptData object for display
    test_receipt_data = ReceiptData.model_validate_json(test_receipt_json)
    test_receipt_data
    return


@app.cell
def _(ocr_bot):
    ocr_bot
    return


@app.cell
def _(get_spans):
    get_spans(operation_name="process_receipt")
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
    ## Part 3: Invoice Writer Agent

    Generate formatted invoices from structured data.
    Invoice generation is like filling out a form - you need the form structure first.
    """)
    return


@app.cell
def _(BaseModel):
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
    return (InvoiceData,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def invoice_generation_sysprompt():
        """You are a professional invoice generation assistant.
        Fill out invoice forms with structured data provided.
        Ensure all fields are professional and business-appropriate.
        """
    return (invoice_generation_sysprompt,)


@app.cell
def _(InvoiceData, invoice_generation_sysprompt, lmb, os):
    invoice_writer_bot = lmb.StructuredBot(
        system_prompt=invoice_generation_sysprompt(),
        pydantic_model=InvoiceData,
        model_name="ollama_chat/gemma3n:latest",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (invoice_writer_bot,)


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

        prompt = f"""Generate an invoice based on this description:
        {invoice_description}

        Use today's date as issue_date and 30 days from today as due_date.
        Generate a professional invoice number in format INV-YYYY-XXX.
        Extract client information, project details, and amount from the description.
        """

        invoice = invoice_writer_bot(prompt)
        s["invoice_number"] = invoice.invoice_number
        s["amount"] = invoice.amount
        return invoice
    return (generate_invoice,)


@app.cell
def _(InvoiceData, get_current_span, span):
    @span
    def render_invoice_html(invoice: InvoiceData) -> str:
        """Render invoice as beautiful HTML."""
        s = get_current_span()
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .invoice-header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
                .invoice-number {{ font-size: 24px; font-weight: bold; }}
                .invoice-details {{ margin: 20px 0; }}
                .invoice-details table {{ width: 100%; border-collapse: collapse; }}
                .invoice-details td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .invoice-details td:first-child {{ font-weight: bold; width: 150px; }}
                .amount {{ font-size: 32px; font-weight: bold; color: #2563eb; margin: 20px 0; }}
                .project-description {{ margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="invoice-header">
                <div class="invoice-number">Invoice {invoice.invoice_number}</div>
            </div>
            <div class="invoice-details">
                <table>
                    <tr><td>Issue Date:</td><td>{invoice.issue_date}</td></tr>
                    <tr><td>Due Date:</td><td>{invoice.due_date}</td></tr>
                    <tr><td>Bill To:</td><td>{invoice.client_name}<br>{invoice.client_address}</td></tr>
                </table>
            </div>
            <div class="project-description">
                <h3>Project Description</h3>
                <p>{invoice.project_description}</p>
            </div>
            <div class="amount">Amount Due: ${invoice.amount:,.2f}</div>
            {f'<div class="notes"><p><strong>Notes:</strong> {invoice.notes}</p></div>' if invoice.notes else ""}
        </body>
        </html>
        """
        s["invoice_number"] = invoice.invoice_number
        s["html_length"] = len(html)
        return html
    return (render_invoice_html,)


@app.cell
def _(generate_invoice, nodeify, render_invoice_html, span, tool):
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
        # Generate invoice (generate_invoice and render_invoice_html are @span decorated)
        invoice = generate_invoice(invoice_description)
        html = render_invoice_html(invoice)

        # Store invoice HTML in globals so it can be returned to user
        if _globals_dict is not None:
            _globals_dict["invoice_html"] = html

        return (
            "Invoice generated successfully. "
            "**YOU MUST NOW**: Call return_object_to_user('invoice_html') immediately to return it to the user, "
            "then call respond_to_user() to confirm completion."
        )
    return (write_invoice,)


@app.cell
def _(mo):
    mo.md("""
    ### Test: Invoice Writer

    Test invoice generation from natural language:
    """)
    return


@app.cell
def _(generate_invoice, mo, render_invoice_html):
    # Test invoice writer with natural language description
    # mo.stop(True)
    # Uncomment to test:
    test_description = "Invoice for Acme Corporation, web development project completed in January 2025, amount $5000, client address: 123 Main St, Boston MA 02101"
    test_invoice = generate_invoice(test_description)
    test_invoice
    mo.Html(render_invoice_html(test_invoice))
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


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def coordinator_sysprompt():
        """You are a back-office coordinator agent.
        You help process receipts, generate invoices for clients, and handle internal concerns.

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
        - "What's in the database?" → query_complaints() → respond_to_user() (with formatted results)

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
           - Do NOT generate invoices with assumed/made-up data
           - If the user asks to "create an invoice" without sufficient details:
             STEP 1: Use respond_to_user() to ask for missing information:
               * Client name and address
               * Project description
               * Amount
               * Due date (optional, defaults to 30 days)
             STEP 2: Wait for user's next message with the details
           - If the user provides sufficient details:
             STEP 1: Call write_invoice() with the invoice description
             STEP 2: IMMEDIATELY call return_object_to_user('invoice_html') to return the HTML to the user
           - Do NOT stop after write_invoice() - you MUST also return the invoice HTML

        3. **Internal Concerns** (CONVERSATIONAL MULTI-STEP WORKFLOW):
           - **Act as a listening partner**: Your goal is to help the user express their concern fully
             so that when anonymized, it will be useful and actionable for someone reading it.

           - **When user first shares a concern**:
             STEP 1: Assess if the concern has enough actionable information:
               * Does it identify WHAT the issue is? (e.g., "deployment process", "communication breakdown")
               * Does it identify WHY it's a problem? (e.g., "team lead hasn't responded", "causing delays")
               * Does it identify IMPACT? (e.g., "blocking my work", "affecting team morale")

             STEP 2A: If information is INSUFFICIENT (vague, too brief, missing context):
               * Use respond_to_user() to ask empathetic, open-ended questions to draw out more details
               * Examples of good questions:
                 - "I hear you're concerned about [topic]. Can you help me understand what specifically is happening?"
                 - "What impact is this having on your work or the team?"
                 - "Is there a particular situation or example that illustrates this concern?"
                 - "What would help resolve this, or what would you like to see happen?"
               * Wait for user's response, then reassess

             STEP 2B: If information is SUFFICIENT (clear issue, context, and impact):
               * Construct the full conversation history as a formatted string by combining:
                 - The user's initial concern (from the conversation memory)
                 - Your clarifying questions (what you asked via respond_to_user calls)
                 - The user's responses to your questions (from subsequent conversation turns)
                 - Any additional context they shared
               * Format it naturally as a dialogue, e.g.:
                 "User: I'm worried about our deployment process.
                 Agent: Can you help me understand what specifically is happening?
                 User: The team lead hasn't responded to my questions about the new workflow.
                 Agent: What impact is this having on your work?
                 User: It's blocking my progress and causing delays."
               * IMPORTANT: Include the full conversation, not just the initial concern.
                 The anonymization will be more useful if it includes the context from your questions.
               * Call anonymize_complaint() with this formatted conversation history string
               * IMMEDIATELY call respond_to_user() to show the anonymized version for review

             - **After showing structured version**:
               * Use respond_to_user() to ask: "Does this structured version capture your concern accurately?
                 It includes the issue, details, impact, and recommended actions. If yes, I can store it.
                 If you'd like to add anything or clarify any section, please let me know."
               * Wait for user confirmation

           - **After user confirms storage** (e.g., "yes", "store it", "that looks good"):
             STEP 1: Call confirm_store_complaint() to store the anonymized complaint
             STEP 2: IMMEDIATELY call respond_to_user() to confirm storage and thank them

           - **Key principles for handling concerns**:
             * Be empathetic and supportive - you're helping them express something important
             * Ask ONE question at a time - don't overwhelm with multiple questions
             * Focus on drawing out WHAT, WHY, and IMPACT - these make concerns actionable
             * Don't rush to anonymize - take time to understand the full picture
             * Once anonymized, always show it for review before storing

        4. **Querying Complaints** (MULTI-STEP REQUIRED):
           - When user asks to "show stored complaints" or "what's in the database":
             STEP 1: Call query_complaints() with an appropriate query (or empty query to get all)
             STEP 2: IMMEDIATELY call respond_to_user() to present the results from 'complaint_query_results' in globals
           - Do NOT stop after query_complaints() - you MUST use respond_to_user() to return the results
           - Format the results nicely for the user (you can summarize, categorize, or list them)

        5. **General Principles**:
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


@app.cell
def _(AgentBot, coordinator_sysprompt, os, process_receipt, write_invoice):
    coordinator_bot = AgentBot(
        tools=[process_receipt, write_invoice],
        system_prompt=coordinator_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (coordinator_bot,)


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


@app.cell
def _(coordinator_bot, span):
    def chat_turn(messages, config):
        user_message = messages[-1].content
        with span(
            "coordinator_chat_turn",
            user_message=user_message[:100],
            category="chat_interaction",
        ):
            result = coordinator_bot(user_message, globals())
        return result
    return (chat_turn,)


@app.cell
def _(chat_turn, mo):
    example_prompts = [
        "Process this receipt and extract the data",
        "Generate an invoice for Acme Corp, web development project, $5000",
        "What information did you extract from the receipt?",
        "Create an invoice for my client for consulting services",
    ]

    chat = mo.ui.chat(chat_turn, max_height=600, prompts=example_prompts)
    return (chat,)


@app.cell
def _(chat, mo):
    mo.vstack(
        [
            mo.md("# Back-Office Coordinator"),
            mo.md("Upload receipts and chat with the coordinator agent."),
            chat,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Test: Coordinator Agent

    Test the coordinator with a sample query:
    """)
    return


@app.cell
def _(mo):
    # Test coordinator agent
    mo.stop(True)
    # Uncomment to test:
    # test_response = coordinator_bot("Process a receipt and generate an invoice", globals())
    # test_response
    return


@app.cell
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
    ## Span Visualization & Observability

    Now that we've enabled span recording, let's explore the spans that were automatically created.
    Spans provide complete trace visibility into the agent workflow:

    - **Automatic spans**: Created by bots (SimpleBot, StructuredBot, AgentBot) and tools
    - **Manual spans**: Created by our custom workflow functions
    - **Nested structure**: Spans form a hierarchy showing the complete execution flow

    The trace hierarchy looks like:
    `coordinator_chat_turn → agentbot_call → decision → tool_call → process_receipt → convert_pdf_to_images → ocr_bot → receipt_structuring_bot`
    """)
    return


@app.cell
def _(get_spans, mo):
    # Query all spans
    all_spans = get_spans()
    mo.md(f"**Total spans recorded:** {len(all_spans)}")
    return (all_spans,)


@app.cell
def _(all_spans):
    all_spans
    return


@app.cell
def _(get_spans, mo):
    # Query spans by category
    receipt_spans = get_spans(category="receipt_processing")
    invoice_spans = get_spans(category="invoice_generation")
    tool_spans = get_spans(operation_name="tool_call")

    mo.md(f"""
    **Spans by category:**
    - Receipt processing: {len(receipt_spans)}
    - Invoice generation: {len(invoice_spans)}
    - Tool executions: {len(tool_spans)}
    """)
    return


@app.cell
def _(all_spans, get_span_tree, mo):
    # Display span tree for the most recent trace
    if all_spans:
        # Get the most recent trace_id
        trace_id = all_spans[-1]["trace_id"]
        span_tree = get_span_tree(trace_id)

        tree_display = mo.md(f"""
        **Span Tree for Trace:** `{trace_id}`

        ```
        {span_tree}
        ```
        """)
    else:
        tree_display = mo.md(
            "No spans found. Run some operations first to see spans."
        )

    tree_display
    return


@app.cell
def _(all_spans, mo):
    # Show span statistics
    if all_spans:
        categories = {}
        operations = {}
        total_duration = 0

        for span_record in all_spans:
            # Count by category
            category = span_record.get("attributes", {}).get(
                "category", "uncategorized"
            )
            categories[category] = categories.get(category, 0) + 1

            # Count by operation
            op_name = span_record.get("operation_name", "unknown")
            operations[op_name] = operations.get(op_name, 0) + 1

            # Sum durations
            if span_record.get("duration_ms"):
                total_duration += span_record["duration_ms"]

        category_list = "\n".join(
            [f"- {k}: {v}" for k, v in sorted(categories.items())]
        )
        operation_list = "\n".join(
            [f"- {k}: {v}" for k, v in sorted(operations.items())[:10]]
        )

        stats_display = mo.md(f"""
        **Span Statistics:**

        **By Category:**
        {category_list}

        **By Operation (top 10):**
        {operation_list}

        **Total Duration:** {total_duration:.2f} ms
        """)
    else:
        stats_display = mo.md(
            "No spans found. Run some operations first to see statistics."
        )

    stats_display
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
    ## Part 5: Bonus - Internal Complaints Agent

    Pre-built agent for handling internal company concerns:
    - Anonymizes chat history
    - Shows anonymized version for review
    - Stores in vector database with date partition (after confirmation)
    - Queries and summarizes complaints by category
    """)
    return


@app.cell
def _():
    from llamabot.components.docstore import LanceDBDocStore
    return (LanceDBDocStore,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def anonymization_sysprompt():
        """You are an expert at anonymizing and structuring internal company concerns.

        Your task is to transform a conversation about a concern into a clear, structured format
        that emphasizes the issue and actionable items. Do NOT preserve the verbatim conversation format.

        **Anonymization Requirements:**
        - Remove all personally identifiable information (names, emails, phone numbers, specific departments)
        - Replace with generic placeholders like [EMPLOYEE], [DEPARTMENT], [TEAM], etc.
        - Preserve the core concern content and sentiment

        **Output Format:**
        Structure the anonymized concern using this format:

        **Issue:** [Brief, clear statement of the core problem or concern]

        **Details:** [Specific context about what's happening, root causes, or contributing factors.
        Synthesize information from the conversation into a coherent narrative, not verbatim quotes.]

        **Impact:** [How this issue is affecting work, team, productivity, deadlines, or morale.
        Be specific about consequences.]

        **Recommended Actions:** [Actionable steps that could address this concern. Focus on what
        management, processes, or systems could do to help. Be concrete and practical.]

        **Example:**
        **Issue:** Deployment process delays due to slow runner spin-up times

        **Details:** The deployment infrastructure experiences significant latency during runner initialization,
        causing confusion and delays. The issue appears to be infrastructure-related rather than user-triggered,
        with runners taking an unusually long time to spin up.

        **Impact:** These delays are blocking progress, causing missed deadlines, and reducing overall team productivity.
        The uncertainty around runner spin-up times creates confusion and makes it difficult to plan deployments effectively.

        **Recommended Actions:**
        - Investigate infrastructure performance and identify bottlenecks in runner spin-up process
        - Review and optimize CI/CD pipeline configuration for runner initialization
        - Provide clearer visibility into runner status and expected spin-up times
        - Consider alternative deployment strategies or infrastructure improvements to reduce latency
        - Establish service level expectations for runner spin-up times and monitor against them
        """
    return (anonymization_sysprompt,)


@app.cell
def _(anonymization_sysprompt, lmb, os):
    anonymization_bot = lmb.SimpleBot(
        system_prompt=anonymization_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (anonymization_bot,)


@app.cell
def _(LanceDBDocStore):
    complaints_db = LanceDBDocStore(
        table_name="internal-complaints",
        enable_partitioning=True,
    )
    return (complaints_db,)


@app.cell
def _():
    from datetime import datetime
    return (datetime,)


@app.cell
def _(anonymization_bot, datetime, nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def anonymize_complaint(
        chat_history: str, _globals_dict: dict = None, globals_dict: dict = None
    ) -> str:
        """Anonymize internal company concern from chat history.

        This tool should be called AFTER you've had a conversation with the user to draw out
        sufficient actionable information (what the issue is, why it's a problem, and what impact it has).

        IMPORTANT: Pass the FULL conversation history as a formatted string, including:
        - The user's initial concern
        - Your clarifying questions (from respond_to_user calls)
        - The user's responses to your questions
        - Any additional context they've shared

        Pass the full conversation history as a natural dialogue string. The anonymization bot will
        transform it into a structured format with Issue, Details, Impact, and Recommended Actions sections.
        Example conversation to pass:
        "User: I'm worried about our deployment process.
        Agent: Can you help me understand what specifically is happening?
        User: The team lead hasn't responded to my questions about the new workflow.
        Agent: What impact is this having on your work?
        User: It's blocking my progress and causing delays."

        This ensures the anonymized version will be useful and actionable for someone reading it.

        Shows the anonymized version for review before storing.
        Use confirm_store_complaint to actually store it.

        :param chat_history: The FULL chat conversation formatted as a string, including all questions and answers
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :param globals_dict: Optional explicit globals dict (for testing/direct calls)
        :return: The anonymized complaint text for review
        """
        # Use explicit globals_dict if provided, otherwise use _globals_dict
        gdict = globals_dict if globals_dict is not None else _globals_dict

        # Anonymize and structure the concern
        anonymized = anonymization_bot(
            f"Transform this internal company concern conversation into a structured format:\n\n{chat_history}"
        )

        # Store in globals for review and potential storage
        if gdict is not None:
            gdict["anonymized_complaint"] = anonymized.content
            gdict["complaint_date"] = datetime.now().strftime("%Y-%m-%d")

        return (
            f"Here is the anonymized and structured version of your concern:\n\n{anonymized.content}\n\n"
            f"**YOU MUST NOW**: Call respond_to_user() immediately to show this structured version to the user for review. "
            f"After the user confirms, call confirm_store_complaint() to store it in the database."
        )
    return (anonymize_complaint,)


@app.cell
def _(complaints_db, datetime, nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def confirm_store_complaint(
        _globals_dict: dict = None, globals_dict: dict = None
    ) -> str:
        """Store the anonymized complaint in the database with date partition.

        This should be called after anonymize_complaint() has been executed
        and the user has reviewed the anonymized version.

        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :param globals_dict: Optional explicit globals dict (for testing/direct calls)
        :return: Confirmation message with storage details
        """
        # Use explicit globals_dict if provided, otherwise use _globals_dict
        gdict = globals_dict if globals_dict is not None else _globals_dict

        if gdict is None:
            raise ValueError(
                "No globals_dict available. "
                "When calling directly, pass globals_dict=globals() explicitly."
            )

        if "anonymized_complaint" not in gdict:
            raise ValueError(
                "No anonymized complaint found. Run anonymize_complaint() first."
            )

        anonymized = gdict["anonymized_complaint"]
        complaint_date = gdict.get(
            "complaint_date", datetime.now().strftime("%Y-%m-%d")
        )

        # Store in vector database with date as partition
        complaints_db.append(anonymized, partition=complaint_date)

        return f"Complaint stored successfully. Date partition: {complaint_date}"
    return (confirm_store_complaint,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def summarization_sysprompt():
        """You are an expert at analyzing internal company concerns.
        Summarize concerns and categorize them by common issue types:
        - Process/workflow issues
        - Team collaboration problems
        - Resource allocation concerns
        - Technical/system problems
        - Communication issues
        - Management/leadership concerns
        - Other

        Provide clear summaries and identify patterns.
        """
    return (summarization_sysprompt,)


@app.cell
def _(lmb, os, summarization_sysprompt):
    summarization_bot = lmb.SimpleBot(
        system_prompt=summarization_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return


@app.cell
def _(complaints_db, nodeify, tool):
    # Design Principle: Separation of Concerns
    # This tool only queries the database and stores results in globals.
    # It does NOT format or summarize the results - that's the agent's job via respond_to_user().
    # This separation prevents infinite loops and makes the tool's responsibility clear:
    # - Tool: Query data and store it
    # - Agent: Format and present data to the user
    @nodeify(loopback_name="decide")
    @tool
    def query_complaints(
        query: str, date_partition: str = None, _globals_dict: dict = None
    ) -> str:
        """Query anonymized internal concerns from the database.

        This tool queries the database and stores the results in globals.
        After calling this tool, use respond_to_user() to present the results to the user.

        Design Principle: Separation of Concerns
        - This tool only queries and stores data (data access layer)
        - The agent handles formatting/presentation via respond_to_user() (presentation layer)
        - This prevents infinite loops and makes responsibilities clear

        :param query: Search query for concerns (e.g., "process issues", "communication problems")
        :param date_partition: Optional date partition to search (YYYY-MM-DD format). If None, searches all partitions.
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Confirmation message indicating results are stored and ready to be returned
        """
        # Query vector database
        partitions = [date_partition] if date_partition else None
        results = complaints_db.retrieve(
            query, n_results=10, partitions=partitions
        )

        if not results:
            return (
                f"No concerns found matching query: {query}. "
                f"**YOU MUST NOW**: Call respond_to_user() immediately to inform the user."
            )

        # Store results in globals
        if _globals_dict is not None:
            _globals_dict["complaint_query_results"] = results
            _globals_dict["complaint_query"] = query

        return (
            f"Found {len(results)} concerns matching '{query}'. Results stored in 'complaint_query_results'. "
            f"**YOU MUST NOW**: Call respond_to_user() immediately to present the results to the user."
        )
    return (query_complaints,)


@app.cell
def _(mo):
    mo.md("""
    ### Test: Internal Complaints Agent

    Test anonymization, confirmation, and summarization:
    """)
    return


@app.cell
def _(anonymize_complaint, confirm_store_complaint, query_complaints):
    # Test internal complaints agent
    # mo.stop(True)
    # Uncomment to test:
    test_concern = """
    Employee: Hi, I'm Sarah from Engineering (sarah@company.com).
    I'm concerned about our deployment process. The team lead hasn't
    responded to my questions about the new workflow.
    """

    # Test anonymization (shows anonymized version)
    # Pass globals() explicitly for direct calls (not through AgentBot)
    result1 = anonymize_complaint(test_concern, _globals_dict=globals())
    print(result1)

    # After reviewing, confirm storage (pass globals() explicitly for direct calls)
    result2 = confirm_store_complaint(_globals_dict=globals())
    print(result2)

    # Test query complaints
    result3 = query_complaints("process issues", _globals_dict=globals())
    print(result3)
    if "complaint_query_results" in globals():
        print("\nQuery results:", globals()["complaint_query_results"])
    return


@app.cell
def _(
    AgentBot,
    anonymize_complaint,
    confirm_store_complaint,
    coordinator_sysprompt,
    os,
    process_receipt,
    query_complaints,
    write_invoice,
):
    # Add complaint tools to coordinator
    coordinator_with_complaints = AgentBot(
        tools=[
            process_receipt,
            write_invoice,
            anonymize_complaint,
            confirm_store_complaint,
            query_complaints,
        ],
        system_prompt=coordinator_sysprompt(),
        model_name="gpt-4.1",
        # model_name="ollama_chat/qwen3:30b",  # works!
        # model_name="ollama/deepseek-r1:14b", # failed on being unhelpful!
        # model_name="ollama_chat/deepseek-r1:14b", # failed on being stuck in a loop!
        # model_name="ollama_chat/qwen3-vl:latest", # doesn't work!
        # model_name="ollama_chat/phi4:latest", # doesn't work!
        api_base=os.getenv("TUTORIAL_API_BASE", None),
    )
    return (coordinator_with_complaints,)


@app.cell
def _(coordinator_with_complaints):
    coordinator_with_complaints.shared
    return


@app.cell
def _(coordinator_with_complaints, span):
    def chat_turn_with_complaints(messages, config):
        user_message = messages[-1].content
        with span(
            "coordinator_chat_turn",
            user_message=user_message[:100],
            category="chat_interaction",
        ):
            result = coordinator_with_complaints(user_message, globals())
        return result
    return (chat_turn_with_complaints,)


@app.cell
def _(chat_turn_with_complaints, mo):
    example_prompts_complaints = [
        "Process this receipt and extract the data",
        "Generate an invoice for Acme Corp, web development project, $5000",
        "Anonymize this internal concern: I'm worried about our deployment process",
        "Store the anonymized complaint",
        "Summarize complaints about process issues",
    ]

    chat_with_complaints = mo.ui.chat(
        chat_turn_with_complaints,
        max_height=600,
        prompts=example_prompts_complaints,
    )
    return (chat_with_complaints,)


@app.cell
def _(chat_with_complaints, mo):
    v = mo.vstack(
        [
            mo.md("# Back-Office Coordinator (with Complaints)"),
            mo.md(
                "Upload receipts, generate invoices, and handle internal concerns. "
                "The coordinator can anonymize and store complaints, and query/summarize them."
            ),
            chat_with_complaints,
        ]
    )
    v
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
    ### Best Practices

    1. **Workflow-first**: Always map workflows before coding
    2. **Schema-first**: Define data models before extraction
    3. **Template-first**: Design forms/templates before generation
    4. **Compose agents**: Build focused agents, then compose them
    5. **Test incrementally**: Test each agent before composing
    """)
    return


@app.cell
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


if __name__ == "__main__":
    app.run()
