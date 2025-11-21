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

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    # Ensure we're using the local editable llamabot installation
    # Add the repository root to sys.path so local changes are picked up
    repo_root = Path(__file__).parent.parent.parent
    llamabot_path = repo_root / "llamabot"
    if llamabot_path.exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return Path, mo


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
def _(Path, convert_from_path, tempfile):
    def convert_pdf_to_images(file_path: str):
        """Convert PDF to list of image paths."""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            images = convert_from_path(file_path, dpi=200)
            image_paths = []
            for i, image in enumerate(images):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_page_{i + 1}.png"
                ) as temp_img:
                    image.save(temp_img.name, "PNG")
                    image_paths.append(temp_img.name)
            return image_paths
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            return [file_path]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    return (convert_pdf_to_images,)


@app.cell
def _():
    from llamabot.components.messages import ImageMessage, user
    return (user,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Implementation: Two Bots

    We create two separate bots for the two-step process:
    """)
    return


@app.cell
def _(lmb):
    # Step 1: OCR extraction with DeepSeek-OCR (SimpleBot)
    # DeepSeek-OCR doesn't support structured outputs, so we use SimpleBot
    ocr_bot = lmb.SimpleBot(
        system_prompt="Extract all text from receipts accurately. "
        "Preserve the structure and include all numbers, dates, and vendor names.",
        model_name="ollama/deepseek-ocr",  # Fixed: ollama/deepseek-ocr
        # api_base="https://<your-modal-endpoint>.modal.run",  # Uncomment and add your endpoint
    )
    return (ocr_bot,)


@app.cell
def _(ReceiptData, lmb, receipt_extraction_sysprompt):
    # Step 2: Structure the data (using a model that supports structured outputs)
    # This bot takes the OCR text and structures it according to ReceiptData schema
    receipt_structuring_bot = lmb.StructuredBot(
        system_prompt=receipt_extraction_sysprompt(),
        pydantic_model=ReceiptData,
        model_name="ollama_chat/gemma3n:latest",  # Or use "gpt-4.1" if available
        # api_base="https://<your-modal-endpoint>.modal.run",  # Uncomment and add your endpoint
    )
    return (receipt_structuring_bot,)


@app.cell
def _(mo):
    mo.md("""
    ### Putting It Together: The Extraction Function

    The `extract_receipt_data` function orchestrates the two-step process:
    - Converts PDFs to images (if needed)
    - Runs OCR on each image using `ocr_bot`
    - Combines OCR results
    - Structures the combined text using `receipt_structuring_bot`
    """)
    return


@app.cell
def _(convert_pdf_to_images, ocr_bot, receipt_structuring_bot, user):
    def extract_receipt_data(file_path: str):
        """Extract receipt data from PDF or image file using two-step process:
        1. OCR extraction with DeepSeek-OCR (SimpleBot)
        2. Structure the data with StructuredBot
        """
        image_paths = convert_pdf_to_images(file_path)

        if len(image_paths) == 1:
            prompt_text = "Extract all text from this receipt image."
        else:
            prompt_text = f"Extract all text from this {len(image_paths)}-page receipt document."

        # Step 1: OCR extraction - extract text from images
        # Process each image and combine the results
        ocr_texts = []
        for image_path in image_paths:
            ocr_response = ocr_bot(user(prompt_text, image_path))
            ocr_texts.append(ocr_response.content)

        # Combine OCR results from all pages
        combined_ocr_text = "\n\n--- Page Break ---\n\n".join(ocr_texts)

        # Step 2: Structure the extracted text according to ReceiptData schema
        result = receipt_structuring_bot(combined_ocr_text)
        return result
    return (extract_receipt_data,)


@app.cell
def _():
    from llamabot.components.tools import tool
    from llamabot.components.pocketflow import nodeify
    return nodeify, tool


@app.cell
def _(Path, extract_receipt_data, nodeify, tool):
    @nodeify(loopback_name="decide")
    @tool
    def process_receipt(file_path: str, _globals_dict: dict = None) -> str:
        """Process a receipt PDF or image and extract structured data.

        :param file_path: Path to the receipt file (PDF, PNG, JPG, or JPEG) or variable name in globals
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: JSON string of extracted receipt data
        """
        # If file_path is a variable name in globals, get its value
        if _globals_dict is not None and file_path in _globals_dict:
            actual_path = _globals_dict[file_path]
            # If it's a string that looks like a path, use it
            if isinstance(actual_path, str):
                file_path = actual_path

        # Verify the file exists
        if not Path(file_path).exists():
            available_files = [
                k
                for k, v in (_globals_dict or {}).items()
                if isinstance(v, str) and Path(v).exists()
            ]
            raise FileNotFoundError(
                f"Receipt file not found: {file_path}. "
                f"Available file variables in globals: {available_files}"
            )

        receipt_data = extract_receipt_data(file_path)
        return receipt_data.model_dump_json()
    return (process_receipt,)


@app.cell
def _(mo):
    mo.md("""
    ### Test: Receipt Processor

    Upload a receipt PDF or image above, then test the receipt processor:
    """)
    return


@app.cell
def _(extract_receipt_data):
    # Test receipt processor with uploaded file
    # Replace 'your_receipt_file' with the variable name from uploaded files
    # mo.stop(True)
    # Uncomment and modify to test:
    test_receipt_data = extract_receipt_data(
        "/Users/ericmjl/github/llamabot/tutorials/pydata-boston-2025/receipt_coffee_1.pdf"
    )
    test_receipt_data
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
def _(InvoiceData, invoice_generation_sysprompt, lmb):
    invoice_writer_bot = lmb.StructuredBot(
        system_prompt=invoice_generation_sysprompt(),
        pydantic_model=InvoiceData,
        model_name="ollama_chat/gemma3n:latest",
        # api_base="https://<your-modal-endpoint>.modal.run",  # Uncomment and add your endpoint
    )
    return (invoice_writer_bot,)


@app.cell
def _(InvoiceData, invoice_writer_bot):
    def generate_invoice(invoice_description: str) -> InvoiceData:
        """Generate invoice from natural language description.

        :param invoice_description: Natural language description of the invoice to generate.
            Should include client name, project description, amount, and any other relevant details.
        """
        prompt = f"""Generate an invoice based on this description:
        {invoice_description}

        Use today's date as issue_date and 30 days from today as due_date.
        Generate a professional invoice number in format INV-YYYY-XXX.
        Extract client information, project details, and amount from the description.
        """

        invoice = invoice_writer_bot(prompt)
        return invoice
    return (generate_invoice,)


@app.cell
def _(InvoiceData):
    def render_invoice_html(invoice: InvoiceData) -> str:
        """Render invoice as beautiful HTML."""
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
        return html
    return (render_invoice_html,)


@app.cell
def _(generate_invoice, nodeify, render_invoice_html, tool):
    @nodeify(loopback_name="decide")
    @tool
    def write_invoice(invoice_description: str, _globals_dict: dict = None) -> str:
        """Generate and render an invoice from natural language description.

        :param invoice_description: Natural language description of the invoice to generate.
            Example: "Invoice for Acme Corp, web development project, $5000, due in 30 days"
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :return: Confirmation message indicating invoice was generated
        """
        invoice = generate_invoice(invoice_description)
        html = render_invoice_html(invoice)

        # Store invoice HTML in globals so it can be returned to user
        if _globals_dict is not None:
            _globals_dict["invoice_html"] = html

        return "Invoice generated successfully. Use return_object_to_user('invoice_html') to return it to the user."
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
        Use the available tools to:
        1. Process receipts to extract structured data
        2. Generate invoices to clients from natural language descriptions
        3. After generating an invoice, use return_object_to_user('invoice_html') to return it to the user
        4. For internal concerns: anonymize first, show the anonymized version, then confirm storage
        5. Handle user requests efficiently

        Always explain what you're doing and why.
        """
    return


@app.cell
def _():
    from llamabot import AgentBot
    return (AgentBot,)


@app.cell
def _(AgentBot, process_receipt, write_invoice):
    coordinator_bot = AgentBot(
        tools=[process_receipt, write_invoice],
        model_name="gpt-4.1",
        # api_base="https://<your-modal-endpoint>.modal.run",  # Uncomment and add your endpoint
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
def _(coordinator_bot):
    def chat_turn(messages, config):
        user_message = messages[-1].content
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
        """You are an expert at anonymizing internal company concerns.
        Remove all personally identifiable information (names, emails, phone numbers, specific departments).
        Replace with generic placeholders like [EMPLOYEE], [DEPARTMENT], etc.
        Preserve the core concern content and sentiment.
        """
    return (anonymization_sysprompt,)


@app.cell
def _(anonymization_sysprompt, lmb):
    anonymization_bot = lmb.SimpleBot(
        system_prompt=anonymization_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
        # api_base="https://<your-modal-endpoint>.modal.run",
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

        Shows the anonymized version for review before storing.
        Use confirm_store_complaint to actually store it.

        :param chat_history: The chat conversation to anonymize
        :param _globals_dict: Internal parameter - automatically injected by AgentBot
        :param globals_dict: Optional explicit globals dict (for testing/direct calls)
        :return: The anonymized complaint text for review
        """
        # Use explicit globals_dict if provided, otherwise use _globals_dict
        gdict = globals_dict if globals_dict is not None else _globals_dict

        # Anonymize
        anonymized = anonymization_bot(
            f"Anonymize this internal company concern conversation:\n\n{chat_history}"
        )

        # Store in globals for review and potential storage
        if gdict is not None:
            gdict["anonymized_complaint"] = anonymized.content
            gdict["complaint_date"] = datetime.now().strftime("%Y-%m-%d")

        return f"Anonymized complaint:\n\n{anonymized}\n\nUse confirm_store_complaint() to store this in the database."
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
def _(lmb, summarization_sysprompt):
    summarization_bot = lmb.SimpleBot(
        system_prompt=summarization_sysprompt(),
        model_name="ollama_chat/gemma3n:latest",
        # api_base="https://<your-modal-endpoint>.modal.run",
    )
    return (summarization_bot,)


@app.cell
def _(complaints_db, nodeify, summarization_bot, tool):
    @nodeify(loopback_name="decide")
    @tool
    def query_and_summarize_complaints(
        query: str, date_partition: str = None
    ) -> str:
        """Query anonymized internal concerns and summarize by category.

        :param query: Search query for concerns (e.g., "process issues", "communication problems")
        :param date_partition: Optional date partition to search (YYYY-MM-DD format). If None, searches all partitions.
        :return: Summary of concerns categorized by issue type
        """
        # Query vector database
        partitions = [date_partition] if date_partition else None
        results = complaints_db.retrieve(
            query, n_results=10, partitions=partitions
        )

        if not results:
            return f"No concerns found matching query: {query}"

        # Summarize and categorize
        summary = summarization_bot(
            f"Analyze these internal concerns and provide a summary categorized by issue type:\n\n{chr(10).join(results)}"
        )

        return summary
    return (query_and_summarize_complaints,)


@app.cell
def _(mo):
    mo.md("""
    ### Test: Internal Complaints Agent

    Test anonymization, confirmation, and summarization:
    """)
    return


@app.cell
def _(
    anonymize_complaint,
    confirm_store_complaint,
    query_and_summarize_complaints,
):
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

    # Test query and summarize
    result3 = query_and_summarize_complaints("process issues")
    print(result3)
    return


@app.cell
def _(
    AgentBot,
    anonymize_complaint,
    confirm_store_complaint,
    process_receipt,
    query_and_summarize_complaints,
    write_invoice,
):
    # Add complaint tools to coordinator
    coordinator_with_complaints = AgentBot(
        tools=[
            process_receipt,
            write_invoice,
            anonymize_complaint,
            confirm_store_complaint,
            query_and_summarize_complaints,
        ],
        model_name="gpt-4.1",
        # api_base="https://<your-modal-endpoint>.modal.run",
    )
    return (coordinator_with_complaints,)


@app.cell
def _(coordinator_with_complaints):
    def chat_turn_with_complaints(messages, config):
        user_message = messages[-1].content
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
    mo.vstack(
        [
            mo.md("# Back-Office Coordinator (with Complaints)"),
            mo.md(
                "Upload receipts, generate invoices, and handle internal concerns. "
                "The coordinator can anonymize and store complaints, and query/summarize them."
            ),
            chat_with_complaints,
        ]
    )
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
