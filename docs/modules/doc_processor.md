# Doc Processor

!!! note
    This tutorial was written by GPT4 and edited by a human.

The doc processor is a Python script designed to preprocess documents
by loading them from various file formats
and splitting them into smaller sub-documents.
It works in two main steps:

**(1) Loading documents**:
The `magic_load_doc` function is used to load a document from a file.
It automatically detects the file format based on the file extension
and uses the appropriate loader to read the content.
Supported file formats include PDF, DOCX, PPTX, XLSX, Markdown, and IPYNB.
If the file format is not recognized, it is treated as a plain text file.

**(2) Splitting documents**:
The `split_document` function is used to split a document
into smaller sub-documents using a token text splitter.
You can specify the maximum length of each sub-document (`chunk_size`)
and the number of tokens to overlap between each sub-document (`chunk_overlap`).
The function returns a list of sub-documents.

To use the doc processor,
simply import the required functions and call them with the appropriate parameters.
For example:

```python
from llamabot.doc_processor import magic_load_doc, split_document

# Load a document from a file
file_path = "path/to/your/document.pdf"
documents = magic_load_doc(file_path)

# Split the document into sub-documents
chunk_size = 2000
chunk_overlap = 0
sub_documents = [split_document(doc, chunk_size, chunk_overlap) for doc in documents]
```

This will give you a list of sub-documents that can be further processed inside QueryBot.
