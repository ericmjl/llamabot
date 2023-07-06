"""
Code for document preprocessing.

This module provides functions for loading and splitting documents.
It supports various file formats, including PDF, DOCX, PPTX, XLSX, Markdown, and IPython Notebook.

Functions:
    - magic_load_doc(file_path: str) -> List[Document]:
        Load a document from a file, inferring the file format from the file extension.
    - split_document(doc: Document, chunk_size: int = 2000, chunk_overlap: int = 0) -> List[Document]:
        Split a document into sub-documents using token text splitter, with a maximum length of `chunk_size` tokens
        and `chunk_overlap` tokens overlap.

Classes:
    - Document: Represents a document with text content.

Exceptions:
    - None

Dependencies:
    - langchain.text_splitter: Provides the TokenTextSplitter class for splitting text.
    - llama_index: Provides the Document class and download_loader function.
"""


from pathlib import Path
from typing import List

from langchain.text_splitter import TokenTextSplitter
from llama_index import Document, download_loader

EXTENSION_LOADER_MAPPING = {
    ".pdf": "PDFReader",
    ".docx": "DocxReader",
    ".pptx": "PptxReader",
    ".xlsx": "PandasExcelReader",
    ".md": "MarkdownReader",
    ".ipynb": "IPYNBReader",
}


def magic_load_doc(file_path: Path) -> List[Document]:
    """Load a document from a file.

    This function is used to magically load a document from a file.
    The file extension is inferred from the file path.
    If the file extension is not recognized, it is assumed to be a plain text file.

    :param file_path: Path to the file to be loaded.
    :return: A list of documents.
    """
    loader_string: str = EXTENSION_LOADER_MAPPING.get(Path(file_path).suffix, None)
    if loader_string is not None:
        # Treat this as a document that needs special processing.
        Loader = download_loader(loader_string)
        loader = Loader()
        documents = loader.load_data(file_path)

        # Concatenate the documents if there are more than 1 document in documents:
        if len(documents) > 1:
            documents = [Document(text=" ".join([doc.text for doc in documents]))]

    else:
        # Treat this as a plain text file.
        with open(file_path, "r+") as f:
            documents = [Document(text=str(file_path) + f.read())]
    return documents


# Split a document into sub-documents using token text splitter, with maximum length of 2000 tokens.


def split_document(
    doc: Document,
    chunk_size: int = 2000,
    chunk_overlap: int = 0,
) -> List[Document]:
    """Split a document into sub-documents using token text splitter.

    This function is used to split a document into sub-documents using token text splitter,
    such that each sub-document has a maximum length of `chunk_size` tokens
    with `chunk_overlap` tokens overlap.

    :param doc: Document to be split.
    :param chunk_size: Maximum length of each sub-document.
    :param chunk_overlap: Number of tokens to overlap between each sub-document.
    :raises ValueError: If `chunk_overlap` is negative.
    :return: A list of sub-documents.
    """
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    sub_texts = splitter.split_text(doc.text)
    sub_texts

    sub_docs = [Document(text=t) for t in sub_texts]
    return sub_docs
