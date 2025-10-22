"""Code for document preprocessing."""

from pathlib import Path
from typing import List

from pdfminer.high_level import extract_text
from chonkie import SemanticChunker


def pdf_loader(path: Path) -> str:
    """Load a PDF file from a file path.

    :param path: Path to the PDF file to be loaded.
    :return: Text extracted from the PDF file.
    """
    return extract_text(str(path))


def plaintext_loader(path: Path) -> str:
    """Load a plain text file from a file path.

    :param path: Path to the plain text file to be loaded.
    :return: Text from the plain text file.
    """
    with open(path, "r+") as f:
        return f.read()


EXTENSION_LOADER_MAPPING = {
    ".pdf": pdf_loader,
    # ".docx": "DocxReader",
    # ".pptx": "PptxReader",
    # ".xlsx": "PandasExcelReader",
}


def magic_load_doc(file_path: Path) -> str:
    """Load a document from a file.

    This function is used to magically load a document from a file.
    The file extension is inferred from the file path.
    If the file extension is not recognized, it is assumed to be a plain text file.

    :param file_path: Path to the file to be loaded.
    :return: A list of documents.
    """
    loader = EXTENSION_LOADER_MAPPING.get(Path(file_path).suffix, plaintext_loader)
    return loader(file_path)


def split_document(
    doc: str,
) -> List[str]:
    """Split a document into sub-documents using token text splitter.

    This function is used to split a document
    into sub-documents using token text splitter,
    such that each sub-document has a maximum length of `chunk_size` tokens
    with `chunk_overlap` tokens overlap.

    :param doc: Document to be split.
    :param chunk_size: Maximum length of each sub-document.
    :param chunk_overlap: Number of tokens to overlap between each sub-document.
    :raises ValueError: If `chunk_overlap` is negative.
    :return: A list of sub-documents.
    """
    chunker = SemanticChunker(
        threshold=0.5,  # Similarity threshold (0-1)
        chunk_size=2048,  # Maximum tokens per chunk
        min_sentences_per_chunk=1,  # Minimum sentences per chunk (renamed parameter)
        skip_window=1,  # Number of chunks to skip when looking for similarities
    )
    chunks = chunker.chunk(doc)
    return [chunk.text for chunk in chunks]
