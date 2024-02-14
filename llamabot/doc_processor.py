"""Code for document preprocessing."""


from functools import lru_cache
from pathlib import Path
from typing import List

from llama_index.core.schema import Document
from llama_index.core.readers import download_loader

EXTENSION_LOADER_MAPPING = {
    ".pdf": "PDFReader",
    ".docx": "DocxReader",
    ".pptx": "PptxReader",
    ".xlsx": "PandasExcelReader",
    ".md": "MarkdownReader",
    ".ipynb": "IPYNBReader",
    ".html": "UnstructuredReader",
}

# We lru_cache this in order to make it run faster.
# Eric Ma profiled this code on 20 Jan 2024; it takes about 0.5 seconds per call.
download_loader = lru_cache(download_loader)


def magic_load_doc(file_path: Path) -> Document:
    """Load a document from a file.

    This function is used to magically load a document from a file.
    The file extension is inferred from the file path.
    If the file extension is not recognized, it is assumed to be a plain text file.

    :param file_path: Path to the file to be loaded.
    :return: A list of documents.
    """
    doc = Document(text="")
    loader_string: str = EXTENSION_LOADER_MAPPING.get(Path(file_path).suffix, None)
    if loader_string is not None:
        # Treat this as a document that needs special processing.
        Loader = download_loader(loader_string)
        loader = Loader()
        documents = loader.load_data(file_path)

        # Concatenate the documents if there are more than 1 document in documents:
        if len(documents) > 1:
            doc = Document(text=" ".join([doc.text for doc in documents]))

    else:
        # Treat this as a plain text file.
        with open(file_path, "r+") as f:
            doc = Document(text=f.read())
    return doc


def split_document(
    doc: Document,
    chunk_size: int = 2000,
    chunk_overlap: int = 0,
) -> List[Document]:
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
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_size < 0:
        raise ValueError("chunk_size must be non-negative.")

    # Split the document into sub-documents with chunk size of `chunk_size` characters
    # and overlap of `chunk_overlap` characters.
    sub_texts = []
    for i in range(0, len(doc.text), chunk_size - chunk_overlap):
        sub_texts.append(doc.text[i : i + chunk_size])

    sub_docs = [Document(text=t) for t in sub_texts]
    return sub_docs
