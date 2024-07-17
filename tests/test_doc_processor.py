"""Tests for doc_processor.py."""

from pathlib import Path

import pytest
from pytest_mock import mocker  # noqa: F401

from llamabot.doc_processor import (
    magic_load_doc,
    split_document,
)


@pytest.fixture
def txt_file(tmp_path: Path) -> Path:
    """Create a test text file.

    :param tmp_path: Pytest fixture, the Path to the temporary directory.
    :return: Path to the test text file.
    """
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test text file.")
    return file_path


@pytest.fixture
def unknown_file(tmp_path: Path) -> Path:
    """Create a test file with an unknown extension.

    :param tmp_path: Pytest fixture, the Path to the temporary directory.
    :return: Path to the test unknown file.
    """
    file_path = tmp_path / "test.unknown"
    file_path.write_text("This is a test file with an unknown extension.")
    return file_path


def test_magic_load_doc_txt(txt_file: Path):
    """Test magic_load_doc for a text file.

    This test tests that magic_load_doc returns a list of one Document object,
    and the text of the Document object is the content of the test text file.

    :param txt_file: Pytest fixture, the Path to the test text file.
    """
    result = magic_load_doc(txt_file)
    # assert len(result) == 1
    assert isinstance(result, str)
    assert result == "This is a test text file."


def test_magic_load_doc_unknown(unknown_file: Path):
    """Test magic_load_doc for a file with an unknown extension.

    This test tests that magic_load_doc returns a list of one Document object,
    and the text of the Document object is the content of the test unknown file.

    :param unknown_file: Pytest fixture, the Path to the test unknown file.
    """
    result = magic_load_doc(unknown_file)
    assert isinstance(result, str)
    assert result == "This is a test file with an unknown extension."


def test_split_document_no_overlap():
    """Test split_document with no overlap.

    This test tests that split_document returns a list of three Document objects,
    and the text of the Document objects are the correct chunks of the test document.
    """
    doc = "This is a test document. It has multiple sentences."
    chunk_size = 5
    chunk_overlap = 0

    result = split_document(doc, chunk_size, chunk_overlap)

    assert len(result) == 11
    assert result[0] == "This "
    assert result[1] == "is a "
    assert result[2] == "test "


def test_split_document_with_overlap():
    """Test split_document with overlap.

    This test tests that split_document returns a list of four Document objects,
    and the text of the Document objects are the correct chunks of the test document.
    """
    doc = "This is a test document. It has multiple sentences."
    chunk_size = 5
    chunk_overlap = 2

    result = split_document(doc, chunk_size, chunk_overlap)

    assert len(result) == 17
    assert result[0] == "This "
    assert result[1] == "s is "
    assert result[2] == "s a t"


def test_split_document_empty_text():
    """Test split_document with empty text.

    This test tests that split_document returns an empty list
    when the text of the Document object is empty.
    """
    doc = ""
    chunk_size = 5
    chunk_overlap = 0

    result = split_document(doc, chunk_size, chunk_overlap)

    assert len(result) == 0


def test_split_document_chunk_size_larger_than_text():
    """Test split_document with chunk_size larger than the text.

    This test tests that split_document returns a list of one Document object,
    and the text of the Document object is the text of the Document object.
    """
    doc = "This is a test document."
    chunk_size = 10
    chunk_overlap = 0

    result = split_document(doc, chunk_size, chunk_overlap)

    assert len(result) == 3
    assert result[0] == "This is a "


def test_split_document_invalid_chunk_size():
    """Test split_document with invalid chunk_size.

    This test tests that split_document raises a ValueError
    when the chunk_size is negative.
    """
    doc = "This is a test document."
    chunk_size = -1
    chunk_overlap = 0

    with pytest.raises(ValueError):
        split_document(doc, chunk_size, chunk_overlap)


def test_split_document_invalid_chunk_overlap():
    """Test split_document with invalid chunk_overlap.

    This test tests that split_document raises a ValueError
    when the chunk_overlap is negative.
    """
    doc = "This is a test document."
    chunk_size = 5
    chunk_overlap = -1

    with pytest.raises(ValueError):
        split_document(doc, chunk_size, chunk_overlap)
