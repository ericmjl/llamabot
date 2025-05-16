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


def test_split_document_empty_text():
    """Test split_document with empty text.

    This test tests that split_document returns an empty list
    when the text of the Document object is empty.
    """
    doc = ""

    result = split_document(doc)

    assert len(result) == 0
