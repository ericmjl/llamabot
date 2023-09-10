"""Tests for the PaperTitleCompleter class."""

from llamabot.zotero.completer import PaperTitleCompleter
from prompt_toolkit.document import Document
import pytest

paper_titles = [
    "Introduction to Machine Learning",
    "Deep Learning for Beginners",
    "Data Science Foundations",
    "Python Programming for Data Analysis",
    "Natural Language Processing with Python",
]


@pytest.mark.parametrize(
    "input_text, expected_count, expected_completion",
    [
        ("", 0, None),
        ("Deep", 1, "Deep Learning for Beginners"),
        ("deep learning", 1, "Deep Learning for Beginners"),
        ("Non-Existent Title", 0, None),
        ("Introduction to Machine Learning", 1, "Introduction to Machine Learning"),
    ],
)
def test_paper_title_completer(input_text, expected_count, expected_completion):
    """Test PaperTitleCompleter.get_completions().

    :param input_text: The input text to complete.
    :param expected_count: The expected number of completions.
    :param expected_completion: The expected completion.
    """
    completer = PaperTitleCompleter(paper_titles)
    document = Document(input_text, len(input_text))
    completions = list(completer.get_completions(document, None))

    assert len(completions) == expected_count

    if expected_completion is not None:
        assert completions[0].text == expected_completion
