"""Tests for the llamabot docs CLI tool.

This also serves as an evaluation suite for local LLMs that are tested.
"""

from pathlib import Path

import pytest
from llamabot import StructuredBot
from llamabot.cli.docs import (
    documentation_information,
    MarkdownSourceFile,
    DocsContainFactuallyIncorrectMaterial,
    SourceContainsContentNotCoveredInDocs,
    DocsDoNotCoverIntendedMaterial,
    DiataxisType,
    diataxis_sources,
)
from pyprojroot import here
from unittest.mock import patch
import tempfile

### Evals
system_prompt1 = """You are an expert in documentation management.
You will be provided information about a written documentation file,
what the documentation is intended to convey,
a list of source files that are related to the documentation,
and their contents.
Your goal is to determine how the document is out of sync
with the source code and intents.
There are two major ways this can happen:
- Either the docs are out of sync with the source code, or
- The intents have changed and the docs do not satisfy the intents.
"""

system_prompt2 = """You are an expert in documentation management.
You will be provided information about a written documentation file,
what the documentation is intended to convey,
a list of source files that are related to the documentation,
and their contents.
You will also be provided a JSON model representing an error mode with the documentation.
Your goal is to identify whether the documentation has the issue described in the error mode.
If yes, return True, and a list of reasons for the issue,
otherwise return False with no reasons.
"""

original_docs = """
## Tutorial: Using the prime number function

This tutorial will guide you through the usage of the `is_prime` and `next_prime` functions provided in the `source.py` file. These functions are designed to work together, allowing you to check if a number is prime and to find the next prime number after a given number.

### Function: `is_prime(n)`

#### Purpose:
The `is_prime` function checks whether a given number `n` is prime. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

#### Usage:
```python
result = is_prime(11)
print(result)  # Output: True
```

#### How it works:
1. The function first handles edge cases:
   - Numbers less than or equal to 1 are not prime.
   - The number 2 is the smallest and only even prime number, so it returns `True`.
   - Any other even number is not prime and returns `False`.

2. For odd numbers greater than 2, the function checks divisibility by all odd numbers up to the square root of `n` and by skipping even numbers (incrementing by 2). This is an optimization, as a larger factor of `n` would necessarily be a multiple of a smaller factor that has already been checked.

### Function: `next_prime(current_number)`

#### Purpose:
The `next_prime` function finds the smallest prime number greater than the given `current_number`.

#### Usage:
```python
next_prime_number = next_prime(11)
print(next_prime_number)  # Output: 13
```

#### How it works:
1. The function starts by incrementing the `current_number` by 1.
2. It then checks each successive number using the `is_prime` function until it finds a prime number.
3. Once a prime number is found, it is returned as the result.

### Optimizations:
- **Square root check**: In the `is_prime` function, checking divisibility only up to the square root of `n` significantly reduces the number of operations, especially for large numbers.
- **Skip even numbers**: After checking divisibility by 2, the `is_prime` function only checks odd numbers. This further reduces unnecessary checks, improving performance.

By combining these two functions, you can efficiently determine the primality of a number and find the next prime number with minimal computation.
"""

with open(Path(__file__).parent / "assets" / "next_prime" / "source.py", "r+") as f:
    original_source_code = f.read()

new_source_code = '''
def is_prime(n):
    """Check if a number is prime.

    :param n: The number to check.
    :return: True if the number is prime, False otherwise.
    """
    if n < 2:  # Changed from n <= 1
        return False
    if n == 2 or n == 3:  # Added an additional base case for 3
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:  # Added check for divisibility by 3
        return False
    for i in range(5, int(n**0.5) + 1, 6):  # Modified to step by 6
        if n % i == 0 or n % (i + 2) == 0:  # Check i and i+2
            return False
    return True

def next_prime(current_number):
    """Find the next prime number after the current number.

    :param current_number: The current number.
    :return: The next prime number.
    """
    next_number = current_number + 1
    while not is_prime(next_number):
        next_number += 1
    return next_number
'''

test_cases = (
    {
        "original_docs": original_docs,
        "new_source_code": new_source_code,
        "system_prompt": system_prompt2,
        "pydantic_model": DocsContainFactuallyIncorrectMaterial,
        "expected_status": True,
    },
    {
        "original_docs": original_docs,
        "new_source_code": new_source_code,
        "system_prompt": system_prompt1,
        "pydantic_model": DocsContainFactuallyIncorrectMaterial,
        "expected_status": True,
    },
    {
        "original_docs": original_docs,
        "new_source_code": new_source_code,
        "system_prompt": system_prompt1,
        "pydantic_model": SourceContainsContentNotCoveredInDocs,
        "expected_status": True,
    },
    {
        "original_docs": original_docs,
        "new_source_code": new_source_code,
        "system_prompt": system_prompt1,
        "pydantic_model": DocsDoNotCoverIntendedMaterial,
        "expected_status": True,
    },
    {
        "original_docs": original_docs,
        "new_source_code": original_source_code,
        "system_prompt": system_prompt2,
        "pydantic_model": DocsContainFactuallyIncorrectMaterial,
        "expected_status": False,
    },
    {
        "original_docs": original_docs,
        "new_source_code": original_source_code,
        "system_prompt": system_prompt1,
        "pydantic_model": DocsContainFactuallyIncorrectMaterial,
        "expected_status": False,
    },
    {
        "original_docs": original_docs,
        "new_source_code": original_source_code,
        "system_prompt": system_prompt1,
        "pydantic_model": SourceContainsContentNotCoveredInDocs,
        "expected_status": False,
    },
    {
        "original_docs": original_docs,
        "new_source_code": original_source_code,
        "system_prompt": system_prompt1,
        "pydantic_model": DocsDoNotCoverIntendedMaterial,
        "expected_status": False,
    },
)


@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "original_docs,new_source_code,system_prompt,pydantic_model,expected_status",
    [(tuple(case.values())) for case in test_cases],
)
@pytest.mark.parametrize(
    "model_name",
    [
        # "ollama_chat/gemma2:2b",  # does not pass all tests.
        "gpt-4-turbo",  # passes all tests, but costs $$ to run.
        # ollama_chat/phi3", # garbage model, doesn't pass any tests.
        "gpt-4o-mini",  # passes all tests, but costs $$ to run.
    ],
)
def test_out_of_date_when_source_changes(
    original_docs: str,
    new_source_code: str,
    system_prompt: str,
    pydantic_model,
    model_name: str,
    expected_status: bool,
):
    """Test out-of-date checker when source changes.

    This test assumes that any model specified by `model_name`
    will correctly identify the out-of-date status of documentation.
    """
    source_file = MarkdownSourceFile(
        here() / "tests" / "cli" / "assets" / "next_prime" / "docs.md"
    )
    source_file.post.content = original_docs
    source_file.linked_files["tests/cli/assets/next_prime/source.py"] = new_source_code

    doc_issue_checker = StructuredBot(
        system_prompt=system_prompt,
        pydantic_model=pydantic_model,
        model_name=model_name,
        stream_target="none",
    )
    doc_issue = doc_issue_checker(documentation_information(source_file))
    assert doc_issue.status == expected_status
    if doc_issue.status:
        assert doc_issue.reasons


def test_markdown_source_file_diataxis():
    """Test that the MarkdownSourceFile can correctly identify the diataxis type and source."""
    # Create a temporary markdown file with diataxis type
    test_content = """---
diataxis_type: howto
---
# Test Content
"""
    with patch("requests.get") as mock_get:
        # Mock the response from the diataxis source
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "Mock diataxis guide content"

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.md"
            test_file.write_text(test_content)

            # Create MarkdownSourceFile instance
            md_file = MarkdownSourceFile(test_file)

            # Check if diataxis type and source are properly set
            assert md_file.diataxis_type == DiataxisType.HOWTO
            assert md_file.diataxis_source == "Mock diataxis guide content"

            # Check if the request was made with the correct URL
            mock_get.assert_called_once_with(diataxis_sources[DiataxisType.HOWTO])
