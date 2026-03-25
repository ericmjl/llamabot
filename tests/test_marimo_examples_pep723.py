"""Ensure converted Marimo example notebooks declare PEP 723 script metadata."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Notebooks converted from Jupyter; must stay runnable with `uvx marimo run --sandbox`.
MARIMO_EXAMPLES = [
    REPO_ROOT / "docs/examples/chatbot_nb.py",
    REPO_ROOT / "docs/examples/docstring_checker.py",
    REPO_ROOT / "docs/examples/groq.py",
    REPO_ROOT / "docs/examples/imagebot.py",
    REPO_ROOT / "docs/examples/json_mode.py",
    REPO_ROOT / "docs/examples/knowledge-graph-bot.py",
    REPO_ROOT / "docs/examples/pdf.py",
    REPO_ROOT / "docs/examples/querybot.py",
    REPO_ROOT / "docs/examples/recorder.py",
    REPO_ROOT / "docs/examples/simple_panel.py",
    REPO_ROOT / "docs/examples/simplebot.py",
    REPO_ROOT / "docs/examples/sse_streaming.py",
    REPO_ROOT / "docs/examples/structuredbot.py",
    REPO_ROOT / "talks/scipy2024/demo.py",
    REPO_ROOT / "talks/pydata-boston-2025/what-makes-an-agent.py",
]


@pytest.mark.parametrize("path", MARIMO_EXAMPLES)
def test_marimo_example_has_pep723_block(path: Path) -> None:
    """Each listed example must start with a PEP 723 script metadata block."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# /// script\n"), f"{path} missing PEP 723 opening"
    assert (
        "requires-python" in text[:800]
    ), f"{path} missing requires-python in script block"
    assert "dependencies" in text[:800], f"{path} missing dependencies in script block"
    assert "# ///\n" in text[:800], f"{path} missing PEP 723 closing"
