"""Tests for the MCP server CLI command."""

from pathlib import Path
from llamabot.mcp_server import (
    collect_markdown_files,
    extract_module_docstring,
    collect_python_modules,
    create_mcp_server,
    build_docstore,
)
from pyprojroot import here
import tempfile


def test_collect_markdown_files():
    """Test that markdown files are discovered from docs directory."""
    project_root = here()
    docs_dir = project_root / "docs"

    markdown_files = collect_markdown_files(docs_dir)

    # Should find at least some markdown files
    assert len(markdown_files) > 0

    # All files should be markdown
    for file_path in markdown_files:
        assert file_path.suffix == ".md"

    # Should include index.md
    index_files = [f for f in markdown_files if f.name == "index.md"]
    assert len(index_files) > 0


def test_extract_module_docstring():
    """Test extraction of module docstrings from Python files."""
    # Create a temporary Python file with a docstring
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write('"""This is a test module docstring."""\n\n')
        tmp_file.write("def test_function():\n")
        tmp_file.write('    """Test function."""\n')
        tmp_file.write("    pass\n")
        tmp_path = Path(tmp_file.name)

    try:
        docstring = extract_module_docstring(tmp_path)
        assert docstring == "This is a test module docstring."
    finally:
        tmp_path.unlink()


def test_extract_module_docstring_no_docstring():
    """Test extraction when there's no docstring."""
    # Create a temporary Python file without a docstring
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write("def test_function():\n")
        tmp_file.write("    pass\n")
        tmp_path = Path(tmp_file.name)

    try:
        docstring = extract_module_docstring(tmp_path)
        assert docstring == ""
    finally:
        tmp_path.unlink()


def test_extract_module_docstring_invalid_syntax():
    """Test extraction with invalid Python syntax."""
    # Create a temporary file with invalid Python
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write("This is not valid Python code!!!\n")
        tmp_path = Path(tmp_file.name)

    try:
        docstring = extract_module_docstring(tmp_path)
        assert docstring == ""
    finally:
        tmp_path.unlink()


def test_collect_python_modules():
    """Test collection of Python modules and their docstrings."""
    project_root = here()
    source_dir = project_root / "llamabot"

    modules = collect_python_modules(source_dir)

    # Should find some modules with docstrings
    assert len(modules) > 0

    # All keys should be relative paths to .py files
    for module_path in modules.keys():
        assert module_path.endswith(".py")

    # All values should be non-empty strings
    for docstring in modules.values():
        assert isinstance(docstring, str)
        assert len(docstring) > 0

    # Should not include __pycache__ directories
    for module_path in modules.keys():
        assert "__pycache__" not in module_path


def test_build_docstore():
    """Test that the docstore can be built."""
    import tempfile
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the docstore path to use a temporary directory
        with (
            patch("llamabot.mcp_server.get_docstore_path") as mock_path,
            patch("llamabot.mcp_server.fetch_docs_from_github") as mock_fetch,
            patch("llamabot.mcp_server.collect_markdown_files") as mock_markdown,
            patch("llamabot.mcp_server.collect_python_modules") as mock_python,
            patch("llamabot.mcp_server.LanceDBDocStore") as mock_docstore_class,
        ):
            mock_path.return_value = Path(temp_dir)

            # Mock GitHub fetch to return None (use local docs)
            mock_fetch.return_value = None

            # Mock file collections to return minimal test data
            # Create a temporary file in the docs directory for the mock
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", dir=docs_dir, delete=False
            ) as temp_md:
                temp_md.write("# Test Document\n\nThis is a test markdown file.")
                temp_md_path = Path(temp_md.name)

            mock_markdown.return_value = [temp_md_path]
            mock_python.return_value = {"test.py": "Test docstring"}

            # Mock the docstore to avoid expensive embedding operations
            mock_docstore = MagicMock()
            mock_docstore_class.return_value = mock_docstore

            # This should not raise an exception
            build_docstore()

            # Check that the docstore directory was created
            assert Path(temp_dir).exists()

            # Verify docstore was called with correct parameters
            mock_docstore_class.assert_called_once_with(
                table_name="llamabot_docs", storage_path=Path(temp_dir)
            )

            # Verify documents were added (append should be called)
            assert mock_docstore.append.called

            # Clean up the temporary markdown file
            temp_md_path.unlink()


def test_create_mcp_server():
    """Test that the MCP server can be created."""
    import tempfile
    import shutil
    from unittest.mock import patch

    # Mock the docstore path to use a temporary directory
    with patch("llamabot.mcp_server.get_docstore_path") as mock_path:
        temp_dir = Path(tempfile.mkdtemp())
        mock_path.return_value = temp_dir

        try:
            # First build the docstore
            build_docstore()

            # Then create the server
            mcp = create_mcp_server()

            # Should be a FastMCP instance
            assert mcp is not None
            assert hasattr(mcp, "run")

            # Server should have a name
            assert hasattr(mcp, "name")
        finally:
            # Clean up
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


def test_cli_command_exists():
    """Test that the MCP CLI module exists and has the expected structure."""
    # Direct import to avoid CLI dependency issues
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "mcp", here() / "llamabot" / "cli" / "mcp.py"
    )
    mcp = importlib.util.module_from_spec(spec)
    sys.modules["mcp"] = mcp
    spec.loader.exec_module(mcp)

    # Should have a Typer app
    assert hasattr(mcp, "app")

    # Should have launch and build commands
    assert hasattr(mcp, "launch")
    assert hasattr(mcp, "build")

    # Clean up
    del sys.modules["mcp"]
