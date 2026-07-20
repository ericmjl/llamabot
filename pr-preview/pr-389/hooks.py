"""MkDocs hooks for pre-build processing."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_conversion_function():
    """Load the convert_marimo_to_markdown function from scripts directory."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    script_path = scripts_dir / "convert_marimo_to_markdown.py"

    spec = spec_from_file_location("convert_marimo_to_markdown", script_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.convert_marimo_to_markdown


convert_marimo_to_markdown = _load_conversion_function()


def on_pre_build(config, **kwargs):
    """Run before MkDocs builds the documentation.

    :param config: The MkDocs configuration object
    :param kwargs: Additional keyword arguments
    """
    print("Running pre-build hook: Converting Marimo notebooks to Markdown...")
    convert_marimo_to_markdown()
    print("Pre-build hook completed.")
