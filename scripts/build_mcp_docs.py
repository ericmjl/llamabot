#!/usr/bin/env python3
"""Build the MCP documentation database for packaging.

This script is run during CI/CD to build the LanceDB database
that gets packaged with the wheel distribution.
"""

from pathlib import Path
import shutil

from llamabot.mcp_server import build_docstore, get_docstore_path


def main():
    """Build the MCP documentation database for packaging."""
    print("Building MCP documentation database for packaging...")

    # Build the database
    build_docstore()

    # Copy the built database to a location that will be packaged
    source_path = get_docstore_path()
    package_path = Path(__file__).parent.parent / "llamabot" / "data" / "mcp_docs"

    # Create the package data directory
    package_path.mkdir(parents=True, exist_ok=True)

    # Copy the database
    if source_path.exists():
        if package_path.exists():
            shutil.rmtree(package_path)
        shutil.copytree(source_path, package_path)
        print(f"Database copied to {package_path}")
    else:
        print(f"Error: Database not found at {source_path}")
        exit(1)


if __name__ == "__main__":
    main()
