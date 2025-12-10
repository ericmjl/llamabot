"""Convert Marimo notebooks to Markdown with Molab shield injection."""

import subprocess
from pathlib import Path


def convert_marimo_to_markdown():
    """Convert all Marimo notebooks in docs/how-to/ to Markdown.

    This script:
    - Finds all .py files in docs/how-to/
    - Converts them to .md using marimo export (via uvx)
    - Injects Molab shield badge at the top
    - Preserves any existing frontmatter
    """
    how_to_dir = Path("docs/how-to")

    if not how_to_dir.exists():
        print(f"Directory {how_to_dir} does not exist. Skipping conversion.")
        return

    for notebook in how_to_dir.glob("*.py"):
        if notebook.stem.startswith("_"):
            continue  # Skip private files

        md_file = notebook.with_suffix(".md")
        print(f"Converting {notebook.name} to {md_file.name}...")

        try:
            # Run marimo export using uvx (no need to install marimo separately)
            subprocess.run(
                [
                    "uvx",
                    "marimo",
                    "export",
                    "md",
                    str(notebook),
                    "--output",
                    str(md_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Read generated markdown
            content = md_file.read_text()

            # Check if Molab shield already exists
            if "[![Open in molab]" in content:
                print(
                    f"  Molab shield already exists in {md_file.name}, skipping injection."
                )
                continue

            # Generate Molab link
            repo_path = f"docs/how-to/{notebook.name}"
            molab_shield = (
                f"[![Open in molab](https://marimo.io/molab-shield.svg)]"
                f"(https://molab.marimo.io/github/ericmjl/llamabot/blob/main/{repo_path})\n\n"
            )

            # Prepend shield (preserve frontmatter if present)
            if content.startswith("---"):
                # Has frontmatter - find where it ends (second ---)
                frontmatter_end = content.find("---", 3)
                if frontmatter_end != -1:
                    # Insert shield after frontmatter (after the closing --- and newline)
                    frontmatter_end += 3
                    # Skip any whitespace/newlines after frontmatter
                    while (
                        frontmatter_end < len(content)
                        and content[frontmatter_end] in "\n\r"
                    ):
                        frontmatter_end += 1
                    content = (
                        content[:frontmatter_end]
                        + molab_shield
                        + content[frontmatter_end:]
                    )
                else:
                    # Malformed frontmatter, just prepend
                    content = molab_shield + content
            else:
                # No frontmatter, just prepend
                content = molab_shield + content

            # Ensure file ends with a single newline
            if not content.endswith("\n"):
                content += "\n"
            elif content.endswith("\n\n"):
                # Remove extra newlines, keep only one
                content = content.rstrip("\n") + "\n"

            md_file.write_text(content)
            print(f"  Successfully converted {notebook.name}")

        except subprocess.CalledProcessError as e:
            print(f"  Error converting {notebook.name}: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
        except Exception as e:
            print(f"  Unexpected error converting {notebook.name}: {e}")


if __name__ == "__main__":
    convert_marimo_to_markdown()
