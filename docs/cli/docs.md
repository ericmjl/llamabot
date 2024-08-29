---
intents:
- Provide reader with a guided tutorial that shows how to use the `llamabot docs write`
  command at the terminal to generate documentation.
- Specifically add in information about the `--force` flag.
- Describe the frontmatter key-value pairs needed to make it work.
linked_files:
- llamabot/cli/docs.py
- pyproject.toml
---

# LlamaBot Documentation

## Installation

To install LlamaBot, follow these steps:

1. Ensure you have Python 3.10 or higher installed on your system.
2. Install LlamaBot using pip:

```bash
pip install llamabot
```

3. Verify the installation by running:

```bash
llamabot --help
```

## Usage

To use LlamaBot to generate documentation, run the following command in the terminal:

```bash
llamabot docs write <path_to_markdown_file>
```

### Options

- `--force`: Use this flag to force the documentation update even if it is not detected as out of date.

## Additional Information

For more information, refer to the [official documentation](https://llamabotdocs.com).

## Explanation

To make the `llamabot docs write` command work, ensure that your Markdown source files are properly formatted and located in the correct directory. The command will read the source files, check if the documentation is up-to-date, and update it if necessary. If you use the `--force` flag, the documentation will be updated regardless of its current status.

### Requirements for Target Documentation File

1. **Proper Formatting**: Ensure your Markdown files are correctly formatted.
2. **Correct Directory**: Place your Markdown files in the appropriate directory as expected by the `llamabot docs write` command.
3. **Linked Files**: If your documentation references other files, ensure these are correctly linked and accessible.

### Frontmatter Key-Value Pairs

To use the `llamabot docs write` command effectively, your Markdown files should include the following frontmatter:

```markdown
---
intents:
- Point 1 that the documentation should cover.
- Point 2 that the documentation should cover.
- ...
linked_files:
- path/to/relevant_file1.py
- path/to/relevant_file2.toml
- ...
---
```

By following these guidelines, you can effectively use LlamaBot to manage and update your documentation.
