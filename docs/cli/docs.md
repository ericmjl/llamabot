---
intents:
- Provide reader with a guided tutorial that shows how to use the `llamabot docs write`
  command at the terminal to generate documentation.
- Specifically add in information about the `--from-scratch` flag.
- Describe the frontmatter key-value pairs needed to make it work.
linked_files:
- llamabot/cli/docs.py
- pyproject.toml
---

# CLI Documentation for `llamabot docs write`

## Overview

The `llamabot docs write` command is a powerful tool designed to help you create and maintain Markdown documentation for your project. This command leverages the capabilities of LLMs (Large Language Models) to generate and update documentation based on the content and intents specified in your Markdown source files.

## Usage

To use the `llamabot docs write` command, open your terminal and navigate to the root directory of your project. Then, run the following command:

```sh
llamabot docs write <path_to_markdown_file>
```

Replace `<path_to_markdown_file>` with the path to the Markdown file you want to generate or update documentation for.

## Flags

### `--from-scratch`

The `--from-scratch` flag allows you to start with a blank documentation file. When this flag is used, the existing content of the Markdown file will be cleared before generating new documentation. This can be useful if you want to completely rewrite the documentation from scratch.

Usage example:

```sh
llamabot docs write <path_to_markdown_file> --from-scratch
```

## Frontmatter Key-Value Pairs

To make the `llamabot docs write` command work effectively, your Markdown source file should include specific frontmatter key-value pairs. These pairs provide the necessary context and instructions for generating the documentation.

Here is an example of the required frontmatter:

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

### `intents`

The `intents` key is a list of points that the documentation should cover. These points guide the content generation process, ensuring that the generated documentation aligns with your specific needs and goals.

### `linked_files`

The `linked_files` key is a list of file paths to relevant source files. These files provide additional context and information that can be referenced in the generated documentation. The paths should be relative to the root of your project repository.

## Example

Here is an example of how to use the `llamabot docs write` command with a Markdown source file that includes the necessary frontmatter:

1. Create a Markdown file (e.g., `docs/cli/docs.md`) with the following content:

    ```markdown
    ---
    intents:
    - Provide an overview of the `llamabot docs write` command.
    - Explain the `--from-scratch` flag.
    - Describe the required frontmatter key-value pairs.
    linked_files:
    - llamabot/cli/docs.py
    - pyproject.toml
    ---

    <The documentation is empty.>
    ```

2. Run the `llamabot docs write` command:

    ```sh
    llamabot docs write docs/cli/docs.md
    ```

3. The command will generate and update the documentation based on the specified intents and linked files.

By following these steps, you can easily create and maintain comprehensive documentation for your project using the `llamabot docs write` command.
