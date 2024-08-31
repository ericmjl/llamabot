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

# CLI Documentation for `llamabot docs write`

## Overview

The `llamabot docs write` command is a powerful tool designed to help you create and maintain Markdown documentation for your project. This command leverages the capabilities of LLMs (Large Language Models) to generate and update documentation based on the content and intent specified in your Markdown files.

## Usage

To use the `llamabot docs write` command, navigate to your terminal and run the following command:

```sh
llamabot docs write <path_to_markdown_file>
```

Replace `<path_to_markdown_file>` with the path to the Markdown file you want to generate or update documentation for.

## Options

### `--force`

The `--force` flag can be used to force the regeneration of the documentation, even if it is not out of date. This is useful if you want to ensure that the documentation is always up-to-date with the latest content and intent.

```sh
llamabot docs write <path_to_markdown_file> --force
```

## Frontmatter Key-Value Pairs

To make the `llamabot docs write` command work effectively, your Markdown files should include specific frontmatter key-value pairs. These pairs provide the necessary context and intent for the documentation generation process.

### Example Frontmatter

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

### Explanation

- **intents**: A list of points that the documentation should cover. These points guide the content generation process to ensure that the documentation is aligned with your goals.
- **linked_files**: A list of paths to relevant files that should be referenced in the documentation. These files provide additional context and content that can be included in the generated documentation.

## Example

Here is an example of how to use the `llamabot docs write` command:

1. Create a Markdown file with the necessary frontmatter:

```markdown
---
intents:
- Explain how to use the `llamabot docs write` command.
- Describe the `--force` flag.
- Provide an example of the frontmatter key-value pairs.
linked_files:
- path/to/relevant_file1.py
- path/to/relevant_file2.toml
---
```

2. Run the `llamabot docs write` command:

```sh
llamabot docs write docs/cli/docs.md
```

3. The command will generate or update the documentation based on the specified intents and linked files.

## Conclusion

The `llamabot docs write` command is a valuable tool for automating the creation and maintenance of your project's documentation. By specifying the intents and linked files in the frontmatter, you can ensure that your documentation is always accurate and up-to-date.
