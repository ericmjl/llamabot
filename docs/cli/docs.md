---
intents:
- Provide reader with a guided tutorial that shows how to use the `llamabot docs write`
  command at the terminal to generate documentation.
- Explain the mechanism by which linked files are referenced.
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

## `--from-scratch` Flag

The `--from-scratch` flag is an optional parameter that you can use with the `llamabot docs write` command. When this flag is set to `True`, the command will start with a blank documentation, ignoring any existing content in the Markdown file. This is useful when you want to completely regenerate the documentation from scratch.

To use the `--from-scratch` flag, run the following command:

```sh
llamabot docs write <path_to_markdown_file> --from-scratch
```

## Frontmatter Key-Value Pairs

For the `llamabot docs write` command to work correctly, your Markdown source file must contain specific frontmatter key-value pairs. The frontmatter should be written in YAML format and placed at the top of the Markdown file. Here is an example of the required frontmatter:

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

### Intents

The `intents` key is a list of points that the documentation should cover. These points guide the LLM in generating the content of the documentation.

### Linked Files

The `linked_files` key is a list of paths to relevant source files that the documentation should reference. These paths must be relative to the root of the repository.

## Example

Here is an example of a complete Markdown source file with the required frontmatter and some initial content:

```markdown
---
intents:
- Provide an overview of the `llamabot docs write` command.
- Explain the `--from-scratch` flag.
- Describe the frontmatter key-value pairs needed to make it work.
linked_files:
- llamabot/cli/docs.py
- pyproject.toml
---

# CLI Documentation for `llamabot docs write`

<The documentation content will be generated here.>
```

By following these guidelines, you can effectively use the `llamabot docs write` command to generate and maintain high-quality documentation for your project.

## How Linked Files are Referenced

The `llamabot docs write` command references linked files specified in the `linked_files` key of the frontmatter. These files are read and their content is used to inform the generated documentation. The paths to these files must be relative to the root of the repository. For example, if you have a file `llamabot/cli/docs.py` that you want to reference, you would include it in the `linked_files` list as shown in the example above.

By understanding and utilizing these features, you can ensure that your documentation is comprehensive, up-to-date, and aligned with the source code and project intents.
