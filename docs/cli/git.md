---
intents:
- Provide a diataxis framework-style tutorial on how to use the LlamaBot Git CLI.
  Covers comprehensively all commands and possible options.
- Show how to use each of the commands in the `llamabot git` CLI.
linked_files:
- llamabot/cli/git.py
- llamabot/prompt_library/git.py
---

## LlamaBot Git CLI Documentation

Welcome to the LlamaBot Git CLI documentation. This guide provides a comprehensive tutorial on how to use the various commands available in the LlamaBot Git CLI, designed to enhance your Git experience with automated commit messages, release notes, and activity reports.

### Getting Started

Before you begin, ensure that you have the LlamaBot CLI installed on your system. You will also need to have Git installed and be within a Git repository to use most of the commands.

### Commands Overview

The LlamaBot Git CLI includes several commands, each tailored for specific Git-related tasks:

#### 1. `hooks`

**Purpose:** Installs a commit message hook that automatically generates commit messages using a structured bot.

**Usage:**

```bash
llamabot git hooks
```

This command sets up a Git hook in your repository that triggers the LlamaBot to compose commit messages if none are provided during commits.

#### 2. `compose`

**Purpose:** Automatically generates a commit message based on the current Git diff.

**Usage:**

```bash
llamabot git compose
```

Use this command to autogenerate a commit message which you can then review and edit as needed. This is particularly useful for ensuring commit messages are consistent and informative.

#### 3. `write_release_notes`

**Purpose:** Generates release notes for the latest tags in your repository.

**Usage:**

```bash
llamabot git write_release_notes
```

This command will create a markdown file in the specified directory containing release notes based on the commits between the last two tags.

#### 4. `report`

**Purpose:** Generates a report based on Git commit logs for a specified time frame.

**Usage:**

```bash
llamabot git report --hours 24
llamabot git report --start-date 2023-01-01 --end-date 2023-01-02
```

This command can be used to generate a detailed report of activities, highlighting key changes and features implemented within the specified period.

### Conclusion

The LlamaBot Git CLI is a powerful tool for automating and enhancing your Git workflow. By understanding and utilizing these commands, you can significantly improve the efficiency and consistency of your version control practices.
