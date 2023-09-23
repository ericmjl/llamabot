# LlamaBot Git CLI Tutorial

In this tutorial, we will explore the Git subcommand for the LlamaBot CLI.
This command-line interface (CLI) provides a set of tools
to automate and enhance your Git workflow,
in particular, the ability to automatically generate commit messages.

## Setup

The `llamabot` prepare message hook requires that you have `llamabot >=0.0.77`.
You will also need an OpenAI API key
(until we have enabled and tested locally-hosted language models).
Be sure to setup and configure LlamaBot
by executing the following two configuration commands
and following the instructions there.

```bash
llamabot configure api-key
```

and

```bash
llamabot configure default-model
```

For the default model, we suggest using a GPT-4 variant.
It is generally of higher quality than GPT-3.5.
If you are concerned with cost,
the GPT-3.5-turbo variant with 16K context window
has anecdotally worked well.

## Install the Commit Message Hook

Once you have configured `llamabot`, the next thing you need to do is install the `prepare-msg-hook` within your `git` repository.
This is a `git` hook that allows you to run commands after the `pre-commit` hooks are run
but before your editor of the commit message is opened.
To install the hook, simply run:

```bash
llamabot git install-commit-message-hook
```

This command will check if the current directory is a Git repository root.
If it is not, it raises a `RuntimeError`.
If it is, it writes a script to the `prepare-commit-msg` file in the `.git/hooks` directory
and changes the file's permissions to make it executable.

## Auto-Compose a Commit Message

The `llamabot git compose-commit` command autowrites a commit message based on the diff.
It first gets the diff using the `get_git_diff` function.
It then generates a commit message using the `commitbot`, which is a LlamaBot SimpleBot.
If any error occurs during this process,
it prints the error message and prompts the user to write their own commit message,
allowing for a graceful fallback to default behaviour.
This can be useful, for example, if you don't have an internet connection
and cannot connect to the OpenAI API,
but still need to commit code.

This command never needs to be explicitly called.
Rather, it is called behind-the-scenes within the `prepare-msg-hook`.

## Conclusion

The `llamabot git` CLI provides a set of tools to automate and enhance your Git workflow.
It provides an automatic commit message writer based on your repo's `git diff`.
By using `llamabot git`, you can streamline your Git workflow and focus on writing code.
