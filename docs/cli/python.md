# Llamabot Python CLI Tutorial

Welcome to the Llamabot Python CLI tutorial! In this tutorial, we will explore the various commands available in the Llamabot Python CLI and learn how to use them effectively. The Llamabot Python CLI is a powerful tool for generating module-level and function docstrings, as well as generating code based on a given description.

## Prerequisites

Before we begin, make sure you have the Llamabot Python CLI installed on your system. You can install it using pip:

```bash
pip install -U llamabot
```

Once installed, you can access the CLI using the `llamabot python` command.

## Commands

The Llamabot Python CLI provides the following commands:

1. `module-docstrings`: Generate module-level docstrings for a given module file.
2. `generate-docstrings`: Generate function docstrings for a specific function in a module file.
3. `code-generator`: Generate code based on a given description.
4. `test-writer`: Write tests for a given object.

Let's dive into each command and see how they can be used.

### 1. module-docstrings

The `module-docstrings` command generates module-level docstrings for a given module file. It takes the following arguments:

- `module_fpath`: Path to the module to generate docstrings for.
- `dirtree_context_path`: (Optional) Path to the directory to use as the context for the directory tree. Defaults to the parent directory of the module file.

Example usage:

```bash
llamabot python module-docstrings /path/to/your/module.py
```

To specify a custom directory tree context path, use the following command:

```bash
llamabot python module-docstrings /path/to/your/module.py /path/to/your/directory
```

### 2. generate-docstrings

The `generate-docstrings` command generates function docstrings for a specific function in a module file. It takes the following arguments:

- `module_fpath`: Path to the module to generate docstrings for.
- `object_name`: Name of the object to generate docstrings for.
- `style`: (Optional) Style of docstring to generate. Defaults to "sphinx".

Example usage:

```bash
llamabot python generate-docstrings /path/to/your/module.py function_name
```

To specify a custom docstring style, use the following command:

```bash
llamabot python generate-docstrings /path/to/your/module.py function_name google
```

### 3. code-generator

The `code-generator` command generates code based on a given description. It takes the following argument:

- `request`: A description of what the code should do.

Example usage:

```bash
llamabot python code-generator "Create a function that adds two numbers"
```

### 4. test-writer

The `test-writer` command writes tests for a given object. It takes the following arguments:

- `module_fpath`: Path to the module to generate tests for.
- `object_name`: Name of the object to generate tests for.

Example usage:

```bash
llamabot python test-writer /path/to/your/module.py function_name
```

## Conclusion

In this tutorial, we have covered the various commands available in the Llamabot Python CLI and learned how to use them effectively. With these commands, you can easily generate module-level and function docstrings, generate code based on a given description, and write tests for your code. Happy coding!
