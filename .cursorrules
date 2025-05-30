You are an expert Pythonista. You are familiar with the Python data science stack.

You are also an expert in prompting LLMs to do stuff.

General rules:

- If you make a change in code, propose always add tests to test that code as well.

Docstrings:

- Use sphinx-style arguments (e.g. `:param arg: description`)

Style:

- Prefer functional programming over objects, except for the Bots submodule, which are PyTorch-like parameterized callable objects.

Stack:

- In this repo, we use Typer to build command line apps.
- pytest is used for testing.
- Use HTMX + FastAPI endpoints to do things that you would otherwise need to in JavaScript.

Editing:

- As much as possible, only edit the file that I ask you to edit.
- Don't worry about linting errors. I have automatic linting tools.
- If I ask you to diagnose a problem, do not jump into editing. Present the diagnosis, and wait for me to confirm whether I want to apply the fix, as I may sometimes disagree with the diagnosis.

Templates and UI:

- Use Jinja2 macros (defined in macros.html) for any UI component that appears in multiple templates.
- When creating a new UI component that might be reused:
  1. Create a macro in macros.html with clear parameter documentation
  2. Use the macro in all templates that need the component
  3. Ensure the macro preserves HTMX attributes and functionality
- Never duplicate UI component code across templates - always use macros for shared components.
- When adding features to a UI component:
  1. Update the macro in macros.html
  2. All templates using the macro will automatically get the new feature
- Keep macros focused and single-purpose - if a macro grows too complex, consider breaking it into smaller macros.

Notebooks:

- If I ask you to generate a Marimo notebook, it generally looks like this:

---

Client-side JS initialization for dynamic HTML (HTMX, Turbo, AJAX):

- When using HTMX, Turbo, or any tool that swaps in HTML fragments dynamically, client-side JS initialization (event listeners, widgets, etc.) must be re-applied after new content is loaded.
- Inline <script> tags in swapped-in HTML are NOT executed by browsers for security reasons.
- To ensure all interactive elements (even those loaded dynamically) get their JS behaviors:
  1. Encapsulate your JS initialization logic in a function (e.g., `function initMyWidget() { ... }`).
  2. Call this function on page load (`DOMContentLoaded`).
  3. Call this function after HTMX swaps (`htmx:afterSwap`), Turbo frame loads, or AJAX completions as appropriate.
- Never rely on inline <script> in HTML fragments. Always use external JS files and initialization functions.
- Make your init functions idempotent (safe to call multiple times).

Summary Table:
| When?                | What to do?                |
|----------------------|---------------------------|
| On page load         | Call your init function   |
| After HTMX swap      | Call your init function   |
| After AJAX/Turbo/etc | Call your init function   |
