"""Example usage of the Feynman panel bot.

Execute this from the root directory of this repo using:

```bash
panel serve examples/feynman_panel.py
```
"""

from llamabot import SimpleBot

feynman = SimpleBot(
    "You are Richard Feynman. "
    "You will be given a difficult concept, and your task is to explain it back."
)

feynman.panel(show=False).servable()
