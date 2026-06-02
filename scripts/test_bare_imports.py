"""Verify that core llamabot imports work without rag extras (no numpy).

This script is the CI gate for issue #372:
    https://github.com/ericmjl/llamabot/issues/372

It must pass when llamabot is installed with `pip install .` (no extras).
If numpy is accidentally pulled into the core import chain, this script fails.
"""

import sys

if "numpy" in sys.modules:
    print("FAIL: numpy already in sys.modules before llamabot import")
    sys.exit(1)

from llamabot import (
    SimpleBot,
)

if "numpy" in sys.modules:
    print("FAIL: numpy was pulled in by core llamabot imports")
    print("Modules loaded:", sorted(m for m in sys.modules if "numpy" in m))
    sys.exit(1)

bot = SimpleBot("test", model_name="gpt-4o-mini", stream_target="none")
assert isinstance(bot, SimpleBot)

print("OK: core llamabot imports work without numpy")
