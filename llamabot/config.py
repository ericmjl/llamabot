"""Configuration for llamabot: paths, defaults, and library settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

llamabotrc_path = Path.home() / ".llamabot/.llamabotrc"

llamabot_config_dir = Path.home() / ".llamabot"


def configured_litellm():
    """Import litellm lazily and return it with llamabot's settings applied.

    Importing litellm eagerly costs multiple seconds, so llamabot never
    imports it at module level. Any function that calls into litellm should
    obtain the module through this accessor instead, which guarantees our
    settings are applied before the first call:

    - ``suppress_debug_info`` silences litellm's noisy "Provider List: ..."
      banner, printed whenever ``get_llm_provider()`` cannot infer a provider
      from a bare model name (e.g. the default ``gpt-4.1``). The banner
      precedes a ``BadRequestError`` that callers already handle, so it is
      pure noise. litellm reads this flag at call time, so setting it here
      (before any litellm call) is equivalent to setting it at import time.

    The settings are idempotent; calling this accessor repeatedly is safe.

    :return: The litellm module with llamabot's settings applied.
    """
    import litellm

    litellm.suppress_debug_info = True
    return litellm


def default_language_model():
    """Return the default language model to be used.

    :return: The default language model to be used.
    """

    load_dotenv(llamabotrc_path)

    return os.getenv("DEFAULT_LANGUAGE_MODEL", "gpt-4.1")
