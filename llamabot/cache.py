"""
This module provides a llamabot global cache for caching responses from bots.
"""

from diskcache import Cache
from pathlib import Path
import os

CACHE_DIR = Path.home() / ".llamabot" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Get cache timeout from environment variable, default to 1 day (86400 seconds)
CACHE_TIMEOUT = int(os.getenv("LLAMABOT_CACHE_TIMEOUT", 86400))

cache = Cache(CACHE_DIR, timeout=CACHE_TIMEOUT)
