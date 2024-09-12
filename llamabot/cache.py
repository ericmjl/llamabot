"""
This module provides a llamabot global cache for caching responses from bots.
"""

from diskcache import Cache
from pathlib import Path

CACHE_DIR = Path.home() / ".llamabot" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

cache = Cache(CACHE_DIR, timeout=86400)  # 86400 seconds = 1 day
