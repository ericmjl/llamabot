"""Pre-cache llamabot's default embedding model.

Downloading ``minishlab/potion-base-8M`` (~60 MB) at *install* time avoids
the first-run latency penalty — especially important inside Docker containers
where every cold start would otherwise hit the network.

Usage::

    # standalone script
    pixi run python scripts/precache_embeddings.py

    # via CLI (after installation)
    llamabot precache

    # in a Dockerfile
    RUN pip install llamabot[rag] && python -c "from llamabot.precache import precache_embedding_model; precache_embedding_model()"
"""

import sys

from llamabot.precache import DEFAULT_MODEL, precache_embedding_model

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    precache_embedding_model(model)
