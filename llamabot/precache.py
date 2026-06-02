"""Pre-cache utilities for llamabot embedding models."""

DEFAULT_MODEL = "minishlab/potion-base-8M"


def precache_embedding_model(model_name: str = DEFAULT_MODEL) -> str:
    """Download and cache *model_name* if not already present.

    :param model_name: HuggingFace model identifier.
    :return: The model name that was cached.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install it with: pip install llamabot[rag]"
        )

    print(f"Pre-caching embedding model: {model_name}")
    SentenceTransformer(model_name, trust_remote_code=True)
    print(f"Done — {model_name} is now cached locally.")
    return model_name
