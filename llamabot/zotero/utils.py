"""Zotero utilities."""

import os


def load_zotero():
    """Load Zotero API client.

    :return: Zotero API client
    """
    try:
        from pyzotero import zotero
    except ImportError:
        raise ImportError(
            "pyzotero is not installed. "
            "Please install it with: `pip install llamabot[cli]`"
        )

    zotero_library_id = os.environ.get("ZOTERO_LIBRARY_ID", None)
    zotero_library_type = os.environ.get("ZOTERO_LIBRARY_TYPE", None)
    zotero_api_key = os.environ.get("ZOTERO_API_KEY", None)

    return zotero.Zotero(
        library_id=zotero_library_id,
        library_type=zotero_library_type,
        api_key=zotero_api_key,
    )
