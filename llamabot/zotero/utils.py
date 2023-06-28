"""Zotero utilities."""
import os

from pyzotero import zotero


def load_zotero():
    """Load Zotero API client.

    :return: Zotero API client
    """
    zotero_library_id = os.environ.get("ZOTERO_LIBRARY_ID", None)
    zotero_library_type = os.environ.get("ZOTERO_LIBRARY_TYPE", None)
    zotero_api_key = os.environ.get("ZOTERO_API_KEY", None)

    return zotero.Zotero(
        library_id=zotero_library_id,
        library_type=zotero_library_type,
        api_key=zotero_api_key,
    )
