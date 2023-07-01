"""Zotero library wrappers."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .utils import load_zotero


@dataclass
class ZoteroLibrary:
    """Zotero library."""

    library_path: Path = field(
        default=Path.home() / ".llamabot/zotero/zotero_index.json"
    )

    def __post_init__(self):
        """Post-initialization hook"""
        library = [
            ZoteroItem(json.loads(line), library=self)
            for line in open(self.library_path, "r")
        ]
        self.library = {i["key"]: i for i in library}

    def __getitem__(self, key):
        """Get item by key.

        :param key: Key to get.
        :return: Value for key.
        """
        return self.library[key]

    def keys(self):
        """Return all of the keys from the library.

        :return: A list of keys.
        """
        return [i["key"] for i in self.library]


@dataclass
class ZoteroItem:
    """Zotero item."""

    info: dict
    library: Optional[ZoteroLibrary]

    def __getitem__(self, key):
        """Get item by key.

        :param key: Key to get.
        :return: Value for key.
        :raises KeyError: If key is not found.
        """
        # Key should be a string that is dot-delimited.
        # we split the string into a list of keys
        # Then we access the keys in order.

        keys = key.split(".")
        value = self.info
        for k in keys:
            try:
                value = value[k]
            except KeyError:
                raise KeyError(f"Key {k} not found in {value}.")
        return value

    def has_pdf(self):
        """Check if this item has a PDF or not.

        We check the "links" section, which typically looks like this:

        ```json
        {
            'self': {
                'href': 'https://api.zotero.org/users/5334442/items/K3WYABBQ',
                'type': 'application/json'
            },
            'alternate': {
                'href': 'https://www.zotero.org/ericmjl/items/K3WYABBQ',
                'type': 'text/html'
            },
            'attachment': {
                'href': 'https://api.zotero.org/users/5334442/items/U6X244QK',
                'type': 'application/json',
                'attachmentType': 'application/pdf',
                'attachmentSize': 4351619
            }
        }
        ```

        :return: True if this item has a PDF, False otherwise.
        """
        if self.get("links.attachment.attachmentType") == "application/pdf":
            return True
        return False

    def get(self, key_string, default_value=None):
        try:
            return self[key_string]
        except KeyError:
            return default_value

    def pdf(self):
        """Get the PDF entry for this item.

        :return: PDF entry.
        :raises KeyError: If no PDF is found.
        """
        if self.has_pdf():
            return self["links.attachment"]
        raise KeyError("No PDF found.")

    def download_pdf(self, directory: Path) -> Path:
        """Download the PDF for this item.

        :param directory: Directory to download the PDF to.
        :return: Path to the downloaded PDF.
        """
        if self.has_pdf():
            pdf = self.pdf()
            zot = load_zotero()
            key = pdf["href"].split("/")[-1]
            fpath = directory / f"{key}.pdf"
            with open(fpath, "wb") as f:
                f.write(zot.file(key))
            return fpath
        else:
            fpath = directory / "abstract.txt"
            with fpath.open("w+") as f:
                f.write()
