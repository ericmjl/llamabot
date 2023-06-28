"""Zotero library wrappers."""
import json
from dataclasses import dataclass
from pathlib import Path

from .utils import load_zotero


@dataclass
class ZoteroLibrary:
    """Zotero library."""

    library_path: Path = Path.home() / ".llamabot/zotero/zotero_index.json"

    def __post_init__(self):
        """Post-initialization hook"""
        self.library = [json.loads(line) for line in open(self.library_path, "r")]

    def __getitem__(self, key):
        """Get item by key.

        :param key: Key to get.
        :return: Value for key.
        """
        return ZoteroItem([i for i in self.library if i["key"] == key][0], library=self)


@dataclass
class ZoteroItem:
    """Zotero item."""

    info: dict
    library: ZoteroLibrary

    def __getitem__(self, key):
        """Get item by key.

        :param key: Key to get.
        :return: Value for key.
        """
        # Key should be a string that is dot-delimited.
        # we split the string into a list of keys
        # Then we access the keys in order.

        keys = key.split(".")
        value = self.info
        for k in keys:
            value = value[k]
        return value

    def pdf(self):
        """Get the PDF entry for this item.

        :return: PDF entry.
        """
        pdf_url = self["links.attachment.href"]
        pdf_key = pdf_url.split("/")[-1]
        pdf_entry = self.library[pdf_key]
        return pdf_entry

    def download_pdf(self, directory: Path) -> Path:
        """Download the PDF for this item.

        :param directory: Directory to download the PDF to.
        :return: Path to the downloaded PDF.
        """
        pdf = self.pdf()
        zot = load_zotero()
        fpath = directory / f"{pdf['key']}.pdf"
        with open(fpath, "wb") as f:
            f.write(zot.file(pdf["key"]))
        return fpath
