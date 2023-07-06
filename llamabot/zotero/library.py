"""Zotero library wrappers."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from pyzotero.zotero import Zotero
from rich.progress import Progress, SpinnerColumn, TextColumn

from .utils import load_zotero

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=False,
)


@dataclass
class ZoteroLibrary:
    """Zotero library object.

    Stores a list of Zotero items.
    """

    zot: Zotero = field(default_factory=load_zotero)
    json_dir: Path = field(default=None)

    def __post_init__(self):
        """Post-initialization hook

        If json_dir is set, load the library from the JSON files in that directory
        and skip querying zotero for everything.
        """
        if self.json_dir is not None:
            # Load the library from the JSON files.
            items = []
            for json_file in self.json_dir.glob("*.json"):
                with json_file.open("r+") as f:
                    items.append(json.loads(f.read()))
        else:
            with progress:
                task = progress.add_task("Synchronizing your Zotero library...")
                items = self.zot.everything(self.zot.items())
                progress.remove_task(task)
        library = [ZoteroItem(i, library=self) for i in items]
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

    def to_json(self, dir: Path, has_pdf=True):
        """Save the library to a JSON file.

        :param dir: Directory in which to save the JSONs.
        :param has_pdf: Whether to save only items with PDFs.
        """
        for key, item in self.library.items():
            if has_pdf and item.has_pdf():
                with open((dir / f"{key}.json"), "w+") as f:
                    f.write(json.dumps(item.info) + "\n")


@dataclass
class ZoteroItem:
    """Zotero item."""

    info: dict
    library: ZoteroLibrary

    def __getitem__(self, key):
        """Get item by key.

        Allows for accessing nested keys via a "."-delimited string.

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
        pdf = self.pdf()
        key = pdf["href"].split("/")[-1]
        fpath = directory / f"{key}.pdf"
        with fpath.open("wb") as f:
            f.write(self.library.zot.file(key))
        return fpath

    def download_abstract(self, directory: Path) -> Path:
        """Download the abstract for this item.

        :param directory: Directory to download the abstract to.
        :return: Path to the downloaded abstract.
        """
        fpath = directory / "abstract.txt"
        with fpath.open("w+") as f:
            f.write(self["data"]["abstractNote"])
        return fpath
