"""Tests for TurboVecDocStore."""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip(
    "turbovec",
    reason="turbovec is not installed; install with `pip install llamabot[turbovec]`",
)

from llamabot.components.docstore import TurboVecDocStore


@pytest.fixture(scope="module")
def turbovec_store():
    """Return a reusable TurboVecDocStore instance.

    This fixture creates a single store instance that is reused across tests
    to avoid expensive model loading. Each test should call .reset() to ensure
    fresh state.
    """
    temp_dir = Path(tempfile.mkdtemp())
    store = TurboVecDocStore(table_name="test_turbovec", storage_path=temp_dir)
    yield store
    try:
        store.reset()
        import shutil

        shutil.rmtree(temp_dir)
    except Exception:
        pass


def test_turbovec_append(turbovec_store):
    """Test TurboVecDocStore append method."""
    turbovec_store.reset()

    document = "Python is a programming language with clean syntax."
    turbovec_store.append(document)

    retrieved = turbovec_store.retrieve("Python programming", n_results=1)
    assert len(retrieved) == 1
    assert retrieved[0] == document


def test_turbovec_extend(turbovec_store):
    """Test TurboVecDocStore extend method."""
    turbovec_store.reset()

    documents = [
        "Python is a programming language with clean syntax.",
        "FastAPI is a modern web framework for building APIs.",
        "LanceDB is a vector database for storing embeddings.",
    ]
    turbovec_store.extend(documents)

    python_docs = turbovec_store.retrieve("Python programming", n_results=1)
    assert len(python_docs) == 1
    assert "Python" in python_docs[0]

    all_docs = turbovec_store.retrieve("programming frameworks databases", n_results=3)
    assert len(all_docs) == 3
    assert set(all_docs) == set(documents)


def test_turbovec_append_avoids_duplicates(turbovec_store):
    """Test that append skips duplicate documents."""
    turbovec_store.reset()

    document = "This is a unique test document."
    turbovec_store.append(document)
    turbovec_store.append(document)

    assert turbovec_store.existing_records.count(document) == 1


def test_turbovec_extend_avoids_duplicates(turbovec_store):
    """Test that extend skips duplicate documents."""
    turbovec_store.reset()

    turbovec_store.append("Document one")
    turbovec_store.extend(["Document one", "Document two"])

    assert len(turbovec_store.existing_records) == 2


def test_turbovec_retrieve_empty(turbovec_store):
    """Test retrieve on an empty store returns empty list."""
    turbovec_store.reset()

    results = turbovec_store.retrieve("anything", n_results=5)
    assert results == []


def test_turbovec_contains(turbovec_store):
    """Test __contains__ method."""
    turbovec_store.reset()

    document = "Test document for membership check"
    turbovec_store.append(document)

    assert document in turbovec_store
    assert "Non-existent document" not in turbovec_store


def test_turbovec_reset(turbovec_store):
    """Test reset clears all data."""
    turbovec_store.reset()

    turbovec_store.append("Some document")
    turbovec_store.append("Another document")

    turbovec_store.reset()

    assert turbovec_store.existing_records == []
    assert turbovec_store.id_to_doc == {}
    assert turbovec_store.retrieve("document", n_results=5) == []


def test_turbovec_persistence():
    """Test that data persists across store instances."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = Path(temp_dir)

        store1 = TurboVecDocStore(table_name="test_persistence", storage_path=storage)
        store1.reset()
        store1.append("Persistent document one")
        store1.append("Persistent document two")

        store2 = TurboVecDocStore(table_name="test_persistence", storage_path=storage)
        assert "Persistent document one" in store2
        assert "Persistent document two" in store2

        results = store2.retrieve("Persistent", n_results=2)
        assert len(results) == 2


def test_turbovec_n_results_capped(turbovec_store):
    """Test that retrieve caps results at available documents."""
    turbovec_store.reset()

    turbovec_store.extend(["Doc A", "Doc B"])
    results = turbovec_store.retrieve("document", n_results=10)
    assert len(results) == 2


class FakeTransformer:
    """Embedder double that records construction and reports a fixed width.

    :param name: Sentence-transformers model name.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def get_sentence_embedding_dimension(self) -> int:
        """Return a fixed vector width.

        :return: Embedding dimension.
        """
        return 8


def test_shared_sentence_transformer_reuses_loaded_model(monkeypatch):
    """The same model name must not construct a second transformer."""
    from llamabot.components import docstore

    constructed: list[str] = []

    class RecordingTransformer(FakeTransformer):
        """Count how many times the cache constructs a transformer.

        :param name: Sentence-transformers model name.
        """

        def __init__(self, name: str) -> None:
            constructed.append(name)
            super().__init__(name)

    monkeypatch.setattr(docstore, "SENTENCE_TRANSFORMER_CACHE", {})
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        RecordingTransformer,
    )

    first = docstore.shared_sentence_transformer("fake-model")
    second = docstore.shared_sentence_transformer("fake-model")

    assert first is second
    assert constructed == ["fake-model"]


def test_turbovec_stores_share_sentence_transformer(tmp_path, monkeypatch):
    """Two stores use the shared transformer instead of loading it twice."""
    from llamabot.components import docstore

    shared = FakeTransformer("fake-model")

    def loader(model_name: str) -> FakeTransformer:
        """Return the one shared double for any requested model.

        :param model_name: Model name requested by the store.
        :return: The shared transformer double.
        """
        assert model_name == "minishlab/potion-base-8M"
        return shared

    monkeypatch.setattr(docstore, "shared_sentence_transformer", loader)

    store1 = TurboVecDocStore(table_name="share-one", storage_path=tmp_path / "one")
    store2 = TurboVecDocStore(table_name="share-two", storage_path=tmp_path / "two")

    assert store1.embedder is shared
    assert store2.embedder is shared


def test_turbovec_import_error(tmp_path, monkeypatch):
    """Test that missing turbovec raises ImportError with helpful message."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        """Block turbovec imports to simulate missing package."""
        if name == "turbovec":
            raise ModuleNotFoundError("No module named 'turbovec'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="turbovec is required"):
        TurboVecDocStore(table_name="test_import", storage_path=tmp_path)
