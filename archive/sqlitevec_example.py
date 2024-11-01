"""Example script demonstrating the use of SQLiteVecDocStore."""

from pathlib import Path
from llamabot.components.docstore import SQLiteVecDocStore


def main():
    """Example script demonstrating the use of SQLiteVecDocStore."""
    # Initialize the document store
    store = SQLiteVecDocStore(
        db_path=Path("example_docs.db"),
        table_name="documents",
        embedding_model="all-MiniLM-L6-v2",  # This is the default, but showing it explicitly
    )

    # Example documents - some movie plots
    documents = [
        "A young wizard discovers he has magical powers and attends a school of witchcraft and wizardry.",
        "Two hobbits journey to destroy a powerful ring in a volcano while being pursued by dark forces.",
        "A group of superheroes must work together to save Earth from an alien invasion.",
        "A criminal mastermind plants ideas in people's minds through their dreams.",
        "A computer programmer discovers that reality is a simulation created by machines.",
    ]

    # Add documents to the store
    print("Adding documents to the store...")
    for doc in documents:
        store.append(doc)

    # Perform some example queries
    print("\nPerforming semantic searches:")

    # Query for fantasy-related content
    print("\nSearching for 'magic and fantasy' content:")
    results = store.retrieve("magic and fantasy", n_results=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Query for sci-fi content
    print("\nSearching for 'science fiction and computers' content:")
    results = store.retrieve("science fiction and computers", n_results=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Query for action content
    print("\nSearching for 'action and fighting' content:")
    results = store.retrieve("action and fighting", n_results=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Test adding more documents
    print("\nAdding more documents...")
    more_documents = [
        "A team of explorers travel through a wormhole in search of a new habitable planet.",
        "A archaeologist searches for ancient artifacts while avoiding traps and rivals.",
    ]
    store.extend(more_documents)

    # Search across all documents
    print("\nSearching across all documents for 'space exploration and science':")
    results = store.retrieve("space exploration and science", n_results=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    # Clean up (optional)
    print("\nCleaning up...")
    store.reset()


if __name__ == "__main__":
    main()
