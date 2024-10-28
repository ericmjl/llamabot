"""Tests for QueryBot."""

from llamabot.bot.querybot import QueryBot
from hypothesis import HealthCheck, strategies as st, given, settings
import re
from pathlib import Path
import tempfile


@given(
    system_prompt=st.text().filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    collection_name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=63
    ).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    dummy_text=st.text(min_size=400).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    mock_response=st.text(min_size=4).filter(lambda x: not re.search(r"(.)\1{3,}", x)),
    stream_target=st.one_of(st.just("panel"), st.just("stdout")),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_querybot(
    tmp_path, system_prompt, collection_name, dummy_text, mock_response, stream_target
):
    """Test initialization of QueryBot."""
    tempfile = tmp_path / "test.txt"
    tempfile.write_text(dummy_text)

    bot = QueryBot(
        system_prompt=system_prompt,
        collection_name=collection_name,
        document_paths=tempfile,
        mock_response=mock_response,
        stream_target=stream_target,
    )

    bot("How are you doing?")
    bot.docstore.reset()


def test_querybot_with_sqlitevec():
    """Test QueryBot with SQLiteVecDocStore."""
    from llamabot.bot.querybot import QueryBot

    # Create a temporary file with some test content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            """
        A young wizard discovers he has magical powers and attends a school of witchcraft and wizardry.
        Two hobbits journey to destroy a powerful ring in a volcano while being pursued by dark forces.
        A group of superheroes must work together to save Earth from an alien invasion.
        """
        )
        temp_path = Path(f.name)

    try:
        # Initialize QueryBot with SQLiteVecDocStore
        bot = QueryBot(
            system_prompt="You are a helpful assistant that answers questions about movies.",
            collection_name="test_movies",
            document_paths=temp_path,
            docstore_type="sqlitevec",  # Specify SQLiteVecDocStore
            mock_response="This is a test response",  # For testing purposes
            stream_target="stdout",
        )

        # Test querying
        response = bot("Tell me about fantasy stories")
        assert response is not None

        # Test with a different query
        response = bot("What superhero content do you have?")
        assert response is not None

        # Clean up
        bot.docstore.reset()

    finally:
        # Clean up temporary file
        temp_path.unlink()
