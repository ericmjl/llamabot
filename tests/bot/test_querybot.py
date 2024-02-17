"""Tests for QueryBot."""
from llamabot.bot.querybot import QueryBot
from hypothesis import HealthCheck, strategies as st, given, settings


@given(
    system_prompt=st.text(),
    collection_name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=63
    ),
    dummy_text=st.text(),
    mock_response=st.text(),
    human_message=st.text(),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_querybot_init(
    tmp_path, system_prompt, collection_name, dummy_text, mock_response, human_message
):
    """Test initialization of QueryBot."""
    tempfile = tmp_path / "test.txt"
    tempfile.write_text(dummy_text)

    QueryBot(
        system_prompt=system_prompt,
        collection_name=collection_name,
        document_paths=tempfile,
        mock_response=mock_response,
    )
