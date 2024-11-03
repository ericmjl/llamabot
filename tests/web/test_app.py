"""Tests for the FastAPI app endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import json
import tempfile
from datetime import datetime

from llamabot.web.app import create_app
from llamabot.recorder import Base, MessageLog, Prompt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def test_db():
    """Create a test database."""
    # Create a temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix=".db")
    engine = create_engine(f"sqlite:///{temp_db.name}")

    # Create tables
    Base.metadata.create_all(engine)

    # Create a session factory
    SessionLocal = sessionmaker(bind=engine)

    return engine, SessionLocal, temp_db


@pytest.fixture
def test_client(test_db):
    """Create a test client with a temporary database."""
    engine, SessionLocal, temp_db = test_db
    app = create_app(Path(temp_db.name))
    client = TestClient(app)

    # Add some test data
    session = SessionLocal()

    # Add a test prompt
    test_prompt = Prompt(
        function_name="test_function", template="test template", hash="test_hash"
    )
    session.add(test_prompt)

    # Add a test message log
    test_log = MessageLog(
        object_name="test_object",
        timestamp=datetime.now(),
        model_name="test-model",
        temperature=0.7,
        message_log=json.dumps(
            [
                {
                    "role": "system",
                    "content": "You are a test bot",
                    "prompt_hash": "test_hash",
                },
                {"role": "user", "content": "Hello test bot"},
                {"role": "assistant", "content": "Hello test user"},
            ]
        ),
        rating=1,  # Positive rating
    )
    session.add(test_log)

    # Add another log without rating
    unrated_log = MessageLog(
        object_name="unrated_object",
        timestamp=datetime.now(),
        model_name="test-model",
        temperature=0.7,
        message_log=json.dumps(
            [
                {"role": "user", "content": "Unrated message"},
                {"role": "assistant", "content": "Unrated response"},
            ]
        ),
        rating=None,
    )
    session.add(unrated_log)

    session.commit()
    session.close()

    yield client

    # Cleanup
    temp_db.close()


def test_root_endpoint(test_client):
    """Test the root endpoint returns HTML."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_logs(test_client):
    """Test getting all logs."""
    response = test_client.get("/logs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_filtered_logs(test_client):
    """Test filtering logs."""
    # Test with text filter
    response = test_client.get("/filtered_logs", params={"text_filter": "test"})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Test with function name filter
    response = test_client.get(
        "/filtered_logs", params={"function_name": "test_function"}
    )
    assert response.status_code == 200


def test_export_openai_format(test_client):
    """Test exporting logs in OpenAI format."""
    response = test_client.get("/export/openai")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    # Parse the content as JSONL
    content = response.content.decode()
    conversations = [json.loads(line) for line in content.strip().split("\n")]

    # Verify the structure of exported data
    for conv in conversations:
        assert "messages" in conv
        for msg in conv["messages"]:
            assert "role" in msg
            assert "content" in msg


def test_export_positive_only(test_client):
    """Test exporting only positively rated logs."""
    response = test_client.get("/export/openai", params={"positive_only": "true"})
    assert response.status_code == 200

    # Parse the content and verify only positive ratings are included
    content = response.content.decode()
    conversations = [json.loads(line) for line in content.strip().split("\n")]
    assert len(conversations) == 1  # Only one log has a positive rating


def test_rate_log(test_client):
    """Test rating a log."""
    response = test_client.post("/log/1/rate", data={"rating": 1})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_prompt_functions(test_client):
    """Test getting prompt functions."""
    response = test_client.get("/prompt_functions")
    assert response.status_code == 200
    data = response.json()
    assert "function_names" in data


def test_get_prompts(test_client):
    """Test getting all prompts."""
    response = test_client.get("/prompts")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_log_details(test_client):
    """Test getting log details."""
    response = test_client.get("/log/1", params={"expanded": True})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_expand_collapse_log(test_client):
    """Test expanding and collapsing log details."""
    # Test expand
    response = test_client.get("/log/1/expand")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Test collapse
    response = test_client.get("/log/1/collapse")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
