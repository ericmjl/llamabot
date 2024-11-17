"""Tests for the FastAPI app endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import json
import tempfile
from datetime import datetime

from llamabot.web.app import create_app
from llamabot.recorder import Base, MessageLog, Prompt, Runs
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


@pytest.fixture
def test_experiment_data(test_db):
    """Add test experiment data to the database."""
    engine, SessionLocal, _ = test_db
    session = SessionLocal()

    # Create all tables
    Base.metadata.create_all(engine)

    # Add test experiment runs
    test_runs = [
        Runs(
            experiment_name="test_experiment",
            timestamp="2024-03-17T10:00:00",
            run_metadata=json.dumps({}),  # Serialize JSON data
            run_data=json.dumps(
                {  # Serialize JSON data
                    "metrics": {
                        "accuracy": {"value": 0.95, "timestamp": "2024-03-17T10:00:00"},
                        "loss": {"value": 0.1, "timestamp": "2024-03-17T10:00:00"},
                    },
                    "message_log_ids": [1, 2],
                    "prompts": [{"hash": "test_hash"}],
                }
            ),
        ),
        Runs(
            experiment_name="test_experiment",
            timestamp="2024-03-17T11:00:00",
            run_metadata=json.dumps({}),  # Serialize JSON data
            run_data=json.dumps(
                {  # Serialize JSON data
                    "metrics": {
                        "accuracy": {"value": 0.97, "timestamp": "2024-03-17T11:00:00"},
                        "loss": {"value": 0.05, "timestamp": "2024-03-17T11:00:00"},
                    },
                    "message_log_ids": [3],
                    "prompts": [{"hash": "test_hash"}],
                }
            ),
        ),
        Runs(
            experiment_name="another_experiment",
            timestamp="2024-03-17T12:00:00",
            run_metadata=json.dumps({}),  # Serialize JSON data
            run_data=json.dumps(
                {  # Serialize JSON data
                    "metrics": {
                        "f1_score": {"value": 0.88, "timestamp": "2024-03-17T12:00:00"}
                    },
                    "message_log_ids": [4],
                    "prompts": [{"hash": "test_hash"}],
                }
            ),
        ),
    ]

    for run in test_runs:
        session.add(run)

    session.commit()
    session.close()


def test_root_endpoint(test_client):
    """Test the root endpoint returns HTML."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_logs(test_client):
    """Test getting all logs."""
    response = test_client.get("/logs/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_filtered_logs(test_client):
    """Test filtering logs."""
    # Test with text filter
    response = test_client.get("/logs/filtered_logs", params={"text_filter": "test"})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Test with function name filter
    response = test_client.get(
        "/logs/filtered_logs", params={"function_name": "test_function"}
    )
    assert response.status_code == 200


def test_export_openai_format(test_client):
    """Test exporting logs in OpenAI format."""
    response = test_client.get("/logs/export/openai")
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
    response = test_client.get("/logs/export/openai", params={"positive_only": "true"})
    assert response.status_code == 200

    # Parse the content and verify only positive ratings are included
    content = response.content.decode()
    conversations = [json.loads(line) for line in content.strip().split("\n")]
    assert len(conversations) == 1  # Only one log has a positive rating


def test_rate_log(test_client):
    """Test rating a log."""
    response = test_client.post("/logs/1/rate", data={"rating": 1})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_prompt_functions(test_client):
    """Test getting prompt functions."""
    response = test_client.get("/prompts/functions")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Check that the response contains the expected text
    assert "test_function (1 versions)" in response.text


def test_get_prompt_history(test_client):
    """Test getting prompt history."""
    response = test_client.get(
        "/prompts/history", params={"function_name": "test_function"}
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_log_details(test_client):
    """Test getting log details."""
    response = test_client.get("/logs/1", params={"expanded": True})
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_expand_collapse_log(test_client):
    """Test expanding and collapsing log details."""
    # Test expand
    response = test_client.get("/logs/1/expand")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Test collapse
    response = test_client.get("/logs/1/collapse")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_prompt_details(test_client):
    """Test getting prompt details."""
    response = test_client.get("/prompts/test_hash")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_nonexistent_log(test_client):
    """Test accessing a nonexistent log."""
    response = test_client.get("/logs/999")
    assert response.status_code == 404


def test_nonexistent_prompt(test_client):
    """Test accessing a nonexistent prompt."""
    response = test_client.get("/prompts/nonexistent_hash")
    assert response.status_code == 404


def test_list_experiments(test_client, test_experiment_data):
    """Test listing all experiments."""
    response = test_client.get("/experiments/list")
    assert response.status_code == 200
    data = response.json()

    # Check that we get the expected experiments
    assert len(data) == 2  # We added two different experiments

    # Check the structure and content
    experiments = {exp["name"]: exp["count"] for exp in data}
    assert experiments["test_experiment"] == 2
    assert experiments["another_experiment"] == 1


def test_get_experiment_details(test_client, test_experiment_data):
    """Test getting details for a specific experiment."""
    response = test_client.get(
        "/experiments/details", params={"experiment_name": "test_experiment"}
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Check that the response contains expected content
    content = response.text
    assert "test_experiment" in content
    assert "accuracy" in content
    assert "loss" in content
    assert "0.95" in content
    assert "0.97" in content


def test_get_experiment_details_nonexistent(test_client, test_experiment_data):
    """Test getting details for a nonexistent experiment."""
    response = test_client.get(
        "/experiments/details", params={"experiment_name": "nonexistent_experiment"}
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Should return empty table
    content = response.text
    assert "nonexistent_experiment" in content
    assert len(response.text) > 0  # Should still return the template


def test_experiment_details_metrics(test_client, test_experiment_data):
    """Test that experiment details correctly displays all metrics."""
    response = test_client.get(
        "/experiments/details", params={"experiment_name": "test_experiment"}
    )
    assert response.status_code == 200

    content = response.text
    # Check for all metrics
    assert "accuracy" in content
    assert "loss" in content
    assert "0.95" in content
    assert "0.1" in content
    assert "0.97" in content
    assert "0.05" in content


def test_experiment_details_message_logs(test_client, test_experiment_data):
    """Test that experiment details correctly displays message log IDs."""
    response = test_client.get(
        "/experiments/details", params={"experiment_name": "test_experiment"}
    )
    assert response.status_code == 200

    content = response.text
    # Check for message log IDs
    assert "1" in content
    assert "2" in content
    assert "3" in content


def test_experiment_details_prompts(test_client, test_experiment_data):
    """Test that experiment details correctly displays prompt information."""
    response = test_client.get(
        "/experiments/details", params={"experiment_name": "test_experiment"}
    )
    assert response.status_code == 200

    content = response.text
    # Check for prompt hash
    assert "test_hash" in content
    assert "test_function" in content  # From the test_prompt fixture


def test_experiment_details_timestamps(test_client, test_experiment_data):
    """Test that experiment details correctly displays timestamps."""
    response = test_client.get(
        "/experiments/details", params={"experiment_name": "test_experiment"}
    )
    assert response.status_code == 200

    content = response.text
    # Check for timestamps
    assert "2024-03-17T10:00:00" in content
    assert "2024-03-17T11:00:00" in content
