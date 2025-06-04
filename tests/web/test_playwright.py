"""Playwright tests for the llamabot web interface."""

import pytest
from playwright.sync_api import Page, expect
from llamabot.web.app import create_app
from fastapi.testclient import TestClient
import tempfile
from pathlib import Path
import json
from datetime import datetime
from llamabot.recorder import Base, MessageLog, Prompt
import uvicorn
import threading
import time
import requests


@pytest.fixture(scope="session")
def test_app():
    """Create a test FastAPI app with a temporary database.

    :returns: Tuple of (app, temp_db)
    """
    temp_db = tempfile.NamedTemporaryFile(suffix=".db")
    app = create_app(Path(temp_db.name))
    return app, temp_db


@pytest.fixture(scope="session")
def test_client(test_app):
    """Create a test client.

    :param test_app: The test app fixture
    :returns: TestClient instance
    """
    app, _ = test_app
    return TestClient(app)


@pytest.fixture(scope="session")
def test_server(test_app):
    """Start a test server.

    :param test_app: The test app fixture
    :yields: None
    """
    app, _ = test_app

    def run_server():
        """Run the uvicorn server in a separate thread."""
        uvicorn.run(app, host="127.0.0.1", port=8000)

    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to be ready
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            if i == max_retries - 1:
                raise Exception("Server failed to start")
            time.sleep(1)

    yield

    # Cleanup
    server_thread.join(timeout=1)


@pytest.fixture(scope="session")
def test_data(test_app):
    """Set up test data in the database.

    :param test_app: The test app fixture
    :returns: None
    """
    app, temp_db = test_app
    engine = app.state.engine

    # Create tables
    Base.metadata.create_all(engine)

    # Create a session
    SessionLocal = app.state.SessionLocal
    session = SessionLocal()

    try:
        # Add test prompts
        test_prompt = Prompt(
            function_name="test_function", template="test template", hash="test_hash"
        )
        session.add(test_prompt)

        # Add test message logs
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
            rating=1,
        )
        session.add(test_log)

        # Add another log with different prompt
        another_log = MessageLog(
            object_name="another_object",
            timestamp=datetime.now(),
            model_name="test-model",
            temperature=0.7,
            message_log=json.dumps(
                [
                    {
                        "role": "system",
                        "content": "You are another test bot",
                        "prompt_hash": "another_hash",
                    },
                    {"role": "user", "content": "Hello another bot"},
                    {"role": "assistant", "content": "Hello another user"},
                ]
            ),
            rating=None,
        )
        session.add(another_log)

        session.commit()
    finally:
        session.close()


def test_log_viewer_layout(page: Page):
    """Test the basic layout of the log viewer page.

    :param page: Playwright page fixture
    :returns: None
    """
    page.goto("http://localhost:8000/logs/")

    # Verify the two-panel layout
    expect(page.locator(".card-body table")).to_be_visible()
    expect(page.locator("#log-details")).to_be_visible()


def test_log_entry_selection(page: Page):
    """Test selecting a log entry and viewing its details.

    :param page: Playwright page fixture
    :returns: None
    """
    page.goto("http://localhost:8000/logs/")

    # Wait for the table to be populated
    page.wait_for_selector("#logs-tbody tr")

    # Click on the first log entry
    page.click("#logs-tbody tr >> nth=0")

    # Verify the details panel is populated
    expect(page.locator("#log-details")).not_to_contain_text(
        "Select a log to view details"
    )


def test_message_expansion(page: Page):
    """Test expanding and collapsing messages in a log entry.

    :param page: Playwright page fixture
    :returns: None
    """
    page.goto("http://localhost:8000/logs/")

    # Wait for the table to be populated
    page.wait_for_selector("#logs-tbody tr")

    # Select a log entry
    page.click("#logs-tbody tr >> nth=0")

    # Wait for details to load
    page.wait_for_selector("#log-details .message")

    # Click on a message to expand it
    page.click("#log-details .message-header >> nth=0")

    # Verify the message is expanded
    expect(page.locator("#log-details .collapse.show")).to_be_visible()
    # Check the first message's content specifically
    expect(page.locator("#message-1 .content")).to_be_visible()

    # Click again to collapse
    page.click("#log-details .message-header >> nth=0")
    expect(page.locator("#log-details .collapse.show")).not_to_be_visible()
