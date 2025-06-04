# Playwright Testing Strategy for Llamabot Log Viewer

This document outlines the testing strategy for the llamabot log viewer using Playwright, a modern end-to-end testing framework for web applications.

## Overview

The llamabot log viewer is a web application built with FastAPI and HTMX that allows users to:
- View and filter message logs
- Compare prompt versions
- Track experiments and their metrics
- Rate and export conversations

This testing strategy ensures that all these features work correctly and reliably across different browsers and scenarios.

## Test Setup

```python
# tests/web/test_playwright.py

import pytest
from playwright.sync_api import Page, expect
from llamabot.web.app import create_app
from fastapi.testclient import TestClient
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_app():
    """Create a test FastAPI app with a temporary database."""
    temp_db = tempfile.NamedTemporaryFile(suffix=".db")
    app = create_app(Path(temp_db.name))
    return app, temp_db

@pytest.fixture(scope="session")
def test_client(test_app):
    """Create a test client."""
    app, _ = test_app
    return TestClient(app)

@pytest.fixture(scope="session")
def test_server(test_app):
    """Start a test server."""
    app, _ = test_app
    import uvicorn
    import threading
    import time

    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(1)  # Wait for server to start
    yield
```

## Test Categories

### A. Navigation Tests
```python
def test_navigation(page: Page):
    """Test basic navigation through the app."""
    # Test navigation to main sections
    page.goto("http://localhost:8000/")
    expect(page).to_have_title("Llamabot Log Viewer")

    # Test navigation to logs
    page.click("text=Logs")
    expect(page).to_have_url("http://localhost:8000/logs/")

    # Test navigation to prompts
    page.click("text=Prompts")
    expect(page).to_have_url("http://localhost:8000/prompts/functions")
```

### B. Log Viewer Tests
```python
def test_log_viewer_layout(page: Page):
    """Test the basic layout of the log viewer page."""
    page.goto("http://localhost:8000/logs/")

    # Verify the two-panel layout
    expect(page.locator(".logs-table")).to_be_visible()
    expect(page.locator(".log-details-panel")).to_be_visible()
```

```python
def test_log_entry_selection(page: Page):
    """Test selecting a log entry and viewing its details."""
    page.goto("http://localhost:8000/logs/")

    # Click on the first log entry
    page.click(".log-entry >> nth=0")

    # Verify the details panel is populated
    expect(page.locator(".log-details-panel")).to_contain_text("Log Details")
    expect(page.locator(".log-details-panel")).to_contain_text("Messages")
```

```python
def test_message_expansion(page: Page):
    """Test expanding and collapsing messages in a log entry."""
    page.goto("http://localhost:8000/logs/")

    # Select a log entry
    page.click(".log-entry >> nth=0")

    # Click on a message to expand it
    page.click(".message-entry >> nth=0")

    # Verify the message is expanded
    expect(page.locator(".message-content")).to_be_visible()

    # Click again to collapse
    page.click(".message-entry >> nth=0")
    expect(page.locator(".message-content")).not_to_be_visible()
```

```python
def test_prompt_filtering(page: Page):
    """Test filtering logs by prompt selection."""
    page.goto("http://localhost:8000/logs/")

    # Get initial count of logs
    initial_count = page.locator(".log-entry").count()

    # Select a prompt from the dropdown
    page.select_option("select[name='prompt_filter']", "test_prompt")

    # Verify the logs are filtered
    filtered_count = page.locator(".log-entry").count()
    assert filtered_count < initial_count

    # Verify all visible logs contain the selected prompt
    for log in page.locator(".log-entry").all():
        expect(log).to_contain_text("test_prompt")
```

### C. Prompt Management Tests
```python
def test_prompt_management(page: Page):
    """Test prompt management functionality."""
    page.goto("http://localhost:8000/prompts/functions")

    # Test prompt version history
    page.click("text=test_function")
    expect(page).to_have_url("http://localhost:8000/prompts/history?function_name=test_function")

    # Test prompt comparison
    page.click("input[type='checkbox'] >> nth=0")
    page.click("input[type='checkbox'] >> nth=1")
    page.click("button:has-text('Compare')")
    expect(page.locator(".prompt-comparison")).to_be_visible()
```

### D. Experiment Tracking Tests
```python
def test_experiment_tracking(page: Page):
    """Test experiment tracking functionality."""
    page.goto("http://localhost:8000/experiments")

    # Test experiment list
    expect(page.locator(".experiment-entry")).to_have_count(2)

    # Test experiment details
    page.click(".experiment-entry >> nth=0")
    expect(page.locator(".experiment-metrics")).to_be_visible()

    # Test metric visualization
    expect(page.locator(".metric-chart")).to_be_visible()
```

## Test Data Setup

```python
@pytest.fixture(scope="session")
def test_data(test_app):
    """Set up test data in the database."""
    app, temp_db = test_app
    # Add test data using the existing test_client fixture
    # This will be similar to the existing test data setup
    # but adapted for Playwright testing
```

## Running Tests

To run the tests:
```bash
pytest tests/web/test_playwright.py -v
```

For debugging:
```bash
PWDEBUG=1 pytest tests/web/test_playwright.py -v
```

## Key Testing Areas

1. **UI Component Testing**
   - Navigation and routing
   - Component rendering
   - Interactive elements (buttons, forms, etc.)
   - Responsive design

2. **Functionality Testing**
   - Log filtering and search
   - Log expansion/collapse
   - Rating system
   - Prompt version comparison
   - Experiment tracking

3. **Data Visualization Testing**
   - Metric charts
   - Log displays
   - Prompt comparisons

4. **Error Handling Testing**
   - Invalid inputs
   - Network errors
   - Missing data scenarios

## Best Practices

1. **Test Isolation**
   - Each test should be independent
   - Use fixtures for setup and teardown
   - Clean up test data after each test

2. **Selectors**
   - Use data-testid attributes for reliable selection
   - Prefer text content over CSS selectors when possible
   - Use role-based selectors for accessibility

3. **Assertions**
   - Test both positive and negative cases
   - Verify UI state changes
   - Check error messages and handling

4. **Performance**
   - Run tests in parallel when possible
   - Use headless mode for CI
   - Implement proper waiting strategies

## Maintenance

- Regular updates to test selectors as UI evolves
- Periodic review of test coverage
- Performance optimization of test suite
- Documentation updates as features change
