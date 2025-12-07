"""
This module provides a set of tests for the PromptRecorder
and autorecord functions in the llamabot.recorder module.
"""

from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from llamabot.components.messages import BaseMessage
from llamabot.recorder import (
    Base,
    MessageLog,
    Span,
    SpanRecord,
    add_column,
    build_hierarchy,
    calculate_nesting_level,
    enable_span_recording,
    escape_html,
    format_duration,
    format_timestamp,
    get_current_span,
    get_span_color,
    get_span_tree,
    get_spans,
    is_span_recording_enabled,
    span,
    span_to_dict,
    sqlite_log,
    upgrade_database,
)


@pytest.fixture
def engine():
    """Create a temporary in-memory SQLite database for testing."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(engine):
    """Create a new session for testing."""
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_message_log_creation(session):
    """Test that a MessageLog instance can be created and saved to the database."""
    log = MessageLog(
        object_name="test_object",
        timestamp="2023-01-01T00:00:00",
        message_log='{"message": "test"}',
    )
    session.add(log)
    session.commit()

    retrieved_log = session.query(MessageLog).first()
    assert retrieved_log.object_name == "test_object"
    assert retrieved_log.timestamp == "2023-01-01T00:00:00"
    assert retrieved_log.message_log == '{"message": "test"}'


def test_upgrade_database(engine):
    """Test that upgrade_database creates the MessageLog table if it doesn't exist."""
    # Drop the table if it exists
    MessageLog.__table__.drop(engine, checkfirst=True)

    # Run upgrade_database
    upgrade_database(engine)

    # Check if the table now exists
    inspector = inspect(engine)
    assert inspector.has_table("message_log")

    # Check if all columns exist
    columns = inspector.get_columns("message_log")
    column_names = [c["name"] for c in columns]
    expected_columns = ["id", "object_name", "timestamp", "message_log"]
    for col in expected_columns:
        assert col in column_names


def test_add_column(engine):
    """Test that add_column adds a new column to an existing table."""
    from sqlalchemy import Column, String

    # Ensure the table exists
    Base.metadata.create_all(engine)

    # Add a new column
    new_column = Column("new_test_column", String)
    with engine.connect() as connection:
        add_column(connection, MessageLog.__tablename__, new_column)

    # Check if the new column exists
    inspector = inspect(engine)
    columns = inspector.get_columns(MessageLog.__tablename__)
    assert any(col["name"] == "new_test_column" for col in columns)


def test_sqlite_log(engine, monkeypatch, tmp_path):
    """Test that sqlite_log correctly logs messages to the database in a temporary directory."""
    # Create a temporary directory for the test database
    temp_db_path = tmp_path / "test_message_log.db"

    # Mock the get_object_name function to return a predictable name
    def mock_get_object_name(obj):
        return "test_object"

    monkeypatch.setattr("llamabot.recorder.get_object_name", mock_get_object_name)

    # Create a mock test object with model_name and temperature attributes
    test_obj = Mock()
    test_obj.model_name = "test_model"
    test_obj.temperature = 0.7

    test_messages = [
        BaseMessage(role="user", content="Hello"),
        BaseMessage(role="assistant", content="Hi there!"),
    ]

    # Call sqlite_log with the temporary database path
    sqlite_log(test_obj, test_messages, db_path=temp_db_path)

    # Create an engine for the temporary database
    temp_engine = create_engine(f"sqlite:///{temp_db_path}")
    Session = sessionmaker(bind=temp_engine)
    session = Session()

    # Query the database to check if the log was saved
    log_entry = session.query(MessageLog).first()

    assert log_entry is not None
    assert log_entry.object_name == "test_object"
    assert "Hello" in log_entry.message_log
    assert "Hi there!" in log_entry.message_log
    assert log_entry.model_name == "test_model"
    assert log_entry.temperature == 0.7

    # Close the session and dispose of the engine
    session.close()
    temp_engine.dispose()

    # Verify that the database file was created
    assert temp_db_path.exists()

    # The temporary directory and its contents will be automatically cleaned up after the test


def test_span_creation(tmp_path):
    """Test that a span can be created and used as context manager."""
    db_path = tmp_path / "test_spans.db"

    with span("test_operation", test_attr="test_value", db_path=db_path) as s:
        assert s.operation_name == "test_operation"
        assert s["test_attr"] == "test_value"
        assert s.status == "started"
        s.log("test_event", data="test_data")

    # Check span was saved to database
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path}")
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    span_record = session.query(SpanRecord).first()
    assert span_record is not None
    assert span_record.operation_name == "test_operation"
    assert span_record.status == "completed"
    assert span_record.duration_ms is not None
    assert span_record.duration_ms > 0

    import json

    attributes = json.loads(span_record.attributes)
    assert attributes["test_attr"] == "test_value"

    events = json.loads(span_record.events)
    assert len(events) == 1
    assert events[0]["name"] == "test_event"

    session.close()


def test_span_dictionary_interface(tmp_path):
    """Test that spans support dictionary-like interface."""
    db_path = tmp_path / "test_spans.db"

    with span("dict_test", db_path=db_path) as s:
        s["key1"] = "value1"
        s["key2"] = 42
        assert s["key1"] == "value1"
        assert s["key2"] == 42
        assert "key1" in s
        assert "key2" in s
        assert s.get("key1") == "value1"
        assert s.get("nonexistent", "default") == "default"
        del s["key1"]
        assert "key1" not in s


def test_nested_spans(tmp_path):
    """Test that spans can be nested."""
    db_path = tmp_path / "test_spans.db"

    with span("outer", db_path=db_path) as outer:
        with outer.span("inner") as inner:
            inner["nested"] = True
            inner.log("inner_event")

        outer.log("outer_event")

    # Check both spans were saved
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path}")
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    spans = session.query(SpanRecord).all()
    assert len(spans) == 2

    outer_span = next(s for s in spans if s.operation_name == "outer")
    inner_span = next(s for s in spans if s.operation_name == "inner")

    assert inner_span.parent_span_id == outer_span.span_id
    assert outer_span.parent_span_id is None
    assert outer_span.trace_id == inner_span.trace_id

    session.close()


def test_span_decorator(tmp_path):
    """Test that span can be used as decorator."""
    db_path = tmp_path / "test_spans.db"

    @span("decorated_function", db_path=db_path)
    def test_function(x: int, y: int) -> int:
        """Test function for span decorator.

        :param x: First number
        :param y: Second number
        :return: Sum of x and y
        """
        current_span = get_current_span()
        if current_span:
            current_span.log("function_called", x=x, y=y)
        return x + y

    result = test_function(2, 3)
    assert result == 5

    # Check span was created
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path}")
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    span_record = session.query(SpanRecord).first()
    assert span_record is not None
    assert span_record.operation_name == "decorated_function"

    import json

    events = json.loads(span_record.events)
    assert len(events) == 1
    assert events[0]["name"] == "function_called"

    session.close()


def test_span_decorator_logs_inputs_and_outputs(tmp_path):
    """Test that span decorator automatically logs function inputs and outputs."""
    db_path = tmp_path / "test_spans.db"

    @span("explodify", db_path=db_path)
    def explodify(s: str, i: int) -> str:
        """Add exclamation marks around a string.

        :param s: String to explodify
        :param i: Number of exclamation marks to add
        :return: String with exclamation marks
        """
        return f"{'!' * i} {s} {'!' * i}"

    result = explodify("Boom!", 3)
    assert result == "!!! Boom! !!!"

    # Check span was created and saved
    spans = get_spans(operation_name="explodify", db_path=db_path)
    assert len(spans) == 1

    span_obj = spans[0]

    # Verify inputs are logged as attributes (coerced to strings)
    assert span_obj.attributes.get("input_s") == "Boom!"
    assert span_obj.attributes.get("input_i") == "3"  # Coerced to string

    # Verify output is logged as attribute (coerced to string)
    assert span_obj.attributes.get("output") == "!!! Boom! !!!"

    # Verify no duplicate events for inputs/outputs (they should only be in attributes)
    event_names = [event["name"] for event in span_obj.events]
    assert "function_inputs" not in event_names
    assert "function_output" not in event_names


def test_span_decorator_logs_args_and_kwargs(tmp_path):
    """Test that span decorator logs *args and **kwargs correctly."""
    db_path = tmp_path / "test_spans.db"

    @span("test_func", db_path=db_path)
    def test_func(*args, **kwargs):
        """Function with *args and **kwargs."""
        return sum(args) + sum(kwargs.values())

    result = test_func(5, 3, multiplier=2)
    assert result == 10

    # Check span was created
    spans = get_spans(operation_name="test_func", db_path=db_path)
    assert len(spans) == 1

    span_obj = spans[0]

    # Verify *args and **kwargs are logged as input_args and input_kwargs
    assert span_obj.attributes.get("input_args") == "(5, 3)"
    assert span_obj.attributes.get("input_kwargs") == "{'multiplier': 2}"

    # Verify output is logged
    assert span_obj.attributes.get("output") == "10"


def test_span_decorator_handles_unstringifiable_objects(tmp_path):
    """Test that span decorator handles objects that can't be stringified."""
    db_path = tmp_path / "test_spans.db"

    class Unstringifiable:
        """Object that raises exception when stringified."""

        def __str__(self):
            """Raise ValueError when stringified."""
            raise ValueError("Cannot stringify")

    @span("test_unstringifiable", db_path=db_path)
    def test_func(obj: Unstringifiable) -> Unstringifiable:
        """Test function that returns the input object."""
        return obj

    obj = Unstringifiable()
    result = test_func(obj)
    assert result is obj

    # Check span was created
    spans = get_spans(operation_name="test_unstringifiable", db_path=db_path)
    assert len(spans) == 1

    span_obj = spans[0]

    # Verify fallback message is used when stringification fails
    assert (
        span_obj.attributes.get("input_obj") == "<unable to stringify Unstringifiable>"
    )
    assert span_obj.attributes.get("output") == "<unable to stringify Unstringifiable>"


def test_get_current_span():
    """Test get_current_span function."""
    assert get_current_span() is None

    with span("test") as s:
        assert get_current_span() is s

    assert get_current_span() is None


def test_get_spans(tmp_path):
    """Test get_spans query function."""
    db_path = tmp_path / "test_spans.db"

    # Create multiple spans
    with span("operation1", category="test", db_path=db_path) as s1:
        s1["value"] = 1

    with span("operation2", category="test", db_path=db_path) as s2:
        s2["value"] = 2

    trace_id = s1.trace_id

    # Query spans
    spans = get_spans(trace_id=trace_id, db_path=db_path)
    assert len(spans) == 2

    # Query by operation name (should return only matching span, no children)
    spans = get_spans(operation_name="operation1", db_path=db_path)
    assert len(spans) == 1
    assert spans[0].operation_name == "operation1"

    # Query by attribute
    spans = get_spans(category="test", db_path=db_path)
    assert len(spans) == 2


def test_get_spans_operation_name_returns_descendants(tmp_path):
    """Test that get_spans(operation_name=...) returns matching spans and their descendants."""
    db_path = tmp_path / "test_spans.db"

    # Create a trace with nested spans
    with span("parent_operation", db_path=db_path) as parent:
        with parent.span("child1"):
            pass
        with parent.span("child2") as child2:
            with child2.span("grandchild"):
                pass
        # Create an unrelated sibling span (same trace, different parent)
        with span("unrelated_sibling", db_path=db_path) as sibling:
            sibling["note"] = "unrelated"

    # Query by parent operation name - should return parent and all descendants
    spans = get_spans(operation_name="parent_operation", db_path=db_path)
    span_names = [s.operation_name for s in spans]

    # Should include parent and all descendants
    assert "parent_operation" in span_names
    assert "child1" in span_names
    assert "child2" in span_names
    assert "grandchild" in span_names

    # Should NOT include unrelated sibling
    assert "unrelated_sibling" not in span_names
    assert len(spans) == 4

    # Query by child operation name - should return child and its descendants only
    spans = get_spans(operation_name="child2", db_path=db_path)
    span_names = [s.operation_name for s in spans]

    assert "child2" in span_names
    assert "grandchild" in span_names
    assert "parent_operation" not in span_names
    assert "child1" not in span_names
    assert "unrelated_sibling" not in span_names
    assert len(spans) == 2


def test_span_decorator_creates_child_span(tmp_path):
    """Test that @span decorator creates child spans when called within a span context."""
    db_path = tmp_path / "test_spans.db"

    @span("inner_function", db_path=db_path)
    def inner_function(x: int) -> int:
        """Inner function that should be a child span."""
        return x * 2

    # Call decorated function within a span context
    with span("outer_operation", db_path=db_path) as outer_span:
        result = inner_function(5)
        assert result == 10

    # Verify spans were created
    spans = get_spans(db_path=db_path)
    span_dict = {s.operation_name: s for s in spans}

    # Check that inner_function span exists and is a child of outer_operation
    assert "outer_operation" in span_dict
    assert "inner_function" in span_dict

    outer_span_from_db = span_dict["outer_operation"]
    inner_span = span_dict["inner_function"]

    # Verify parent-child relationship
    assert inner_span.parent_span_id == outer_span_from_db.span_id
    assert inner_span.trace_id == outer_span_from_db.trace_id
    # Also verify it matches the context manager span
    assert inner_span.parent_span_id == outer_span.span_id

    # Verify hierarchy when querying by operation name
    spans = get_spans(operation_name="outer_operation", db_path=db_path)
    span_names = [s.operation_name for s in spans]
    assert "outer_operation" in span_names
    assert "inner_function" in span_names
    assert len(spans) == 2


def test_get_span_tree(tmp_path):
    """Test get_span_tree function."""
    db_path = tmp_path / "test_spans.db"

    with span("root", db_path=db_path) as root:
        with root.span("child1"):
            pass
        with root.span("child2") as child2:
            with child2.span("grandchild"):
                pass

    trace_id = root.trace_id

    tree = get_span_tree(trace_id, db_path=db_path)
    assert tree["operation_name"] == "root"
    assert len(tree["children"]) == 2
    assert tree["children"][0]["operation_name"] == "child1"
    assert tree["children"][1]["operation_name"] == "child2"
    assert len(tree["children"][1]["children"]) == 1
    assert tree["children"][1]["children"][0]["operation_name"] == "grandchild"


def test_span_error_handling(tmp_path):
    """Test that spans capture errors correctly."""
    db_path = tmp_path / "test_spans.db"

    try:
        with span("error_test", db_path=db_path):
            raise ValueError("Test error")
    except ValueError:
        pass

    # Check span was saved with error status
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path}")
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    span_record = session.query(SpanRecord).first()
    assert span_record.status == "error"
    assert "Test error" in span_record.error_message

    session.close()


def test_enable_span_recording():
    """Test enable_span_recording function.

    Note: Bots now always create spans by default, but enable_span_recording()
    can still be used for other parts of the codebase that check this flag.
    """
    # Reset state
    import llamabot.recorder as recorder_module

    recorder_module._span_recording_enabled = False

    assert not is_span_recording_enabled()
    enable_span_recording()
    assert is_span_recording_enabled()


def test_sqlite_log_links_to_span(tmp_path, monkeypatch):
    """Test that sqlite_log links messages to current span."""
    db_path = tmp_path / "test_message_log.db"

    def mock_get_object_name(obj):
        return "test_object"

    monkeypatch.setattr("llamabot.recorder.get_object_name", mock_get_object_name)

    test_obj = Mock()
    test_obj.model_name = "test_model"
    test_obj.temperature = 0.7

    test_messages = [
        BaseMessage(role="user", content="Hello"),
        BaseMessage(role="assistant", content="Hi there!"),
    ]

    # Create a span and log within it
    with span("test_operation", db_path=db_path):
        log_id = sqlite_log(test_obj, test_messages, db_path=db_path)

    # Check message log is linked to span
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{db_path}")
    upgrade_database(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    log_entry = session.query(MessageLog).filter_by(id=log_id).first()
    assert log_entry.span_id is not None
    assert log_entry.trace_id is not None

    # Verify span exists
    span_record = session.query(SpanRecord).filter_by(span_id=log_entry.span_id).first()
    assert span_record is not None
    assert span_record.operation_name == "test_operation"

    session.close()


def test_span_to_dict(tmp_path):
    """Test span_to_dict conversion function."""
    db_path = tmp_path / "test_spans.db"

    # Test with complete span
    with span("test_operation", category="test", db_path=db_path) as s:
        s["result"] = "success"
        s.log("test_event", data="test_data")

    span_dict = span_to_dict(s)
    assert span_dict["span_id"] == s.span_id
    assert span_dict["trace_id"] == s.trace_id
    assert span_dict["operation_name"] == "test_operation"
    assert span_dict["status"] == "completed"
    assert span_dict["duration_ms"] is not None
    assert span_dict["end_time"] is not None
    assert span_dict["attributes"]["category"] == "test"
    assert span_dict["attributes"]["result"] == "success"
    assert len(span_dict["events"]) == 1

    # Test with incomplete span
    incomplete_span = Span("incomplete", db_path=db_path)
    incomplete_dict = span_to_dict(incomplete_span)
    assert incomplete_dict["status"] == "started"
    assert incomplete_dict["end_time"] is None
    assert incomplete_dict["duration_ms"] is None

    # Test with error span
    try:
        with span("error_operation", db_path=db_path) as error_s:
            raise ValueError("Test error")
    except ValueError:
        pass

    error_dict = span_to_dict(error_s)
    assert error_dict["status"] == "error"
    assert error_dict["error_message"] == "Test error"


def test_escape_html():
    """Test HTML escaping function."""
    # Test special characters
    assert (
        escape_html("<script>alert('xss')</script>")
        == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
    )
    assert escape_html('Hello "world"') == "Hello &quot;world&quot;"
    assert escape_html("A & B") == "A &amp; B"

    # Test normal text
    assert escape_html("Hello world") == "Hello world"

    # Test None/empty strings
    assert escape_html(None) == ""
    assert escape_html("") == ""


def test_format_timestamp():
    """Test timestamp formatting."""
    # Test ISO format
    timestamp = "2024-01-01T12:34:56.123456"
    formatted = format_timestamp(timestamp)
    assert "12:34:56" in formatted

    # Test None handling
    assert format_timestamp(None) == ""
    assert format_timestamp("") == ""


def test_format_duration():
    """Test duration formatting."""
    # Test milliseconds
    assert format_duration(123) == "123ms"
    assert format_duration(999) == "999ms"

    # Test seconds
    assert format_duration(1000) == "1.00s"
    assert format_duration(1500) == "1.50s"
    assert format_duration(59999) == "60.00s"  # 59999ms = 59.999s rounds to 60.00s

    # Test minutes
    assert format_duration(60000) == "1.00min"
    assert format_duration(120000) == "2.00min"

    # Test in-progress spans
    assert format_duration(None, is_in_progress=True) == "in progress"
    assert format_duration(None, is_in_progress=False) == "in progress"


def test_get_span_color():
    """Test color mapping."""
    assert get_span_color("completed") == "#A3BE8C"  # aurora green
    assert get_span_color("error") == "#BF616A"  # aurora red
    assert get_span_color("started") == "#5E81AC"  # frost blue
    assert get_span_color("unknown") == "#5E81AC"  # default


def test_calculate_nesting_level(tmp_path):
    """Test nesting level calculation."""
    db_path = tmp_path / "test_spans.db"

    with span("root", db_path=db_path) as root:
        with root.span("child1") as child1:
            with child1.span("grandchild") as grandchild:
                pass

    # Get all spans (returns Span objects)
    spans = get_spans(trace_id=root.trace_id, db_path=db_path)
    # Convert to dictionaries for calculate_nesting_level
    spans_dicts = [span_to_dict(s) for s in spans]
    span_dict = {s["span_id"]: s for s in spans_dicts}

    # Check nesting levels
    root_level = calculate_nesting_level(
        next(s for s in spans_dicts if s["span_id"] == root.span_id), span_dict
    )
    child_level = calculate_nesting_level(
        next(s for s in spans_dicts if s["span_id"] == child1.span_id), span_dict
    )
    grandchild_level = calculate_nesting_level(
        next(s for s in spans_dicts if s["span_id"] == grandchild.span_id), span_dict
    )

    assert root_level == 0
    assert child_level == 1
    assert grandchild_level == 2


def test_build_hierarchy(tmp_path):
    """Test hierarchy building."""
    db_path = tmp_path / "test_spans.db"

    with span("root", db_path=db_path) as root:
        with root.span("child1"):
            pass
        with root.span("child2"):
            pass

    spans = get_spans(trace_id=root.trace_id, db_path=db_path)
    # Convert Span objects to dictionaries for build_hierarchy
    spans_dicts = [span_to_dict(s) for s in spans]
    tree = build_hierarchy(spans_dicts)

    assert tree["operation_name"] == "root"
    assert len(tree["children"]) == 2
    assert tree["children"][0]["operation_name"] in ["child1", "child2"]
    assert tree["children"][1]["operation_name"] in ["child1", "child2"]


def test_repr_html_complete_span(tmp_path):
    """Test _repr_html_() with complete span."""
    db_path = tmp_path / "test_spans.db"

    with span("test_operation", category="test", db_path=db_path) as s:
        s["result"] = "success"

    html = s._repr_html_()

    # Verify HTML contains span data
    assert "test_operation" in html
    assert "test" in html
    assert "success" in html
    assert "span-container" in html
    assert "span-timeline" in html
    assert "span-details" in html
    assert s.span_id in html


def test_repr_html_incomplete_span(tmp_path):
    """Test _repr_html_() with incomplete span."""
    db_path = tmp_path / "test_spans.db"

    incomplete_span = Span("incomplete", db_path=db_path)
    html = incomplete_span._repr_html_()

    # Verify "in progress" indicator
    assert "incomplete" in html
    assert "in progress" in html
    assert incomplete_span.span_id in html


def test_repr_html_with_trace(tmp_path):
    """Test _repr_html_() with multiple spans in trace."""
    db_path = tmp_path / "test_spans.db"

    with span("root", db_path=db_path) as root:
        with root.span("child1"):
            pass
        with root.span("child2"):
            pass

    html = root._repr_html_()

    # Verify all spans appear
    assert "root" in html
    assert "child1" in html
    assert "child2" in html
    assert root.span_id in html


def test_repr_html_pagination(tmp_path):
    """Test pagination in _repr_html_()."""
    db_path = tmp_path / "test_spans.db"

    # Create 30 spans
    with span("root", db_path=db_path) as root:
        for i in range(29):
            with root.span(f"child{i}"):
                pass

    html = root._repr_html_()

    # Verify pagination controls appear
    assert "pagination-controls" in html
    assert "Showing" in html
    assert "spans" in html


def test_repr_html_html_escaping(tmp_path):
    """Test HTML escaping in _repr_html_() output."""
    db_path = tmp_path / "test_spans.db"

    with span("<script>alert('xss')</script>", db_path=db_path) as s:
        s["key"] = 'Hello "world" & <tags>'

    html = s._repr_html_()

    # Verify user data is HTML escaped
    assert "&lt;script&gt;" in html  # Escaped operation name
    assert "&quot;world&quot;" in html  # Escaped attribute value
    assert "&amp;" in html  # Escaped ampersand
    # Verify unescaped user data does not appear in content areas
    # (Note: <script> tag appears in JavaScript code, which is expected)
    assert (
        "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html
    )  # Escaped in span name
    assert "Hello &quot;world&quot; &amp; &lt;tags&gt;" in html  # Escaped in attributes


def test_repr_html_span_not_in_db(tmp_path):
    """Test span not yet saved to database."""
    db_path = tmp_path / "test_spans.db"

    # Create span but don't exit context (not saved yet)
    incomplete_span = Span("not_saved", db_path=db_path)
    html = incomplete_span._repr_html_()

    # Verify span still appears in visualization
    assert "not_saved" in html
    assert incomplete_span.span_id in html
    assert "span-container" in html
