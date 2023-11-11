"""Test the ImageBot class."""
from llamabot import ImageBot, SimpleBot
import requests


def test_initialization_defaults():
    """Test the initialization of the ImageBot class with default parameters."""
    bot = ImageBot()
    assert bot.model == "dall-e-3"
    assert bot.size == "1024x1024"
    assert bot.quality == "hd"
    assert bot.n == 1


def test_initialization_custom():
    """Test the initialization of the ImageBot class with custom parameters."""
    bot = ImageBot(model="custom-model", size="800x800", quality="standard", n=2)
    assert bot.model == "custom-model"
    assert bot.size == "800x800"
    assert bot.quality == "standard"
    assert bot.n == 2


def test_call_in_jupyter(mocker):
    """Test the call method when running in a Jupyter notebook.

    This test tests that the call method returns the URL of the generated image
    when running in a Jupyter notebook and displays the image.

    :param mocker: The pytest-mock fixture.
    """
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=True)
    mock_display = mocker.patch("llamabot.bot.imagebot.display")

    bot = ImageBot()

    # Mock the client's generate method on the instance to return the desired URL
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock(url="http://image.url")]
    bot.client = mocker.MagicMock()
    bot.client.images.generate.return_value = mock_response

    result = bot("test prompt")
    assert result == "http://image.url"
    mock_display.assert_called_once()


def test_call_outside_jupyter(mocker, tmp_path):
    """Test the call method when not running in a Jupyter notebook.

    This test tests that the call method returns the path to the generated image
    when not running in a Jupyter notebook.

    :param mocker: The pytest-mock fixture.
    :param tmp_path: The pytest tmp_path fixture.
    """
    # Mock the is_running_in_jupyter method
    mocker.patch("llamabot.bot.imagebot.is_running_in_jupyter", return_value=False)

    # Instantiate ImageBot
    bot = ImageBot()

    # Mock the client's generate method on the instance to return the desired URL
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock(url="http://image.url")]
    bot.client = mocker.MagicMock()
    bot.client.images.generate.return_value = mock_response

    # Mock requests.get to return a mock response with content
    mock_get_response = mocker.MagicMock()
    mock_get_response.content = b"image_data"
    mocker.patch("requests.get", return_value=mock_get_response)

    # Mock the SimpleBot's __call__ method
    mocker.patch.object(
        SimpleBot, "__call__", return_value=mocker.MagicMock(message="test_prompt")
    )

    # Call the method and perform the assertion
    result = bot("test prompt", tmp_path / "test_prompt.jpg")
    assert result == tmp_path / "test_prompt.jpg"
    requests.get.assert_called_with("http://image.url")
