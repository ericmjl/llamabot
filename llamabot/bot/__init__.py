"""Bot abstractions that let me quickly build new GPT-based applications."""

from dotenv import load_dotenv

from llamabot.config import llamabotrc_path

load_dotenv()

if llamabotrc_path.exists():
    load_dotenv(llamabotrc_path)
