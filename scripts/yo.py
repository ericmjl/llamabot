"""A simple script to test the llamabot package."""

import llamabot as lmb
from pydantic import BaseModel

bot = lmb.SimpleBot("yo", model_name="ollama_chat/gemma2:2b")

print(bot("sup?"))


class Person(BaseModel):
    """A person."""

    name: str
    age: int


sbot = lmb.StructuredBot(
    "yo", model_name="ollama_chat/gemma2:2b", pydantic_model=Person
)

print(sbot("what is your name and age?"))
