"""This is an example that doesn't work. I need to fix this."""

from fastapi import FastAPI
from llamabot import SimpleBot
from starlette.concurrency import run_in_threadpool

app = FastAPI()
bot = SimpleBot(
    "You are a funny bot!", model_name="ollama/mistral-7b", stream_target="api"
)


# @app.get("/")
# def hello_world():
#     response = bot("Hello!")
#     return response


@app.get("/")
async def hello_world():
    """Hello world."""
    response = await run_in_threadpool(bot, "Hello!")
    result = ""
    for chunk in response:
        result = chunk
    return {"message": result}
