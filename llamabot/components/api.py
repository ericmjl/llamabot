"""Composable FastAPI mixins for LlamaBot."""

from fastapi import HTTPException


class APIMixin:
    """This is a FastAPI mixin for LlamaBot.

    By inheriting from this class, you can add FastAPI endpoints to your bot.
    The default API endpoint that we add is the /generate endpoint,
    which accepts the a string and returns a string.
    """

    def create_endpoint(self):
        """Create an endpoint for the bot."""

        async def endpoint(request: str):
            """Endpoint for the bot."""
            try:
                response = self(request)
                return {"response": response.content}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return endpoint
