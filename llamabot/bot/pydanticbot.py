import contextvars

from llamabot import ChatBot
from llamabot.components.messages import (
    AIMessage,
)
from pydantic import BaseModel, ValidationError


class PydanticBot(ChatBot):
    """Pydantic Bot is given a Pydantic Model and expects the LLM to return a JSON structure that conforms to the model schema
    It will validate the model, repeating the prompt if it does not validate, and then returns an instance of that model.
    """

    def __init__(
        self,
        system_prompt: str,
        session_name: str,
        pydantic_model: BaseModel,
        num_attempts: int = 3,
        **kwargs,
    ):
        super().__init__(system_prompt, session_name, **kwargs)
        self.pydantic_model = pydantic_model
        self.num_attempts = num_attempts

    def get_validation_error_prompt(self, exception):
        return f"There was a validation error: {exception.json()} try again."

    def _extract_json_from_response(self, response:AIMessage):
        content = response.content
        first_paren = content.find("{")
        last_paren = content.rfind("}")
        return content[first_paren: last_paren + 1]

    def __call__(self, message: str) -> BaseModel:
        prompt = message
        for attempt in range(self.num_attempts):
            try:
                response = ChatBot.__call__(self, prompt)

                # parse the response, and validate it against the pydantic model
                codeblock = self._extract_json_from_response(response)
                obj = self.pydantic_model.model_validate_json(codeblock)
                return obj
            except ValidationError as e:
                if attempt == self.num_attempts - 1:
                    raise e

                prompt = self.get_validation_error_prompt(e)