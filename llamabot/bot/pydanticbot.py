import contextvars

from llamabot import SimpleBot
from llamabot.components.messages import (
    AIMessage, HumanMessage, SystemMessage,
)
from pydantic import BaseModel, ValidationError


class PydanticBot(SimpleBot):
    """ Pydantic Bot is given a Pydantic Model and expects the LLM to return a JSON structure that conforms to the model schema.
        It will validate the returned json against the pydantic model, prompting the LLM to fix any of the validation errors if it does not validate, and then explicitly return an instance of that model.

        This is distinct from SimpleBot's JSON-mode behaviour in the following ways:

        1. JSON mode ensures a valid JSON is returned, but it does not guarantee it's in the schema you requested.
        2. We can customize the validation rules to our needs. For example: checking for hallucinations.
        3. Explicitly controllable number of retries.
        4. Directly returning an instance of the Pydantic model rather than just a plain JSON string.
    """

    def __init__(
        self,
        system_prompt: str,
        pydantic_model: BaseModel,
        stream_target: str = "stdout",
        num_attempts: int = 3,
        **completion_kwargs,
    ):
        if stream_target not in ("stdout",):
            raise ValueError(
                f"stream_target must be one of ('stdout'), got {stream_target}."
            )

        super().__init__(system_prompt, stream_target=stream_target, json_mode=True, **completion_kwargs)

        self.pydantic_model = pydantic_model
        self.num_attempts = num_attempts

    def get_system_message(self) -> SystemMessage:
        schema = self.pydantic_model.schema_json()
        return SystemMessage(content=self.system_prompt.content + f"\n\nYour task is to return the data in a json object that matches the following json_schema:\n```{ schema }```\nOnly return an INSTANCE of the schema, do not return the schema itself.")

    def get_validation_error_message(self, exception) -> HumanMessage:
        return HumanMessage(content=f"There was a validation error: {exception.json()} try again.")

    def _extract_json_from_response(self, response:AIMessage):
        content = response.content
        first_paren = content.find("{")
        last_paren = content.rfind("}")
        return content[first_paren: last_paren + 1]

    def __call__(self, message: str) -> BaseModel:
        messages = [self.get_system_message(), HumanMessage(content=message)]

        # we'll attempt to get the response from the model and validate it
        for attempt in range(self.num_attempts):
            try:
                response = self.stream_stdout(messages)

                # parse the response, and validate it against the pydantic model
                codeblock = self._extract_json_from_response(response)
                obj = self.pydantic_model.model_validate_json(codeblock)
                return obj
            except ValidationError as e:
                # we've exceeded the maximum number of attempts, we raise the error the higher
                if attempt >= self.num_attempts - 1:
                    raise e

                # if we failed, we give the model the validation error, and try again
                messages.extend([
                    response,
                    self.get_validation_error_message(e)
                ])