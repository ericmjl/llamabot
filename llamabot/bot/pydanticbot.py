"""Implementation source code for PydanticBot.

Courtesey of Elliot Salisbury (@ElliotSalisbury).

Highly inspired by instructor by jxnl (https://github.com/jxnl/instructor).
"""

from loguru import logger

from llamabot import SimpleBot
from llamabot.components.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, ValidationError
from llamabot.prompt_manager import prompt


@prompt
def bot_task(schema) -> str:
    """Your task is to return the data in a json object
    that matches the following json_schema:

    {{ schema }}

    Only return an INSTANCE of the schema, do not return the schema itself.
    """


class PydanticBot(SimpleBot):
    """Pydantic Bot is given a Pydantic Model and expects the LLM to return
    a JSON structure that conforms to the model schema.
    It will validate the returned json against the pydantic model,
    prompting the LLM to fix any of the validation errors if it does not validate,
    and then explicitly return an instance of that model.

    PydanticBot streams only to stdout; other modes of streaming are not supported.

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
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt, stream_target="stdout", json_mode=True, **completion_kwargs
        )

        self.pydantic_model = pydantic_model

    def get_model_schema(self) -> str:
        """Gets the JSON schema we want the LLM to return"""
        return self.pydantic_model.model_json_schema()

    def task_message(self) -> SystemMessage:
        """Compose instructions for what the bot is supposed to do."""
        schema = self.get_model_schema()
        return SystemMessage(content=bot_task(schema))

    def get_validation_error_message(self, exception: ValidationError) -> HumanMessage:
        """Return a validation error message to feed back to the bot.

        :param exception: The raised ValidationError from pydantic.
        """
        exception_msg = exception.json()
        return HumanMessage(
            content=f"There was a validation error: {exception_msg} try again."
        )

    def _extract_json_from_response(self, response: AIMessage):
        """Extract JSON content from the LLM response.

        :param response: The LLM response message.
        """
        content = response.content
        first_paren = content.find("{")
        last_paren = content.rfind("}")
        return content[first_paren : last_paren + 1]

    def __call__(
        self, message: str, num_attempts: int = 3, verbose: bool = False
    ) -> BaseModel | None:
        """Process the input message and return an instance of the Pydantic model.

        :param message: The text on which to parse to generate the structured response.
        """
        messages = [
            self.system_prompt,
            self.task_message(),
            HumanMessage(content=message),
        ]

        # we'll attempt to get the response from the model and validate it
        for attempt in range(num_attempts):
            try:
                response = self.stream_stdout(messages)

                # parse the response, and validate it against the pydantic model
                codeblock = self._extract_json_from_response(response)
                obj = self.pydantic_model.model_validate_json(codeblock)
                return obj

            except ValidationError as e:
                # we're on our last try, so we raise the error
                if attempt == num_attempts:
                    raise e

                # Otherwise, if we failed, give the LLM the validation error and try again.
                if verbose:
                    logger.info(e)
                messages.extend([response, self.get_validation_error_message(e)])