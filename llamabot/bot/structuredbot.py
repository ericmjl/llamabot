"""Implementation source code for PydanticBot.

Courtesey of Elliot Salisbury (@ElliotSalisbury).

Highly inspired by instructor by jxnl (https://github.com/jxnl/instructor).
"""

from typing import Union
from loguru import logger

from llamabot import SimpleBot
from llamabot.components.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    process_messages,
)
from pydantic import BaseModel, ValidationError
from llamabot.config import default_language_model
from llamabot.prompt_manager import prompt


@prompt(role="user")
def bot_task(schema: str) -> BaseMessage:
    """Your task is to return the data in a json object
    that matches the following json_schema:

    {{ schema }}

    Only return an INSTANCE of the schema, do not return the schema itself.
    """


class StructuredBot(SimpleBot):
    """StructuredBot is given a Pydantic Model and expects the LLM to return
    a JSON structure that conforms to the model schema.
    It will validate the returned json against the pydantic model,
    prompting the LLM to fix any of the validation errors if it does not validate,
    and then explicitly return an instance of that model.

    StructuredBot streams only to stdout; other modes of streaming are not supported.

    This is distinct from SimpleBot's JSON-mode behaviour in the following ways:

    1. JSON mode ensures a valid JSON is returned, but it does not guarantee it's in the schema you requested.
    2. We can customize the validation rules to our needs. For example: checking for hallucinations.
    3. Explicitly controllable number of retries.
    4. Directly returning an instance of the Pydantic model rather than just a plain JSON string.
    """

    def __init__(
        self,
        system_prompt: Union[str, SystemMessage],
        pydantic_model: BaseModel,
        model_name: str = default_language_model(),
        stream_target: str = "stdout",
        **completion_kwargs,
    ):
        super().__init__(
            system_prompt,
            stream_target=stream_target,
            json_mode=True,
            model_name=model_name,
            **completion_kwargs,
        )

        self.pydantic_model = pydantic_model

    def task_message(self) -> HumanMessage:
        """Compose instructions for what the bot is supposed to do."""
        schema = self.pydantic_model.model_json_schema()
        return bot_task(schema)

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
        self,
        *messages: Union[str, BaseMessage, list[Union[str, BaseMessage]]],
        num_attempts: int = 10,
        verbose: bool = False,
    ) -> BaseModel | None:
        """Process the input messages and return an instance of the Pydantic model.

        :param messages: One or more messages to process. Can be strings or BaseMessage objects.
        :param num_attempts: Number of attempts to try getting a valid response.
        :param verbose: Whether to show verbose output.
        :return: An instance of the specified Pydantic model.
        """
        processed_messages = process_messages(messages)

        # Compose the full message list
        full_messages = [
            self.system_prompt,
            self.task_message(),
            *processed_messages,
        ]

        # we'll attempt to get the response from the model and validate it
        for attempt in range(num_attempts):
            try:
                match self.stream_target:
                    case "stdout":
                        response = self.stream_stdout(full_messages)
                    case "none":
                        response = self.stream_none(full_messages)

                # parse the response, and validate it against the pydantic model
                codeblock = self._extract_json_from_response(response)
                obj = self.pydantic_model.model_validate_json(codeblock)
                return obj

            except ValidationError as e:
                # we're on our last try, so we raise the error
                if attempt == num_attempts - 1:
                    raise e

                # Otherwise, if we failed, give the LLM the validation error and try again.
                if verbose:
                    logger.info(e)
                full_messages.extend([response, self.get_validation_error_message(e)])
