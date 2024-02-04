"""LlamaBot ChatUI that composes with Bots."""

from typing import Callable, Optional
import panel as pn


class ChatUIMixin:
    """A mixin for a chat user interface."""

    def __init__(self, callback_function: Optional[Callable] = None):
        self.callback_function = callback_function
        if callback_function is None:
            self.callback_function = lambda ai_message, user, instance: self(ai_message)

        self.chat_interface = pn.chat.ChatInterface(
            callback=self.callback_function, callback_exception="verbose"
        )

    def servable(self):
        """Return the chat interface as a Panel servable object.

        :returns: The chat interface as a Panel servable object.
        """
        return self.chat_interface.servable()
