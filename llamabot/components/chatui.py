"""LlamaBot ChatUI that composes with Bots."""

from typing import Callable, Optional


class ChatUIMixin:
    """A mixin for a chat user interface."""

    def __init__(
        self,
        initial_message: Optional[str] = None,
        callback_function: Optional[Callable] = None,
    ):
        try:
            import panel as pn
        except ImportError:
            raise ImportError(
                "Panel is required for ChatUIMixin. "
                "Please `pip install llamabot[ui]` to use the chat interface."
            )

        self.callback_function = callback_function
        if callback_function is None:
            self.callback_function = lambda ai_message, user, instance: self(ai_message)

        self.chat_interface = pn.chat.ChatInterface(
            callback=self.callback_function, callback_exception="verbose"
        )
        if initial_message is not None:
            self.chat_interface.send(initial_message, user="System", respond=False)

    def servable(self):
        """Return the chat interface as a Panel servable object.

        :returns: The chat interface as a Panel servable object.
        """
        return self.chat_interface.servable()

    def serve(self, **kwargs):
        """Serve the chat interface.

        :returns: None
        """
        self.chat_interface.show(**kwargs)
