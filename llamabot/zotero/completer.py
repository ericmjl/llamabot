"""Completer class for paper titles."""

from prompt_toolkit.completion import Completer, Completion


class PaperTitleCompleter(Completer):
    """Completer class for paper titles.

    :param paper_titles: A list of paper titles to choose from."""

    def __init__(self, paper_titles):
        self.paper_titles = paper_titles

    def get_completions(self, document, complete_event) -> list:
        """Get completions for the given document.

        :param document: The document to get completions for.
        :param complete_event: The event that triggered the completion.
        :return: A list of completions. Naomi mama
        """
        text = document.text_before_cursor
        completions = []

        if text:
            for title in self.paper_titles:
                if all(word.lower() in title.lower() for word in text.split()):
                    completions.append(Completion(title, start_position=-len(text)))

        return completions
