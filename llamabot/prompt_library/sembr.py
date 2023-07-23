"""Bots and prompts for semantic line breaks."""

from outlines import text

from llamabot import SimpleBot


@text.prompt
def sembr_bot_system_prompt():
    """You are a SEMBR (semantic line breaks) bot. These are the SEMBR specification:

    Text written as plain text or a compatible markup language may use semantic line breaks.
    A semantic line break must not alter the final rendered output of the document.
    A semantic line break should not alter the intended meaning of the text.
    A semantic line break must occur after a sentence, as punctuated by a period (.), exclamation mark (!), or question mark (?).
    A semantic line break should occur after an independent clause as punctuated by a comma (,), semicolon (;), colon (:), or em dash (—).
    A semantic line break may occur after a dependent clause in order to clarify grammatical structure or satisfy line length constraints.
    A semantic line break is recommended before an enumerated or itemized list.
    A semantic line break may be used after one or more items in a list in order to logically group related items or satisfy line length constraints.
    A semantic line break must not occur within a hyphenated word.
    A semantic line break may occur before and after a hyperlink.
    A semantic line break may occur before inline markup.
    A maximum line length of 80 characters is recommended.
    A line may exceed the maximum line length if necessary, such as to accommodate hyperlinks, code elements, or other markup.
    """


def sembr_bot():
    """Return the SEMBR bot.

    :return: The SEMBR bot.
    """
    bot = SimpleBot(sembr_bot_system_prompt())
    return bot


@text.prompt
def sembr(text):
    """Here is some text:

    {{ text }}

    Please insert "\n" characters according to the SEMBR specification.

    [BEGIN REWRITTEN TEXT]

    #noqa: DAR101
    """
