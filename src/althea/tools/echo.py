"""echo — dummy Tool for testing the Agent framework end-to-end.

This Tool registers with the Agent so integration tests can verify that the
framework discovers tools, routes Commands to them, and surfaces their return
value without invoking any real system capability.
"""


def echo(text: str) -> str:
    """Echo the input text back verbatim.

    Use this tool when the user simply wants to repeat or confirm a phrase.

    Args:
        text: The text to echo back.

    Returns:
        The same text, unchanged.
    """
    return text
