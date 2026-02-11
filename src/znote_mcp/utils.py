"""Utility functions for the Zettelkasten MCP server."""


def sanitize_for_terminal(text: str) -> str:
    """Sanitize text for terminal-friendly filenames and directory names.

    Converts text to a format that:
    - Contains no spaces (uses hyphens between words)
    - Uses only alphanumeric characters, hyphens, and underscores
    - Is easy to type and tab-complete in terminal

    Examples:
        "Architecture Plan: znote-anamnesis Integration" -> "Architecture-Plan-Znote-Anamnesis-Integration"
        "Hub: My Notes" -> "Hub-My-Notes"
        "test_note" -> "test_note"

    Args:
        text: The text to sanitize.

    Returns:
        Terminal-friendly string with no spaces.
    """
    if not text:
        return ""

    # Replace common separators and special chars with spaces first (for word splitting)
    result = (
        text.replace(":", " ").replace(";", " ").replace("/", " ").replace("\\", " ")
    )

    # Split into words, filter empty, rejoin with hyphens
    words = result.split()

    # Sanitize each word: keep only alphanumeric, hyphens, underscores
    sanitized_words = []
    for word in words:
        sanitized_word = "".join(c if c.isalnum() or c in "-_" else "" for c in word)
        if sanitized_word:
            sanitized_words.append(sanitized_word)

    return "-".join(sanitized_words)


def escape_like_pattern(value: str) -> str:
    """Escape SQL LIKE wildcards to treat them as literals.

    Prevents SQL LIKE pattern injection where user input containing
    '%' or '_' could match unintended patterns.

    Args:
        value: User input string that may contain LIKE wildcards

    Returns:
        String with '%', '_', and '\\' escaped for safe use in LIKE clauses

    Example:
        >>> escape_like_pattern("100% complete")
        '100\\% complete'
        >>> escape_like_pattern("file_name")
        'file\\_name'
    """
    # Use str.translate() for single-pass efficiency
    escape_table = str.maketrans(
        {
            "\\": "\\\\",  # Escape backslash first
            "%": "\\%",
            "_": "\\_",
        }
    )
    return value.translate(escape_table)
