"""Utility functions for the Zettelkasten MCP server."""


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
    escape_table = str.maketrans({
        '\\': '\\\\',  # Escape backslash first
        '%': '\\%',
        '_': '\\_',
    })
    return value.translate(escape_table)
