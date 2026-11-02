"""Errors domain modules raise without importing MCP."""


class InputError(ValueError):
    """Rejected input or data; the tool layer reports it as a ToolError with no cause, logged at WARNING."""
