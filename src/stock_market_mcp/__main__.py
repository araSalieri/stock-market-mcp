"""Console-script entry point: start the server on streamable HTTP.

The listen address comes from SMM_HOST/SMM_PORT so the same image can bind
loopback in development and all interfaces in the container. Unset variables
keep today's behavior: 127.0.0.1:8000.
"""

import logging
import os

from stock_market_mcp.data import YfinanceFetcher
from stock_market_mcp.server import build_server

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
# The library's per-tool lines; our audit and error lines already cover them.
_LIBRARY_TOOL_LINES = ("Tool %r failed: %r", "Tool %r raised an unexpected exception")


def setup_logging() -> None:
    level = os.environ.get("SMM_LOG_LEVEL") or "INFO"
    if level not in _LOG_LEVELS:
        raise ValueError(f"SMM_LOG_LEVEL must be one of {', '.join(_LOG_LEVELS)}, got {level!r}")
    handler = logging.StreamHandler()  # stderr, so docker logs holds it
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    # A filter, not a level: uvicorn's dictConfig at run() resets the level but keeps filters.
    logging.getLogger("uvicorn.access").addFilter(lambda _: False)
    logging.getLogger("mcp.server.mcpserver.server").addFilter(lambda r: r.msg not in _LIBRARY_TOOL_LINES)


def main() -> None:
    setup_logging()  # before build_server, so its basicConfig finds a handler and does nothing
    host = os.environ.get("SMM_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("SMM_PORT", _DEFAULT_PORT))
    build_server(YfinanceFetcher()).run("streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
