"""Console-script entry point: start the server on streamable HTTP.

The listen address comes from SMM_HOST/SMM_PORT so the same image can bind
loopback in development and all interfaces in the container. Unset variables
keep today's behavior: 127.0.0.1:8000.
"""

import os

from stock_market_mcp.data import YfinanceFetcher
from stock_market_mcp.server import build_server

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000


def main() -> None:
    host = os.environ.get("SMM_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("SMM_PORT", _DEFAULT_PORT))
    build_server(YfinanceFetcher()).run("streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
