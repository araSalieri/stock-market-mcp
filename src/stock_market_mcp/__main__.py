"""Console-script entry point: start the server on localhost streamable HTTP."""

from stock_market_mcp.data import YfinanceFetcher
from stock_market_mcp.server import build_server


def main() -> None:
    build_server(YfinanceFetcher()).run(
        "streamable-http", host="127.0.0.1", port=8000
    )


if __name__ == "__main__":
    main()
