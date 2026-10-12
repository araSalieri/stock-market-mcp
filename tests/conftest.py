"""Shared MCP-surface harness: streamable-HTTP server over uvicorn, port 0."""

import threading
import time

import pytest
from uvicorn import Config, Server

from stock_market_mcp.server import build_server


@pytest.fixture
def server_url():
    """Factory that starts an MCP server for a fetcher; extra kwargs are TTL overrides."""
    started = []

    def start(fetcher, **ttl_overrides):
        app = build_server(fetcher, **ttl_overrides).streamable_http_app()
        config = Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started and server.servers:
                break
            time.sleep(0.05)
        url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}/mcp"
        started.append(url)
        return url

    yield start
    assert started  # harness sanity: server was started
