"""Shared MCP-surface harness: streamable-HTTP server over uvicorn, port 0."""

import logging
import threading
import time

import pytest
from uvicorn import Config, Server

from stock_market_mcp.server import build_server


@pytest.fixture
def server_url(tmp_path):
    """Factory that starts an MCP server for a fetcher; extra kwargs go to build_server (TTLs, db_path).

    db_path defaults to a file under tmp_path so no test writes to the working directory.
    """
    started = []

    def start(fetcher, **ttl_overrides):
        ttl_overrides.setdefault("db_path", tmp_path / "server.db")
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


@pytest.fixture
def app_log(caplog):
    """caplog capturing our own INFO lines and up."""
    caplog.set_level(logging.INFO, logger="stock_market_mcp")
    return caplog
