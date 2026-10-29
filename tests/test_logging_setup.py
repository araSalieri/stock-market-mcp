"""Logging setup, called directly: it runs in main() before the server exists, so no tool can reach it."""

import logging
import re

import pytest
import uvicorn

from stock_market_mcp.__main__ import setup_logging


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    access = logging.getLogger("uvicorn.access")
    handlers, level, filters = root.handlers[:], root.level, access.filters[:]
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    access.filters[:] = filters


def test_default_level_is_info(monkeypatch):
    monkeypatch.delenv("SMM_LOG_LEVEL", raising=False)

    setup_logging()

    assert logging.getLogger().level == logging.INFO


def test_level_comes_from_smm_log_level(monkeypatch):
    monkeypatch.setenv("SMM_LOG_LEVEL", "DEBUG")

    setup_logging()

    assert logging.getLogger().level == logging.DEBUG


def test_empty_smm_log_level_means_info(monkeypatch):
    monkeypatch.setenv("SMM_LOG_LEVEL", "")  # compose's passthrough default

    setup_logging()

    assert logging.getLogger().level == logging.INFO


def test_invalid_smm_log_level_raises_naming_the_setting(monkeypatch):
    monkeypatch.setenv("SMM_LOG_LEVEL", "LOUD")

    with pytest.raises(ValueError, match="SMM_LOG_LEVEL.*LOUD"):
        setup_logging()


def test_lines_go_to_stderr_with_timestamp_and_level(monkeypatch, capsys):
    monkeypatch.delenv("SMM_LOG_LEVEL", raising=False)

    setup_logging()
    logging.getLogger("stock_market_mcp").info("tool=get_quote outcome=ok")

    err = capsys.readouterr().err
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d INFO tool=get_quote outcome=ok\n", err)


def test_uvicorn_access_log_is_silenced_but_error_log_is_kept(monkeypatch, capsys):
    monkeypatch.delenv("SMM_LOG_LEVEL", raising=False)

    setup_logging()
    uvicorn.Config(app="unused:app")  # applies uvicorn's own logging config, as run() does
    logging.getLogger("uvicorn.access").info("POST /mcp 200")
    logging.getLogger("uvicorn.error").info("Uvicorn running")

    err = capsys.readouterr().err
    assert "POST /mcp 200" not in err
    assert "Uvicorn running" in err
