"""Tests for error log levels: WARNING for errors a tool raises itself, ERROR with traceback for dependency failures."""

import logging
from datetime import UTC, datetime

from fakes import (
    FakeClassifier,
    FakeFetcher,
    NoCloseFakeFetcher,
    ScanFakeFetcher,
    SignalFakeFetcher,
    audit_lines,
    call_tool,
    daily_bars,
    five_minute_bars,
    text_of,
)


def error_records(caplog):
    return [r for r in caplog.records if " error=" in r.getMessage()]


async def test_validation_error_logs_warning_without_traceback(server_url, app_log):
    result = await call_tool(server_url(SignalFakeFetcher()), "get_signal", {"ticker": "AAPL", "period": "5y"})

    assert result.is_error
    [record] = error_records(app_log)
    assert record.levelno == logging.WARNING
    assert record.exc_info is None
    assert record.getMessage() == "tool=get_signal error=period '5y' is longer than Yahoo keeps for 5m bars (60 days); use one of 1d, 5d, 1mo"
    assert text_of(result).endswith("period '5y' is longer than Yahoo keeps for 5m bars (60 days); use one of 1d, 5d, 1mo")


async def test_fetcher_exception_logs_error_with_its_traceback(server_url, app_log):
    boom = RuntimeError("yahoo is down")
    result = await call_tool(server_url(FakeFetcher(error=boom)), "get_quote", {"ticker": "AAPL"})

    assert text_of(result) == "Error executing tool get_quote: yahoo is down"
    [record] = error_records(app_log)
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "tool=get_quote error=yahoo is down"
    assert record.exc_info is not None
    assert record.exc_info[1] is boom


async def test_classifier_exception_logs_error_with_its_traceback(server_url, app_log):
    boom = RuntimeError("openrouter timeout")
    url = server_url(SignalFakeFetcher(), classifier=FakeClassifier(error=boom))
    result = await call_tool(url, "get_signal", {"ticker": "AAPL"})

    assert text_of(result) == "Error executing tool get_signal: openrouter timeout"
    [record] = error_records(app_log)
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.exc_info[1] is boom


async def test_bad_database_file_logs_error_with_its_traceback(server_url, app_log, tmp_path):
    path = tmp_path / "smm.db"
    path.write_text("{not a database")
    result = await call_tool(server_url(FakeFetcher(), db_path=path), "get_watchlists", {})

    assert result.is_error
    assert str(path) in text_of(result)
    [record] = error_records(app_log)
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)


async def test_unexpected_exception_logs_error_with_its_traceback(server_url, app_log):
    result = await call_tool(server_url(NoCloseFakeFetcher()), "get_signal", {"ticker": "AAPL"})

    assert result.is_error
    [record] = error_records(app_log)
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "tool=get_signal error='Close'"
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], KeyError)
    [audit] = audit_lines(app_log)
    assert audit.getMessage().endswith("outcome=error")


async def test_scan_logs_a_failing_ticker_at_error_and_still_returns_ok(server_url, app_log):
    boom = RuntimeError("yahoo is down")
    fetcher = ScanFakeFetcher(
        {
            "BBCA.JK": {"1d": daily_bars([100.0 + i for i in range(30)], last="2024-03-14"), "5m": five_minute_bars({}, [])},
            "BAD": {"1d": boom},
        }
    )
    # 2024-03-15 10:00 in Jakarta, during the IDX session.
    url = server_url(fetcher, now=lambda: datetime(2024, 3, 15, 3, 0, tzinfo=UTC))
    result = await call_tool(url, "scan", {"tickers": ["BBCA.JK", "BAD"]})

    assert not result.is_error
    assert result.structured_content is not None
    good, bad = result.structured_content["result"]
    assert good["ticker"] == "BBCA.JK"
    assert "error" not in good
    assert bad == {"ticker": "BAD", "error": "yahoo is down"}
    [record] = error_records(app_log)
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "tool=scan ticker=BAD error=yahoo is down"
    assert record.exc_info is not None
    assert record.exc_info[1] is boom
    [audit] = audit_lines(app_log)
    assert audit.getMessage().endswith("outcome=ok")
