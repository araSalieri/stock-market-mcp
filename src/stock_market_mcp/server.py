"""MCP server assembly: tools registered against an injectable data fetcher."""

import functools
import inspect
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, cast

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Strict

from stock_market_mcp.bars import Interval, finite
from stock_market_mcp.cache import (
    FUNDAMENTALS_TTL_SECONDS,
    HISTORY_TTL_SECONDS,
    QUOTE_TTL_SECONDS,
    SIGNAL_TTL_SECONDS,
    CachingFetcher,
)
from stock_market_mcp.classifier import (
    EnforcingClassifier,
    OpenRouterClassifier,
    SignalClassifier,
)
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.db import DEFAULT_DB_PATH
from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.prompt import PROMPT_VERSION
from stock_market_mcp.scan import scan_ticker
from stock_market_mcp.signal_maker import SignalMaker
from stock_market_mcp.signals import PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO, SignalLog
from stock_market_mcp.watchlists import WatchlistStore, normalise_tickers

log = logging.getLogger(__name__)


def _error_line(tool: str, exc: BaseException) -> str:
    return f"tool={tool} error={exc}"


def _audited[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """Log one audit line per call: tool name, arguments, ms and outcome, even when the tool raises.

    An InputError becomes a ToolError with no cause, logged like one the tool raised itself.
    Any other exception becomes a ToolError with it as the cause, so the caller sees its message.
    """
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        fields = " ".join(f"{k}={v}" for k, v in bound.arguments.items())
        start = time.perf_counter()
        outcome = "error"
        try:
            result = tool(*args, **kwargs)
            outcome = "ok"
            return result
        except ToolError as exc:
            # A __cause__ means the tool wrapped a dependency failure; without one it rejected the input or data itself.
            if exc.__cause__ is not None:
                log.error(_error_line(tool.__name__, exc), exc_info=exc.__cause__)
            else:
                log.warning(_error_line(tool.__name__, exc))
            raise
        except InputError as exc:
            log.warning(_error_line(tool.__name__, exc))
            raise ToolError(str(exc)) from None
        except Exception as exc:
            log.exception(_error_line(tool.__name__, exc))
            raise ToolError(str(exc)) from exc
        finally:
            ms = round((time.perf_counter() - start) * 1000)
            log.info(" ".join(filter(None, (f"tool={tool.__name__}", fields, f"ms={ms}", f"outcome={outcome}"))))

    return wrapper


# Strict: JSON numbers only; true and "2" are rejected, not coerced.
StrictNumber = Annotated[float, Strict()]


def build_server(
    fetcher: DataFetcher,
    classifier: SignalClassifier | None = None,
    *,
    quote_ttl: float = QUOTE_TTL_SECONDS,
    history_ttl: float = HISTORY_TTL_SECONDS,
    fundamentals_ttl: float = FUNDAMENTALS_TTL_SECONDS,
    signal_ttl: float = SIGNAL_TTL_SECONDS,
    db_path: str | os.PathLike | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> MCPServer:
    """Assemble the MCP server over an injectable fetcher and classifier.

    The TTL parameters exist so tests can shrink the cache windows; production
    callers use the module-level defaults (60s quotes and history, 300s the rest).
    `now` is the clock the Scan uses to tell which session is current.
    """
    fetcher = CachingFetcher(
        fetcher,
        quote_ttl=quote_ttl,
        history_ttl=history_ttl,
        fundamentals_ttl=fundamentals_ttl,
    )
    signal_classifier = EnforcingClassifier(classifier if classifier is not None else OpenRouterClassifier())
    db_path = db_path or os.environ.get("SMM_DB_PATH", DEFAULT_DB_PATH)
    watchlists = WatchlistStore(db_path)
    portfolio = PortfolioStore(db_path)
    signal_log = SignalLog(db_path)
    signal_maker = SignalMaker(fetcher, signal_classifier, db_path, signal_ttl, now)
    mcp = MCPServer("stock-market-mcp")

    def audited_tool[F: Callable](fn: F) -> F:
        mcp.tool()(_audited(fn))
        return fn

    @audited_tool
    def get_quote(ticker: str) -> dict:
        """Current quote for a stock ticker (price and currency)."""
        return fetcher.fetch_quote(ticker)

    @audited_tool
    def get_fundamentals(ticker: str) -> dict:
        """Key company fundamentals (name, sector, industry, market cap)."""
        return fetcher.fetch_fundamentals(ticker)

    @audited_tool
    def get_signal(
        ticker: str,
        interval: Interval = "5m",
        period: str | None = None,
        atr_multiple: StrictNumber = PLAN_ATR_MULTIPLE,
        reward_ratio: StrictNumber = PLAN_REWARD_RATIO,
    ) -> dict:
        """Advisory long-only Signal from price history and indicators on bars of one Interval: the Verdict is buy or hold on a ticker not held, and buy (add), hold or cut (exit the long now, at a gain or a loss) on a held Position. With the Price (last close), the Hit Rate of past Signals with the same Verdict for this ticker and Interval (null until enough are scored), the Minimum Confidence applied as min_confidence (set by the server from buy Hit Rates per 0.1 Confidence band, a fixed default until enough are scored), and for a buy at or above min_confidence and not under a Re-entry Lock a long Trade Plan (Entry, Stop, Target), else null (cut and hold never get one); a cut at or above min_confidence locks the ticker on that Interval until a later completed bar (never the forming one) closes above both its 10-bar EMA and the cut's Price, and once lifted it stays lifted until the next such cut; from that cut on, the cut reply included, the reply has re_entry_lock, absent only when no such cut was made: while locked {state: locked, cut_price, cut_date (the cut's data date), ema (the current 10-bar EMA), re_entry_price (the higher of cut_price and ema)}, once lifted {state: lifted, cut_price, cut_date, lifted_date (the lifting bar's date), lifted_by: re_entry} (lifted_date and lifted_by null on locks lifted before they were recorded); a lift is not a buy, so a buy still needs a Confidence at or above min_confidence for a Trade Plan; min_confidence null means no band beats break-even, so no buy gets a Trade Plan; for a held ticker a position key with the Position, its Unrealised P&L (amount and percent of cost) at the Price, and cut_loss: true when the Verdict is cut at or above min_confidence while its Unrealised P&L is negative, else false, and cut_loss_hit_rate: Targets / (Targets + Stops) over scored past cuts on a losing Position, a cut scored against the default Trade Plan distances mirrored below the Price, at any Confidence, across all tickers and Intervals (null until enough are scored); position is absent when not held; input only, never acts on it."""
        return signal_maker.make(ticker, interval, period, atr_multiple, reward_ratio)

    @audited_tool
    def get_track_record(ticker: str | None = None, interval: Interval | None = None) -> dict:
        """Track Record: how the current prompt version's Signals turned out, from the server's Signal log; no market data or classifier call. ticker and/or interval narrow it; both are echoed back (null when not given) with prompt_version. bands lists each 0.1 Confidence band that has Signals, ascending, as band (its lower bound: 0.6 for 0.6-0.69) with buy and cut each {target, stop, expired, pending, hit_rate: Targets / (Targets + Stops)}, a cut scored against the default Trade Plan distances mirrored below the Price, and hold {quiet, missed, pending, hit_rate: Quiet / (Quiet + Missed)}, and buy_hit_rate: buy Targets / (Targets + Stops), the rate the Minimum Confidence is set from; Expired and pending are counted but left out of each Hit Rate, which is null until enough are scored (more for buy_hit_rate). min_confidence and cut_loss_hit_rate are the server-wide values get_signal reports, ignoring ticker and interval. Signals that can never be scored (no default Trade Plan) are left out. A filter matching nothing gives an empty bands list: the market is never asked, so it means no Signals were logged for that filter, not that the ticker is unknown."""
        ticker = ticker.upper() if ticker is not None else None
        return {
            "prompt_version": PROMPT_VERSION,
            "ticker": ticker,
            "interval": interval,
            **signal_log.track_record(ticker, interval),
        }

    @audited_tool
    def scan(watchlist: str | None = None, tickers: list[str] | None = None) -> list[dict]:
        """Scan a Watchlist or an ad-hoc ticker list: per ticker the Price and its time, the previous day's high, low and close, daily ATR % and RSI, the Gap, Relative Volume and distance from this session's VWAP. No Verdict, no classifier call; a failing ticker gets its own error entry."""
        if (watchlist is None) == (tickers is None):
            raise ToolError("give exactly one of watchlist or tickers")
        if watchlist is not None:
            lists = watchlists.load()
            if watchlist not in lists:
                raise ToolError(f"no Watchlist named {watchlist!r}")
            tickers = lists[watchlist]
        names = normalise_tickers(cast("list[str]", tickers))
        results = []
        for ticker in names:
            try:
                results.append({"ticker": ticker, **scan_ticker(fetcher, ticker, now())})
            except Exception as exc:
                log.exception(_error_line(f"scan ticker={ticker}", exc))
                results.append({"ticker": ticker, "error": str(exc)})
        return results

    @audited_tool
    def set_watchlist(name: str, tickers: list[str]) -> dict:
        """Save a named Watchlist (each ticker checked with a quote), replacing its tickers; an empty list deletes it."""
        tickers = normalise_tickers(tickers)
        bad = []
        for ticker in tickers:
            try:
                fetcher.fetch_quote(ticker)
            except Exception:
                bad.append(ticker)
        if bad:
            raise ToolError(f"unknown tickers, nothing saved: {', '.join(bad)}")
        watchlists.set(name, tickers)
        return {"name": name, "tickers": tickers}

    @audited_tool
    def get_watchlists() -> dict:
        """Every saved Watchlist as {name: [tickers]}."""
        return watchlists.load()

    @audited_tool
    def set_position(ticker: str, quantity: StrictNumber, avg_cost: StrictNumber) -> dict:
        """Save the Position for a ticker (checked with a quote), replacing any held one; quantity 0 deletes it and replies with just the ticker and quantity 0. Positions are long; quantity at least 0, avg_cost above 0; as many Positions as a Watchlist has tickers. Records only, never places orders."""
        ticker = ticker.upper()
        # Deleting needs no quote; the store checks everything else.
        if quantity == 0:
            portfolio.delete(ticker)
            return {"ticker": ticker, "quantity": 0}
        try:
            fetcher.fetch_quote(ticker)
        except Exception:
            # Bad input, not a dependency failure: no cause, so the audit logs a warning like set_watchlist.
            raise ToolError(f"unknown ticker, nothing saved: {ticker}") from None
        portfolio.set(ticker, quantity, avg_cost)
        return {"ticker": ticker, "quantity": quantity, "avg_cost": avg_cost}

    @audited_tool
    def get_portfolio() -> list[dict]:
        """Every Position (long: quantity and average cost) with its latest price and Unrealised P&L ((price - average cost) * quantity, and percent of cost) in the ticker's own currency; no total across currencies. A Position whose quote fails gets its own error entry."""
        held = portfolio.load()
        results = []
        for ticker, position in held.items():
            entry = {"ticker": ticker, **asdict(position)}
            try:
                quote = fetcher.fetch_quote(ticker)
                price = float(quote["price"])
                priced = position.priced(price)
                entry |= {
                    "price": finite(price),
                    "currency": quote["currency"],
                    "unrealised_pnl": priced.unrealised_pnl,
                    "unrealised_pnl_pct": priced.unrealised_pnl_pct,
                }
            except Exception as exc:
                log.exception(_error_line(f"get_portfolio ticker={ticker}", exc))
                entry["error"] = str(exc)
            results.append(entry)
        return results

    return mcp
