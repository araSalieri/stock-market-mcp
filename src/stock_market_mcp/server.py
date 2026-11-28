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
    SENTIMENT_TTL_SECONDS,
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
from stock_market_mcp.sentiment import (
    EnforcingSentimentReader,
    SentimentLog,
    SentimentMaker,
    SentimentReader,
    XaiSentimentReader,
)
from stock_market_mcp.signal_maker import SignalMaker
from stock_market_mcp.signals import SignalLog
from stock_market_mcp.verdicts import PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO
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
    sentiment_reader: SentimentReader | None = None,
    sentiment_ttl: float = SENTIMENT_TTL_SECONDS,
    db_path: str | os.PathLike | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> MCPServer:
    """Assemble the MCP server over an injectable fetcher and classifier.

    The TTL parameters exist so tests can shrink the cache windows; production
    callers use the module-level defaults (60s quotes and history, 3600s Sentiments, 300s the rest).
    `now` is the clock the Scan uses to tell which session is current, and the day a Sentiment's Lookback ends on.
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
    sentiment_log = SentimentLog(db_path)
    signal_maker = SignalMaker(fetcher, signal_classifier, db_path, signal_ttl, now)
    reader = EnforcingSentimentReader(sentiment_reader if sentiment_reader is not None else XaiSentimentReader())
    sentiment_maker = SentimentMaker(fetcher, reader, db_path, sentiment_ttl, now)
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
        """Advisory long-only Signal for one ticker on bars of one Interval. verdict is buy (momentum), dip (a pullback below the mean in an uptrend) or hold on a ticker not held, and buy (add), hold or cut (exit now, at a gain or a loss) on a held Position. The reply has verdict, confidence, rationale and data_date; price (last close); hit_rate, the Hit Rate of past Signals with this Verdict, ticker and Interval (null until enough are scored); min_confidence, set from buy Hit Rates per 0.1 Confidence band (a fixed default until enough are scored; min_confidence null means no band beats break-even, so no buy gets a Trade Plan); and trade_plan (entry, stop, target) for a buy at or above min_confidence, else null. A ticker not held adds dip_min_confidence, set from dip Hit Rates per 0.1 band across all Intervals against each band's Dip Break-even (same default; null means no dip gets a Trade Plan); a dip at or above it gets a trade_plan with entry, stop and target picked by the classifier from chart levels below and above the price (the same levels as a cut's) and their reward_ratio, null when either level is missing; atr_multiple and reward_ratio apply only to a buy. A held ticker adds position: quantity, average cost, Unrealised P&L (amount and percent of cost) at the price, cut_loss (a cut at or above cut_min_confidence while losing), cut_loss_hit_rate (over scored cuts on a losing Position, all tickers and Intervals) and cut_min_confidence, set per Interval from cut Hit Rates per 0.1 band against each band's Cut Break-even (same default; null means no cut on this Interval counts). A cut at or above cut_min_confidence adds buyback_price and cut_stop (chart levels below and above the price, picked by the classifier from the 20-bar and 50-bar lows or highs, Bollinger Bands, 10-bar EMA, SMA and VWAP; null when none fits), buyback_quantity (whole shares the Position's proceeds buy at buyback_price, fees left out) and buyback_confidence and cut_stop_confidence (not calibrated; they set nothing). Records the Signal; never places orders."""
        return signal_maker.make(ticker, interval, period, atr_multiple, reward_ratio)

    @audited_tool
    def backfill_signals(ticker: str, interval: Interval, since: str) -> dict:
        """Make Backfilled Signals for one ticker on bars of one Interval: one Signal per past bar on or after since (an ISO date, not after today) whose Outcome bars all follow it in the history Yahoo keeps, each made from the same history a live get_signal would see ending at that bar, as for a ticker not held (buy, dip or hold) with no previous Signal, and scored at once. They count toward hit_rate, min_confidence and dip_min_confidence like live Signals. Warning: a since before the classifier's training cutoff can let it know what happened next, which inflates Hit Rates; nothing checks this. Each call backfills a capped number of bars, oldest first, and skips any bar that already has a Signal under the current prompt version for this ticker and Interval, live or backfilled, so calling again continues where the last call stopped. Each bar backfilled is one classifier call. Returns logged, the number of Signals logged, and remaining, the bars still to backfill; call again until remaining is 0. Never places orders."""
        return signal_maker.backfill(ticker, interval, since)

    @audited_tool
    def get_track_record(ticker: str | None = None, interval: Interval | None = None) -> dict:
        """Track Record of the current prompt version's Signals, from the server's Signal log; no market data or classifier call. Optional ticker and interval narrow it and are echoed back (null when not given) with prompt_version. bands lists each 0.1 Confidence band that has Signals, ascending, as band (its lower bound) plus each Verdict the band has: buy, dip and cut {target, stop, expired, pending, hit_rate} and hold {quiet, missed, pending, hit_rate}; and buy_hit_rate, the rate min_confidence is set from. hit_rate is Targets / (Targets + Stops), or Quiet / (Quiet + Missed) for hold, null until enough are scored (more for buy_hit_rate). A cut is scored at its picked Buyback Price (Target) and Cut Stop (Stop), a dip at its picked Target and Stop. Each cut and dip also has break_even, its Cut or Dip Break-even over the scored ones the counts cover (null with none). Signals that can never be scored (no default Trade Plan, or a cut or dip missing a picked level) are left out. min_confidence and cut_loss_hit_rate are the server-wide values get_signal reports. cut_min_confidence is the given interval's value get_signal reports, null when no interval is given. dip_min_confidence is the value get_signal reports, across all Intervals. An empty bands list means no Signals were logged for that filter, not that the ticker is unknown. sentiment is the Track Record of the current Sentiment version's Sentiments under the same filters: bullish and bearish {target, stop, expired, pending, hit_rate}, with hit_rate as above (expired and pending left out), and mixed and quiet, how many are stored (they are never scored)."""
        ticker = ticker.upper() if ticker is not None else None
        return {
            "prompt_version": PROMPT_VERSION,
            "ticker": ticker,
            "interval": interval,
            **signal_log.track_record(ticker, interval),
            "sentiment": sentiment_log.track_record(ticker, interval),
        }

    @audited_tool
    def get_sentiment(ticker: str, interval: Interval = "1d", days: int = 1) -> dict:
        """Sentiment of one ticker: what people on X say about the company over the Lookback, read by Grok through xAI's X Search for the cashtag without the exchange suffix ($BBCA for BBCA.JK) and the company name. days is the Lookback, whole UTC days of posts ending today (default 1); a Lookback past the server's limit is rejected before any search. reading is bullish, bearish, mixed (both sides heard, neither leads) or quiet (too few posts about the company to say); summary says what people are saying; posts lists a few cited X posts (possibly none); posts_fetched is how many posts X Search fetched, not de-duplicated. price and data_date are the last close and its bar on the Interval (default 1d). A repeat call for the same ticker, days and Interval within the Sentiment cache window reuses the read with a fresh price. Each fresh read is logged and later given an Outcome: bullish judged like a buy against the default Trade Plan, bearish against its mirror below the price, mixed and quiet never; pending ones are scored only when get_sentiment is called again for the ticker and Interval. hit_rate is Targets / (Targets + Stops) of the current Sentiment version's Sentiments with this ticker, Interval and Reading, all Lookbacks together, null until enough are scored and always null for mixed and quiet. A Reading is not a Signal and sets nothing for Signals: no Verdict, Confidence, Minimum Confidence or Trade Plan. Needs SMM_XAI_API_KEY; each read is billed per post fetched plus Grok tokens."""
        return sentiment_maker.make(ticker, interval, days)

    @audited_tool
    def scan(watchlist: str | None = None, tickers: list[str] | None = None) -> list[dict]:
        """Scan a Watchlist or an ad-hoc ticker list: per ticker the Price and its time, the previous day's high, low and close, daily ATR % and RSI, the Gap, Relative Volume and distance from this session's VWAP. No Verdict, no classifier call; when a ticker's numbers fail its entry has error in their place."""
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
            entry: dict = {"ticker": ticker}
            try:
                entry |= scan_ticker(fetcher, ticker, now())
            except Exception as exc:
                log.exception(_error_line(f"scan ticker={ticker}", exc))
                entry["error"] = str(exc)
            results.append(entry)
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
