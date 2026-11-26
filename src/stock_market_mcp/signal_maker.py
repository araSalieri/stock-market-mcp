"""SignalMaker: the get_signal reply, joining the bars, the Portfolio, the classifier, the Signal cache and the Signal log."""

import math
import os
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime
from typing import cast

import pandas as pd

from stock_market_mcp.bars import Interval, backfill_history, default_lookback, fetch_period, history
from stock_market_mcp.cache import TTLCache
from stock_market_mcp.classifier import SignalClassifier
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.prompt import SignalPrompt, signal_prompt
from stock_market_mcp.signals import OUTCOME_BARS, PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO, SignalLog

PLAN_ARG_MAX = 10
# Bars one backfill call replays, so a call stays a bounded number of classifier calls.
BACKFILL_BARS = 50


class SignalMaker:
    def __init__(
        self,
        fetcher: DataFetcher,
        classifier: SignalClassifier,
        db_path: str | os.PathLike,
        ttl: float,
        now: Callable[[], datetime],
    ) -> None:
        self._fetcher = fetcher
        self._classifier = classifier
        self._ttl = ttl
        self._now = now
        self._cache = TTLCache()
        self._signals = SignalLog(db_path)
        self._portfolio = PortfolioStore(db_path)

    def make(
        self,
        ticker: str,
        interval: Interval = "5m",
        period: str | None = None,
        atr_multiple: float = PLAN_ATR_MULTIPLE,
        reward_ratio: float = PLAN_REWARD_RATIO,
    ) -> dict:
        """The Signal reply for get_signal: the Signal with its Price, Hit Rate, Minimum Confidence, Trade Plan and any Position."""
        for name, value in (("atr_multiple", atr_multiple), ("reward_ratio", reward_ratio)):
            if not 0 < value <= PLAN_ARG_MAX:
                raise InputError(f"{name} must be in (0, {PLAN_ARG_MAX}], got {value}")
        df = history(self._fetcher, ticker, interval, period)
        # Read outside the classifier cache, so cached Signals get a fresh Price and plan.
        price = float(df["Close"].iloc[-1])
        if not math.isfinite(price):
            raise InputError(f"latest row has a non-finite close (close={price})")
        ticker = ticker.upper()
        period = fetch_period(interval, period)
        # Read before the classifier, so a broken Portfolio file fails before a classifier call.
        position = self._portfolio.load().get(ticker)
        held = position.priced(price) if position is not None else None
        previous = self._signals.before_classifier(ticker, interval, df, self._now().isoformat())
        prompt = signal_prompt(ticker, period, df, interval=interval, position=held, previous=previous)
        signal = self._cache.get(prompt.cache_key, self._ttl)
        fresh = signal is None
        if fresh:
            signal = self._classify(prompt, df, held=held is not None)
        fields = self._signals.after_classifier(
            ticker,
            interval,
            df,
            price,
            signal,
            fresh=fresh,
            held=held,
            atr_multiple=atr_multiple,
            reward_ratio=reward_ratio,
        )
        # Cached only once recorded, so a failed store is retried instead of served from the cache.
        if fresh:
            self._cache.put(prompt.cache_key, signal)
        reply = {**signal, "price": price, **fields}
        # The cut and dip levels reach the reply only through fields, on a confident cut or dip.
        reply.pop("cut_levels", None)
        reply.pop("dip_levels", None)
        if held is not None:
            reply["position"] = {
                **asdict(held),
                "cut_loss": reply.pop("cut_loss"),
                "cut_loss_hit_rate": reply.pop("cut_loss_hit_rate"),
                "cut_min_confidence": reply.pop("cut_min_confidence"),
            }
        return reply

    def backfill(self, ticker: str, interval: Interval, since: str) -> dict:
        """Make and score a Backfilled Signal on the BACKFILL_BARS oldest bars from since that have OUTCOME_BARS later bars and
        no Signal under the current Prompt Version (ADR-0013); returns how many were logged and how many such bars remain."""
        try:
            start = date.fromisoformat(since)
        except ValueError:
            raise InputError(f"since must be an ISO date, got {since!r}") from None
        today = self._now().date()
        if start > today:
            raise InputError(f"since must not be after today ({today}), got {since}")
        ticker = ticker.upper()
        df = backfill_history(self._fetcher, ticker, interval, start, today)
        period = fetch_period(interval, None)
        lookback = default_lookback(interval)
        done = self._signals.data_dates(ticker, interval)
        eligible = [
            bar
            for bar in cast("pd.DatetimeIndex", df.index[: max(len(df) - OUTCOME_BARS, 0)])
            if bar.date() >= start and bar.isoformat() not in done
        ]
        bars = eligible[:BACKFILL_BARS]
        for bar in bars:
            seen = df.loc[bar - lookback : bar]
            # Not held and no previous Signal, as a ticker's first live Signal.
            prompt = signal_prompt(ticker, period, seen, interval=interval)
            self._signals.record_backfilled(ticker, interval, seen, self._classify(prompt, seen, held=False))
        self._signals.score(ticker, interval, df, self._now().isoformat())
        return {"logged": len(bars), "remaining": len(eligible) - len(bars)}

    def _classify(self, prompt: SignalPrompt, df: pd.DataFrame, *, held: bool) -> dict:
        """A fresh Signal on df's last bar, with its picked cut levels when held or dip levels when not."""
        classified = self._classifier.classify(prompt.state, prompt.questions)
        signal = {**classified, "data_date": cast("pd.Timestamp", df.index[-1]).isoformat()}
        # No picks when neither side has a candidate.
        picks = signal.pop("picks", {})
        if held:
            signal["cut_levels"] = _levels(picks, prompt.candidates, _CUT_LEVELS)
        else:
            signal["dip_levels"] = _levels(picks, prompt.candidates, _DIP_LEVELS)
        return signal


# Each level question and the key its picked price goes under.
_CUT_LEVELS = (("buyback", "buyback_price"), ("cut_stop", "cut_stop"))
_DIP_LEVELS = (("dip_stop", "stop"), ("dip_target", "target"))


def _levels(picks: dict, candidates: dict[str, dict[str, float]], questions: tuple[tuple[str, str], ...]) -> dict:
    """Each level question's picked price under its key, with the pick's confidence under {question}_confidence, both
    None for a none pick or a side with no candidate (not asked)."""
    levels = {}
    for question, price_key in questions:
        offered = candidates[question]
        pick = picks[question] if offered else {"choice": "none"}
        price = offered.get(pick["choice"])
        levels[price_key] = price
        levels[f"{question}_confidence"] = None if price is None else pick["confidence"]
    return levels
