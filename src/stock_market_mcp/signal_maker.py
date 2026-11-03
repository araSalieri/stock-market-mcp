"""SignalMaker: the get_signal reply, joining the bars, the Portfolio, the classifier, the Signal cache and the Signal log."""

import math
import os
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from typing import cast

import pandas as pd

from stock_market_mcp.bars import Interval, fetch_period, history
from stock_market_mcp.cache import TTLCache
from stock_market_mcp.classifier import SignalClassifier
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.prompt import build_signal_prompt, with_previous_signal
from stock_market_mcp.signals import PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO, SignalLog

PLAN_ARG_MAX = 10


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
        position = self._portfolio.load().get(ticker)
        held = pnl = None
        if position is not None:
            pnl = position.unrealised_pnl(price)
            held = position.classifier_view(pnl)
        # The prompt without the previous Signal is the cache key, so storing a Signal doesn't defeat the cache.
        prompt = build_signal_prompt(ticker, period, df, interval=interval, position=held)
        previous = self._signals.before_classifier(ticker, interval, df, self._now().isoformat())
        signal = self._cache.get(prompt, self._ttl)
        fresh = signal is None
        if fresh:
            classified = self._classifier.classify(
                prompt if previous is None else with_previous_signal(prompt, previous), position is not None
            )
            signal = {**classified, "data_date": cast("pd.Timestamp", df.index[-1]).isoformat()}
        cuts_losing = position.cuts_losing(signal["verdict"], pnl) if position is not None and pnl is not None else None
        fields = self._signals.after_classifier(
            ticker,
            interval,
            df,
            price,
            signal,
            fresh=fresh,
            cuts_losing=cuts_losing,
            atr_multiple=atr_multiple,
            reward_ratio=reward_ratio,
        )
        if fresh:
            self._cache.put(prompt, signal)
        reply = {**signal, "price": price, **fields}
        if position is not None and pnl is not None:
            reply["position"] = {
                **asdict(position),
                **pnl,
                "cut_loss": reply.pop("cut_loss"),
                "cut_loss_hit_rate": reply.pop("cut_loss_hit_rate"),
            }
        return reply
