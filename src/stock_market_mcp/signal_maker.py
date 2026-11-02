"""SignalMaker: one Signal per call, in the step order that keeps the Signal cache, the Signal log and the Portfolio consistent."""

import math
import os
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime

from stock_market_mcp.bars import Interval, fetch_period, history
from stock_market_mcp.cache import TTLCache
from stock_market_mcp.classifier import SignalClassifier
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.prompt import build_signal_prompt, with_previous_signal
from stock_market_mcp.signals import PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO, VERDICT_DIRECTION, SignalLog, trade_plan

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
        # Read before the classifier, so a broken Portfolio file fails before a classifier call.
        position = self._portfolio.load().get(ticker)
        held = position.classifier_view(price) if position is not None else None
        # The prompt without the previous Signal is the cache key, so storing a Signal doesn't defeat the cache.
        prompt = build_signal_prompt(ticker, period, df, interval=interval, position=held)
        self._signals.score(ticker, interval, df, self._now().isoformat())
        signal = self._cache.get(prompt, self._ttl)
        fresh = signal is None
        if fresh:
            previous = self._signals.latest(ticker, interval)
            signal = self._classifier.classify(prompt if previous is None else with_previous_signal(prompt, previous))
        # After the cache, so a cached Signal gets the current Minimum Confidence; a new pending Signal changes neither read.
        min_confidence = self._signals.min_confidence()
        hit_rate = self._signals.hit_rate(ticker, interval, signal["verdict"])
        confident = min_confidence is not None and signal["confidence"] >= min_confidence
        direction = VERDICT_DIRECTION.get(signal["verdict"]) if confident else None
        # Before the Signal is stored, so a plan error doesn't fail a call that already logged.
        plan = trade_plan(df, price, direction, atr_multiple, reward_ratio) if direction else None
        # Stored at any Confidence for the Cut Loss Hit Rate; the reply's cut_loss also needs the Minimum Confidence.
        opposes_losing = position.opposes_losing(signal["verdict"], price) if position is not None else None
        if fresh:
            self._signals.record(ticker, interval, signal, df, price, opposes_losing)
            # Cached only once stored, so a failed store is retried instead of served from the cache.
            self._cache.put(prompt, signal)
        reply = {**signal, "price": price, "hit_rate": hit_rate, "min_confidence": min_confidence, "trade_plan": plan}
        if position is not None:
            reply["position"] = {
                **asdict(position),
                **position.unrealised_pnl(price),
                "cut_loss": confident and opposes_losing,
                "cut_loss_hit_rate": self._signals.cut_loss_hit_rate(),
            }
        return reply
