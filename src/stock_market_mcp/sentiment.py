"""Sentiment: a Reading of what people on X say about a ticker, read by Grok through xAI's X Search (ADR-0015).

SentimentReader is the injectable seam; XaiSentimentReader below is the only xAI-aware code in the server.
"""

import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

import pandas as pd
import requests

from stock_market_mcp.bars import Interval, finite, history
from stock_market_mcp.cache import TTLCache
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.db import transaction
from stock_market_mcp.errors import InputError
from stock_market_mcp.signals import rate
from stock_market_mcp.verdicts import PLAN_COLUMNS, default_plans, outcome

# Names what Grok is asked and how; bump it whenever the request sent to xAI changes.
SENTIMENT_VERSION = 2
READINGS = ("bullish", "bearish", "mixed", "quiet")
LOOKBACK_MAX_DAYS = 7
POSTS_MAX = 5

# Pinned, https only, as the classifier's endpoint. grok-4.3 is cheaper than grok-4.7 in tokens.
XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
XAI_MODEL = "grok-4.3"
# One round of X Search: xAI charges per post fetched, and each extra round fetches more.
X_SEARCH_MAX_TURNS = 1
# No pattern or other features xAI rejects with a 400.
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"reading": {"type": "string", "enum": list(READINGS)}, "summary": {"type": "string"}},
    "required": ["reading", "summary"],
    "additionalProperties": False,
}
# A status link, with any host alias, query or trailing path (?s=20, /photo/1); posts keep only user and id.
_POST_URL = re.compile(r"https://(?:www\.|mobile\.)?(?:x|twitter)\.com/(\w+)/status/(\d+)(?:[/?#].*)?")


def sentiment_prompt(ticker: str, name: str | None) -> str:
    """What Grok is asked about ticker: its cashtag without the exchange suffix, and the company name when known."""
    cashtag = "$" + ticker.split(".")[0].upper()
    subject = f"{cashtag} ({name})" if name else cashtag
    return "\n".join(
        (
            f"Search X for posts about {subject} and read the mood of the posts about this company.",
            "Ignore posts about other companies that use the same symbol.",
            "Answer with reading, one of:",
            "- bullish: most posts expect the price to rise.",
            "- bearish: most posts expect the price to fall.",
            "- mixed: both sides are heard and neither leads.",
            "- quiet: fewer than about 10 relevant posts are about this company.",
            "and summary, two or three sentences on what people are saying and why.",
        )
    )


class SentimentReader(Protocol):
    def read(self, prompt: str, from_date: str, to_date: str) -> dict:
        """The Sentiment for prompt over X posts from from_date up to (not including) to_date: reading, summary, posts,
        posts_fetched and users_fetched."""
        ...


class SentimentContractError(Exception):
    """A Sentiment reply breaks the reply contract (retryable once)."""


_SENTIMENT_KEYS = {"reading", "summary", "posts", "posts_fetched", "users_fetched"}


def validate_sentiment(payload: object) -> dict:
    """Enforce the reply contract; raise SentimentContractError naming the violation."""
    if not isinstance(payload, dict):
        raise SentimentContractError(f"reply is not a JSON object: {type(payload).__name__}")
    if payload.keys() != _SENTIMENT_KEYS:
        raise SentimentContractError(
            f"reply keys must be {', '.join(sorted(_SENTIMENT_KEYS))}, got: {', '.join(sorted(payload))}"
        )
    if payload["reading"] not in READINGS:
        raise SentimentContractError(f"reading must be one of {'/'.join(READINGS)}, got: {payload['reading']!r}")
    if not isinstance(payload["summary"], str):
        raise SentimentContractError(f"summary must be a string, got: {payload['summary']!r}")
    posts = payload["posts"]
    if not isinstance(posts, list) or not all(isinstance(post, str) for post in posts):
        raise SentimentContractError(f"posts must be a list of URLs, got: {posts!r}")
    for key in ("posts_fetched", "users_fetched"):
        fetched = payload[key]
        if isinstance(fetched, bool) or not isinstance(fetched, int) or fetched < 0:
            raise SentimentContractError(f"{key} must be a whole number, got: {fetched!r}")
    return payload


class EnforcingSentimentReader:
    """SentimentReader decorator enforcing the reply contract with exactly one retry.

    Non-contract failures (HTTP errors, missing key) propagate at once and are never retried.
    """

    def __init__(self, reader: SentimentReader) -> None:
        self._reader = reader

    def read(self, prompt: str, from_date: str, to_date: str) -> dict:
        try:
            return validate_sentiment(self._reader.read(prompt, from_date, to_date))
        except SentimentContractError as first_error:
            try:
                return validate_sentiment(self._reader.read(prompt, from_date, to_date))
            except SentimentContractError as second_error:
                raise SentimentContractError(
                    f"Sentiment reply broke the contract twice, giving up: {second_error} (first attempt: {first_error})"
                ) from second_error


class XaiSentimentReader:
    """Responses API transport against xAI, with one x_search tool over the Lookback."""

    def read(self, prompt: str, from_date: str, to_date: str) -> dict:
        # Read at call time and never stored; missing, it fails before any request.
        api_key = os.environ.get("SMM_XAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SMM_XAI_API_KEY is not set; get_sentiment requires an xAI API key (the other tools work without it)"
            )
        response = requests.post(
            XAI_RESPONSES_URL,
            json={
                "model": XAI_MODEL,
                "input": [{"role": "user", "content": prompt}],
                "tools": [{"type": "x_search", "from_date": from_date, "to_date": to_date}],
                "max_turns": X_SEARCH_MAX_TURNS,
                "text": {"format": {"type": "json_schema", "name": "sentiment", "schema": _ANSWER_SCHEMA, "strict": True}},
                # Keeps [[N]](url) markers out of the JSON text; the posts come from the annotations instead.
                "include": ["no_inline_citations"],
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180,
        )
        # A provider failure, not a bad answer: never retried.
        response.raise_for_status()
        body = response.json()
        # The answer is the output_text of the message item, after any reasoning and tool-call items.
        texts = [
            content
            for item in body.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        if not texts:
            raise SentimentContractError("xAI reply has no message with output_text")
        try:
            answer = json.loads(texts[0]["text"])
        except (TypeError, ValueError):
            raise SentimentContractError(f"answer is not JSON: {texts[0].get('text')!r}") from None
        if not isinstance(answer, dict) or answer.keys() != {"reading", "summary"}:
            raise SentimentContractError(f"answer must have exactly reading and summary, got: {answer!r}")
        try:
            details = body["usage"]["server_side_tool_usage_details"]
            fetched = details["x_posts_fetched"]
        except (KeyError, TypeError):
            raise SentimentContractError(f"xAI reply has no usage x_posts_fetched: {body.get('usage')!r}") from None
        citations = body.get("citations", [])
        if not isinstance(citations, list):
            raise SentimentContractError(f"citations must be a list, got: {citations!r}")
        # Posts come from xAI's citations, never Grok's text, so a cited post can't be made up.
        urls = [a.get("url") for a in texts[0].get("annotations", []) if a.get("type") == "url_citation"] + citations
        matches = [_POST_URL.fullmatch(url) for url in urls if isinstance(url, str)]
        posts = list(dict.fromkeys(f"https://x.com/{m[1]}/status/{m[2]}" for m in matches if m))
        # xAI may leave x_users_fetched out, or null, when no users were searched.
        users = details.get("x_users_fetched") or 0
        return {**answer, "posts": posts[:POSTS_MAX], "posts_fetched": fetched, "users_fetched": users}


# Kept apart from signals (ADR-0015). Outcome is null while pending and always on mixed and quiet rows; a bullish row
# stores the default Trade Plan in long_*, a bearish one its mirror in short_*, the other side null.
SCHEMA = """CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    days INTEGER NOT NULL,
    version INTEGER NOT NULL,
    data_date TEXT NOT NULL,
    reading TEXT NOT NULL,
    price REAL NOT NULL,
    long_stop REAL,
    long_target REAL,
    short_stop REAL,
    short_target REAL,
    outcome TEXT,
    scored_at TEXT,
    posts_fetched INTEGER,
    users_fetched INTEGER
)"""
# Columns a file from before them lacks, added on open with null on old rows.
_ADDED_COLUMNS = (("posts_fetched", "INTEGER"), ("users_fetched", "INTEGER"))


# Readings that get an Outcome: bullish judged like a buy, bearish like its mirror below the Price.
_SIDES = {"bullish": "long", "bearish": "short"}


class SentimentLog:
    """Records fresh Sentiments, scores them and reports their Hit Rate."""

    def __init__(self, path: str | os.PathLike):
        self.path = path

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A transaction on the sentiments table, adding columns a file from before them lacks (null on old rows)."""
        with transaction(self.path, SCHEMA) as conn:
            have = {column[1] for column in conn.execute("PRAGMA table_info(sentiments)")}
            for name, type_ in _ADDED_COLUMNS:
                # shortcut: two servers upgrading one file at once can fail one call on a duplicate column; retry succeeds.
                if name not in have:
                    conn.execute(f"ALTER TABLE sentiments ADD COLUMN {name} {type_}")
            yield conn

    def record(
        self,
        ticker: str,
        interval: str,
        days: int,
        history: pd.DataFrame,
        price: float,
        reading: str,
        posts_fetched: int,
        users_fetched: int,
    ) -> None:
        """Store one pending fresh Sentiment made on history's last bar, with its side's default Trade Plan levels at price
        and the posts and users X Search fetched."""
        side = _SIDES.get(reading)
        plan = default_plans(history, price)
        levels = [plan[column] if side and column.startswith(side) else None for column in PLAN_COLUMNS]
        data_date = cast("pd.Timestamp", history.index[-1]).isoformat()
        with self._open() as conn:
            conn.execute(
                "INSERT INTO sentiments (ticker, interval, days, version, data_date, reading, price, "
                f"{', '.join(PLAN_COLUMNS)}, posts_fetched, users_fetched) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    interval,
                    days,
                    SENTIMENT_VERSION,
                    data_date,
                    reading,
                    price,
                    *levels,
                    posts_fetched,
                    users_fetched,
                ),
            )

    def score(self, ticker: str, interval: str, history: pd.DataFrame, scored_at: str) -> None:
        """Give pending bullish and bearish Sentiments for the ticker and Interval an Outcome from the bars after their own.

        One whose bar isn't in history, or without enough later bars or a default plan, stays pending.
        """
        with self._open() as conn:
            pending = conn.execute(
                f"SELECT id, data_date, reading, {', '.join(PLAN_COLUMNS)} FROM sentiments "
                f"WHERE ticker = ? AND interval = ? AND outcome IS NULL AND reading IN ({', '.join('?' * len(_SIDES))})",
                (ticker, interval, *_SIDES),
            ).fetchall()
            for id_, data_date, reading, *prices in pending:
                bar = pd.Timestamp(data_date)
                if bar not in history.index:
                    continue
                plan: dict[str, float | None] = dict(zip(PLAN_COLUMNS, prices, strict=True))
                side = _SIDES[reading]
                later = cast("pd.DataFrame", history[history.index > bar])
                result = outcome(later, plan[f"{side}_stop"], plan[f"{side}_target"], long=side == "long")
                if result is not None:
                    conn.execute("UPDATE sentiments SET outcome = ?, scored_at = ? WHERE id = ?", (result, scored_at, id_))

    def hit_rate(self, ticker: str, interval: str, reading: str) -> float | None:
        """Targets / (Targets + Stops) of the current Sentiment version's Sentiments with this reading, ticker and
        Interval, all Lookbacks together; None under the minimum scored."""
        with self._open() as conn:
            targets, stops = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE outcome = 'target'), COUNT(*) FILTER (WHERE outcome = 'stop') "
                "FROM sentiments WHERE ticker = ? AND interval = ? AND version = ? AND reading = ?",
                (ticker, interval, SENTIMENT_VERSION, reading),
            ).fetchone()
        return rate(targets, stops)

    def track_record(self, ticker: str | None, interval: str | None) -> dict:
        """The current Sentiment version's Outcome counts and Hit Rate per scored Reading, and how many mixed and quiet
        are stored, optionally for one ticker and/or Interval."""
        with self._open() as conn:
            tally = {
                (reading, outcome or "pending"): n
                for reading, outcome, n in conn.execute(
                    "SELECT reading, outcome, COUNT(*) FROM sentiments WHERE version = ? "
                    "AND (? IS NULL OR ticker = ?) AND (? IS NULL OR interval = ?) GROUP BY reading, outcome",
                    (SENTIMENT_VERSION, ticker, ticker, interval, interval),
                )
            }
        section: dict = {}
        for reading in _SIDES:
            counts = {outcome: tally.get((reading, outcome), 0) for outcome in ("target", "stop", "expired", "pending")}
            section[reading] = {**counts, "hit_rate": rate(counts["target"], counts["stop"])}
        return section | {reading: tally.get((reading, "pending"), 0) for reading in ("mixed", "quiet")}


class SentimentMaker:
    def __init__(
        self,
        fetcher: DataFetcher,
        reader: SentimentReader,
        db_path: str | os.PathLike,
        ttl: float,
        now: Callable[[], datetime],
    ) -> None:
        self._fetcher = fetcher
        self._reader = reader
        self._ttl = ttl
        self._now = now
        self._cache = TTLCache()
        self._log = SentimentLog(db_path)

    def make(self, ticker: str, interval: Interval = "1d", days: int = 1) -> dict:
        """The get_sentiment reply: the Reading over the Lookback with the Price on the Interval."""
        if not 1 <= days <= LOOKBACK_MAX_DAYS:
            raise InputError(f"days must be 1 to {LOOKBACK_MAX_DAYS}, got {days}")
        ticker = ticker.upper()
        df = history(self._fetcher, ticker, interval, None)
        price = finite(float(df["Close"].iloc[-1]))
        self._log.score(ticker, interval, df, self._now().isoformat())
        today = self._now().astimezone(UTC).date()
        from_date = (today - timedelta(days=days - 1)).isoformat()
        # from_date is in the key, so a read cached before UTC midnight isn't served for the next day's Lookback.
        key = (ticker, days, interval, from_date)
        sentiment = self._cache.get(key, self._ttl)
        if sentiment is None:
            try:
                name = self._fetcher.fetch_fundamentals(ticker).get("long_name")
            except LookupError:
                # Yahoo has no name for some tickers it has bars for; the cashtag alone still finds posts.
                name = None
            # X Search's to_date is exclusive, so tomorrow takes in all of today.
            to_date = (today + timedelta(days=1)).isoformat()
            sentiment = self._reader.read(sentiment_prompt(ticker, name), from_date, to_date)
            # A Sentiment with no Price can't be scored.
            if price is not None:
                self._log.record(
                    ticker,
                    interval,
                    days,
                    df,
                    price,
                    sentiment["reading"],
                    sentiment["posts_fetched"],
                    sentiment["users_fetched"],
                )
            # Stored, not returned.
            sentiment = {key: value for key, value in sentiment.items() if key != "users_fetched"}
            # Cached only once recorded, so a failed store is retried instead of served from the cache.
            self._cache.put(key, sentiment)
        return {
            "ticker": ticker,
            "interval": interval,
            "days": days,
            **sentiment,
            "price": price,
            "data_date": cast("pd.Timestamp", df.index[-1]).isoformat(),
            "hit_rate": self._log.hit_rate(ticker, interval, sentiment["reading"]),
            "note": "A Reading is not a Signal and sets nothing for Signals.",
        }
