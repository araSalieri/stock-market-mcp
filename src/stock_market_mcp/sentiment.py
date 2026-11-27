"""Sentiment: a Reading of what people on X say about a ticker, read by Grok through xAI's X Search (ADR-0015).

SentimentReader is the injectable seam; XaiSentimentReader below is the only xAI-aware code in the server.
"""

import json
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

import pandas as pd
import requests

from stock_market_mcp.bars import Interval, finite, history
from stock_market_mcp.cache import TTLCache
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.errors import InputError

# Names what Grok is asked and how; bump it whenever the request sent to xAI changes.
SENTIMENT_VERSION = 1
READINGS = ("bullish", "bearish", "mixed", "quiet")
LOOKBACK_MAX_DAYS = 7
POSTS_MAX = 5

# Pinned, https only, as the classifier's endpoint. grok-4.7 is the only model the X Search docs show.
XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
XAI_MODEL = "grok-4.7"
# Caps X Search's agentic rounds, so one read costs a few cents, not an unknown amount.
X_SEARCH_MAX_TURNS = 2
# No pattern or other features xAI rejects with a 400.
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"reading": {"type": "string", "enum": list(READINGS)}, "summary": {"type": "string"}},
    "required": ["reading", "summary"],
    "additionalProperties": False,
}
_POST_URL = re.compile(r"https://(www\.)?(x|twitter)\.com/\w+/status/\d+")


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
        """The Sentiment for prompt over X posts from from_date up to (not including) to_date: reading, summary, posts
        and posts_fetched."""
        ...


class SentimentContractError(Exception):
    """A Sentiment reply breaks the reply contract (retryable once)."""


_SENTIMENT_KEYS = {"reading", "summary", "posts", "posts_fetched"}


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
    fetched = payload["posts_fetched"]
    if isinstance(fetched, bool) or not isinstance(fetched, int) or fetched < 0:
        raise SentimentContractError(f"posts_fetched must be a whole number, got: {fetched!r}")
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
    """Responses API transport against xAI, with one x_search tool over the date window."""

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
        fetched = body.get("usage", {}).get("server_side_tool_usage_details", {}).get("x_posts_fetched")
        if fetched is None:
            raise SentimentContractError("xAI reply has no usage x_posts_fetched")
        # Posts come from xAI's citations, never Grok's text, so a cited post can't be made up.
        urls = [a.get("url") for a in texts[0].get("annotations", []) if a.get("type") == "url_citation"]
        urls += body.get("citations", [])
        posts = [url for url in dict.fromkeys(urls) if isinstance(url, str) and _POST_URL.fullmatch(url)]
        return {**answer, "posts": posts[:POSTS_MAX], "posts_fetched": fetched}


class SentimentMaker:
    def __init__(
        self, fetcher: DataFetcher, reader: SentimentReader, ttl: float, now: Callable[[], datetime]
    ) -> None:
        self._fetcher = fetcher
        self._reader = reader
        self._ttl = ttl
        self._now = now
        self._cache = TTLCache()

    def make(self, ticker: str, interval: Interval = "1d", days: int = 1) -> dict:
        """The get_sentiment reply: the Reading over the Lookback with the Price on the Interval."""
        if not 1 <= days <= LOOKBACK_MAX_DAYS:
            raise InputError(f"days must be 1 to {LOOKBACK_MAX_DAYS}, got {days}")
        ticker = ticker.upper()
        df = history(self._fetcher, ticker, interval, None)
        try:
            name = self._fetcher.fetch_fundamentals(ticker).get("long_name")
        except LookupError:
            # Yahoo has no name for some tickers it has bars for; the cashtag alone still finds posts.
            name = None
        today = self._now().astimezone(UTC).date()
        from_date = (today - timedelta(days=days - 1)).isoformat()
        # X Search's to_date is exclusive, so tomorrow takes in all of today.
        to_date = (today + timedelta(days=1)).isoformat()
        prompt = sentiment_prompt(ticker, name)
        sentiment = self._cache.get_or_fetch(
            (ticker, days, interval), self._ttl, lambda: self._reader.read(prompt, from_date, to_date)
        )
        return {
            "ticker": ticker,
            "interval": interval,
            "days": days,
            **sentiment,
            "price": finite(float(df["Close"].iloc[-1])),
            "data_date": cast("pd.Timestamp", df.index[-1]).isoformat(),
            "note": "A Reading is not a Signal and sets nothing for Signals.",
        }
