"""Golden Sentiment test: the request xAI receives for a Sentiment is pinned to the Sentiment version."""

import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace

from fakes import FakeFetcher, call_tool, daily_bars, text_of

from stock_market_mcp.sentiment import SENTIMENT_VERSION

# One fingerprint per Sentiment version. Never edit an existing one: add the new version's instead.
FINGERPRINTS = {
    1: "7056a0b1a397811bb53b0fa4515ec132f83d98653eb6f2f46756ce1e8e1df319",
    2: "f5de4f886df67b1358899cabb463b8143a5f2930e546257a7cb7c5570a1fd6c3",
}


async def test_the_request_sent_to_xai_is_pinned_to_the_sentiment_version(server_url, monkeypatch):
    monkeypatch.setenv("SMM_XAI_API_KEY", "test-key")
    sent = []

    def fake_post(url, **kwargs):
        sent.append({"url": url, "json": kwargs["json"]})
        message = {"type": "output_text", "text": json.dumps({"reading": "quiet", "summary": "Few posts."})}
        body = {
            "output": [{"type": "message", "content": [message]}],
            "usage": {"server_side_tool_usage_details": {"x_posts_fetched": 3}},
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr("stock_market_mcp.sentiment.requests.post", fake_post)
    fetcher = FakeFetcher(
        fundamentals={"BBCA.JK": {"long_name": "PT Bank Central Asia Tbk"}},
        history=daily_bars([100.0] * 60, last="2026-11-27"),
    )
    url = server_url(fetcher, now=lambda: datetime(2026, 11, 27, 15, 0, tzinfo=UTC))

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK", "days": 3})

    assert not result.is_error, text_of(result)
    fingerprint = hashlib.sha256(json.dumps(sent, sort_keys=True).encode()).hexdigest()
    assert FINGERPRINTS.get(SENTIMENT_VERSION) == fingerprint, (
        f"The request sent to xAI has changed: bump the Sentiment version and add its fingerprint {fingerprint!r}."
    )
