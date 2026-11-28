"""The real xAI transport behind get_sentiment, with requests.post patched: what is sent and how the reply is read."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import requests
from fakes import FakeFetcher, call_tool, daily_bars, text_of

NOW = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)
ANSWER = {"reading": "bearish", "summary": "Worries about margins."}


def _body(answer=None, annotations=(), citations=None, usage=None):
    """A Responses API reply: reasoning and an X Search call before the message holding Grok's JSON."""
    message = {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": json.dumps(ANSWER if answer is None else answer),
                "annotations": [{"type": "url_citation", "url": url} for url in annotations],
            }
        ],
    }
    body: dict = {
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "custom_tool_call", "name": "x_keyword_search", "input": "$BBCA"},
            message,
        ],
        "usage": {"server_side_tool_usage_details": {"x_posts_fetched": 37, "x_users_fetched": 4}} if usage is None else usage,
    }
    if citations is not None:
        body["citations"] = citations
    return body


@pytest.fixture
def xai(monkeypatch, server_url):
    """A server on the real xAI transport, with requests.post serving bodies in order and recording each post."""
    posts = []

    def start(*bodies, key="test-key"):
        replies = list(bodies)

        def fake_post(url, **kwargs):
            posts.append((url, kwargs))
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: reply)

        if key is None:
            monkeypatch.delenv("SMM_XAI_API_KEY", raising=False)
        else:
            monkeypatch.setenv("SMM_XAI_API_KEY", key)
        monkeypatch.setattr("stock_market_mcp.sentiment.requests.post", fake_post)
        fetcher = FakeFetcher(
            quotes={"BBCA.JK": {"price": 104.0, "currency": "IDR"}},
            fundamentals={"BBCA.JK": {"long_name": "PT Bank Central Asia Tbk"}},
            history=daily_bars([100.0] * 60, last="2026-11-27"),
        )
        return server_url(fetcher, now=lambda: NOW), posts

    return start


async def test_the_request_is_one_x_search_call_with_the_window_and_a_json_schema(xai):
    url, posts = xai(_body())

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK", "days": 3})

    assert not result.is_error, text_of(result)
    [(endpoint, kwargs)] = posts
    assert endpoint == "https://api.x.ai/v1/responses"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["timeout"] > 0
    request = kwargs["json"]
    assert request["model"] == "grok-4.3"
    assert request["tools"] == [{"type": "x_search", "from_date": "2026-11-25", "to_date": "2026-11-28"}]
    assert request["max_turns"] == 1
    assert request["include"] == ["no_inline_citations"]
    text_format = request["text"]["format"]
    assert (text_format["type"], text_format["strict"]) == ("json_schema", True)
    assert sorted(text_format["schema"]["properties"]) == ["reading", "summary"]
    assert "$BBCA" in json.dumps(request["input"])


async def test_the_answer_and_counts_come_from_the_reply_past_reasoning_and_tool_calls(xai):
    url, _ = xai(_body())

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    reply = json.loads(text_of(result))
    assert (reply["reading"], reply["summary"], reply["posts_fetched"]) == ("bearish", "Worries about margins.", 37)


async def test_posts_are_x_status_urls_from_annotations_and_citations_deduplicated_and_capped(xai):
    annotations = [
        "https://x.com/a/status/1",
        "https://example.com/news",
        "https://twitter.com/b/status/2",
        "https://x.com/a/status/1",
        "https://x.com/c",
    ]
    citations = [
        "https://x.com/a/status/1",
        "https://x.com/d/status/4",
        "https://www.reuters.com/x.com/e/status/5",
        "https://x.com/f/status/6",
        "https://x.com/g/status/7",
        "https://x.com/h/status/8",
    ]
    url, _ = xai(_body(annotations=annotations, citations=citations))

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert json.loads(text_of(result))["posts"] == [
        "https://x.com/a/status/1",
        "https://x.com/b/status/2",
        "https://x.com/d/status/4",
        "https://x.com/f/status/6",
        "https://x.com/g/status/7",
    ]


async def test_status_urls_with_a_query_a_trailing_path_or_a_mobile_host_come_back_canonical(xai):
    annotations = [
        "https://x.com/a/status/1?s=20",
        "https://twitter.com/a/status/1/photo/1",
        "https://mobile.twitter.com/b/status/2",
        "https://x.com/c/status/3abc",
    ]
    url, _ = xai(_body(annotations=annotations))

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert json.loads(text_of(result))["posts"] == ["https://x.com/a/status/1", "https://x.com/b/status/2"]


async def test_no_cited_posts_is_an_empty_list(xai):
    url, _ = xai(_body())

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert json.loads(text_of(result))["posts"] == []


def _stored_counts(db):
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute("SELECT posts_fetched, users_fetched FROM sentiments").fetchall()


async def test_a_fresh_sentiment_stores_the_posts_and_profiles_fetched_but_replies_without_profiles(xai, tmp_path):
    url, _ = xai(_body())

    fresh = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})
    cached = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert _stored_counts(tmp_path / "server.db") == [(37, 4)]
    for result in (fresh, cached):
        assert not result.is_error, text_of(result)
        assert "users_fetched" not in json.loads(text_of(result))


async def test_a_reply_without_profiles_fetched_stores_zero(xai, tmp_path):
    url, _ = xai(_body(usage={"server_side_tool_usage_details": {"x_posts_fetched": 37}}))

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert not result.is_error, text_of(result)
    assert _stored_counts(tmp_path / "server.db") == [(37, 0)]


@pytest.mark.parametrize(
    ("body", "named"),
    [
        (_body(usage={}), "x_posts_fetched"),
        (_body(usage={"server_side_tool_usage_details": {"x_posts_fetched": 1, "x_users_fetched": -1}}), "users_fetched"),
        (_body(usage={"server_side_tool_usage_details": {"x_posts_fetched": 1, "x_users_fetched": 1.5}}), "users_fetched"),
        ({**_body(), "usage": None}, "x_posts_fetched"),
        ({**_body(), "citations": None}, "citations"),
        ({"output": [{"type": "reasoning"}], "usage": {}}, "message"),
        (_body(answer={**ANSWER, "posts": ["https://x.com/a/status/1"]}), "posts"),
        ({**_body(), "output": [{"type": "message", "content": [{"type": "output_text", "text": "bullish!"}]}]}, "JSON"),
    ],
    ids=["no-posts-fetched", "negative-users-fetched", "fractional-users-fetched", "null-usage", "null-citations", "no-message", "extra-key", "not-json"],
)
async def test_a_reply_breaking_the_contract_is_retried_once_then_named(xai, body, named):
    url, posts = xai(body, body)

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert result.is_error
    assert named in text_of(result)
    assert len(posts) == 2


async def test_an_http_error_is_loud_and_never_retried(xai):
    url, posts = xai(requests.HTTPError("503 service unavailable"))

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert result.is_error
    assert "503" in text_of(result)
    assert len(posts) == 1


async def test_a_missing_key_fails_before_any_request_and_other_tools_still_work(xai):
    url, posts = xai(key=None)

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})
    quote = await call_tool(url, "get_quote", {"ticker": "BBCA.JK"})

    assert result.is_error
    assert "SMM_XAI_API_KEY" in text_of(result)
    assert posts == []
    assert not quote.is_error
