"""Scoring Sentiments: bullish ones judged like a buy, bearish ones like its mirror, and the Hit Rate in each reply."""

import json
from datetime import UTC, datetime

import pytest
from fakes import FakeFetcher, FakeSentimentReader, call_tool, flat_history, seed_sentiments, seed_signals, text_of

NOW = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)


async def _hit_rate(server_url, tmp_path, *seeds, reading="bullish") -> float | None:
    """hit_rate in a get_sentiment reply with this reading, after seeding (outcomes, seed_sentiments kwargs) pairs."""
    for outcomes, kwargs in seeds:
        seed_sentiments(tmp_path / "server.db", outcomes, **kwargs)
    reader = FakeSentimentReader(reply={**FakeSentimentReader().reply, "reading": reading})
    url = server_url(FakeFetcher(fundamentals={}, history=flat_history()), sentiment_reader=reader, now=lambda: NOW)
    result = await call_tool(url, "get_sentiment", {"ticker": "AAPL"})
    assert not result.is_error, text_of(result)
    return json.loads(text_of(result))["hit_rate"]


async def test_hit_rate_is_targets_over_targets_and_stops(server_url, tmp_path):
    seeds = (["target"] * 3 + ["stop"] * 7 + ["expired"] * 4, {})
    assert await _hit_rate(server_url, tmp_path, seeds) == pytest.approx(3 / 10)


async def test_hit_rate_is_null_below_the_minimum_scored(server_url, tmp_path):
    assert await _hit_rate(server_url, tmp_path, (["target"] * 9 + ["expired"] * 5, {})) is None


@pytest.mark.parametrize(
    "seeds",
    [(["target"] * 10, {"version": 0}), (["target"] * 10, {"reading": "bearish"}), (["target"] * 10, {"interval": "5m"})],
    ids=["older-version", "other-reading", "other-interval"],
)
async def test_hit_rate_counts_only_the_current_version_reading_and_interval(server_url, tmp_path, seeds):
    assert await _hit_rate(server_url, tmp_path, (["target"] * 9, {}), seeds) is None


def _later(*bars):
    """The 60 flat bars the first Sentiment is made on, then later bars given as (high, low)."""
    df = flat_history(rows=60 + len(bars))
    for i, (high, low) in enumerate(bars):
        df.loc[df.index[60 + i], ["High", "Low"]] = [high, low]
    return df


FLAT = (101.0, 99.0)


async def _scored(server_url, tmp_path, reading, *histories, **server_kwargs) -> float | None:
    """hit_rate in the last of one get_sentiment call per history, with 5 Targets and 4 Stops seeded for reading:
    0.6 once one more Target is scored, 0.5 once one more Stop is, null while none is.

    Entry 100 on 60 flat bars, ATR 2: a bullish Stop 96 / Target 108, a bearish Stop 104 / Target 92.
    """
    seed_sentiments(tmp_path / "server.db", ["target"] * 5 + ["stop"] * 4, reading=reading)
    reader = FakeSentimentReader(reply={**FakeSentimentReader().reply, "reading": reading})
    fetcher = FakeFetcher(fundamentals={}, history=list(histories))
    url = server_url(fetcher, sentiment_reader=reader, history_ttl=0, now=lambda: NOW, **server_kwargs)
    results = [await call_tool(url, "get_sentiment", {"ticker": "AAPL"}) for _ in histories]
    assert not results[-1].is_error, text_of(results[-1])
    return json.loads(text_of(results[-1]))["hit_rate"]


@pytest.mark.parametrize(
    ("reading", "bars", "hit_rate"),
    [
        ("bullish", (FLAT, (109.0, 99.0), (101.0, 90.0)), 0.6),
        ("bullish", (FLAT, (101.0, 95.0), (109.0, 99.0)), 0.5),
        ("bullish", (FLAT, (109.0, 95.0)), 0.5),
        ("bullish", (*[FLAT] * 20, (109.0, 99.0)), None),
        ("bullish", (*[FLAT] * 19,), None),
        ("bearish", (FLAT, (101.0, 91.0), (105.0, 99.0)), 0.6),
        ("bearish", (FLAT, (105.0, 99.0), (101.0, 91.0)), 0.5),
        ("bearish", (FLAT, (105.0, 91.0)), 0.5),
        ("bearish", (*[FLAT] * 20, (101.0, 91.0)), None),
        ("bearish", (*[FLAT] * 19,), None),
    ],
    ids=[f"{r}-{o}" for r in ("bullish", "bearish") for o in ("target", "stop", "same-bar-stop", "expired", "pending")],
)
async def test_a_sentiment_is_scored_like_a_buy_or_its_mirror(server_url, tmp_path, reading, bars, hit_rate):
    # The second call is a cache hit: it scores the first Sentiment and logs nothing.
    assert await _scored(server_url, tmp_path, reading, flat_history(), _later(*bars)) == hit_rate


async def test_a_pending_sentiment_is_scored_on_a_later_call(server_url, tmp_path):
    assert await _scored(server_url, tmp_path, "bullish", flat_history(), _later(FLAT), _later(FLAT, (109.0, 99.0))) == 0.6


@pytest.mark.parametrize(("ttl", "hit_rate"), [(0, 1.0), (300, None)], ids=["fresh", "cached"])
async def test_each_fresh_read_logs_one_sentiment_and_a_cache_hit_logs_none(server_url, tmp_path, ttl, hit_rate):
    # Two calls on the same bars, then one where the price reaches the bullish Target: 8 seeded Targets plus 2 logged
    # Sentiments reach the minimum scored, 8 plus 1 don't.
    seed_sentiments(tmp_path / "server.db", ["target"] * 8)
    fetcher = FakeFetcher(fundamentals={}, history=[flat_history(), flat_history(), _later(FLAT, (109.0, 99.0))])
    url = server_url(fetcher, sentiment_reader=FakeSentimentReader(), history_ttl=0, sentiment_ttl=ttl, now=lambda: NOW)
    results = [await call_tool(url, "get_sentiment", {"ticker": "AAPL"}) for _ in range(3)]

    assert json.loads(text_of(results[-1]))["hit_rate"] == hit_rate


@pytest.mark.parametrize("reading", ["mixed", "quiet"])
async def test_mixed_and_quiet_sentiments_are_never_scored(server_url, tmp_path, reading):
    # 9 seeded Targets: one more scored Sentiment of this Reading would make a Hit Rate.
    seed_sentiments(tmp_path / "server.db", ["target"] * 9, reading=reading)
    reader = FakeSentimentReader(reply={**FakeSentimentReader().reply, "reading": reading})
    later = _later(FLAT, (109.0, 91.0), *[FLAT] * 20)
    url = server_url(FakeFetcher(fundamentals={}, history=[flat_history(), later]), sentiment_reader=reader, history_ttl=0, now=lambda: NOW)
    await call_tool(url, "get_sentiment", {"ticker": "AAPL"})
    result = await call_tool(url, "get_sentiment", {"ticker": "AAPL"})

    assert not result.is_error, text_of(result)
    assert json.loads(text_of(result))["hit_rate"] is None


async def test_a_sentiment_never_changes_a_signal_hit_rate_or_minimum_confidence(server_url, tmp_path):
    db = tmp_path / "server.db"
    seed_signals(db, ["target"] * 20 + ["stop"] * 10)
    seed_signals(db, ["target"] * 20 + ["stop"] * 10, verdict="dip")
    track_record = server_url(FakeFetcher())
    before = json.loads(text_of(await call_tool(track_record, "get_track_record", {})))

    seed_sentiments(db, ["stop"] * 40)
    seed_sentiments(db, ["target"] * 40, reading="bearish")
    await _scored(server_url, tmp_path, "bullish", flat_history(), _later(FLAT, (101.0, 95.0)))
    after = json.loads(text_of(await call_tool(track_record, "get_track_record", {})))

    # Only the sentiment section may move.
    assert after["sentiment"] != before.pop("sentiment")
    after.pop("sentiment")
    assert after == before
    assert before["bands"][0]["buy"]["hit_rate"] == pytest.approx(2 / 3)
