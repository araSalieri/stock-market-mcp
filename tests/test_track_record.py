"""Tests for get_track_record over MCP: Outcome counts and Hit Rates by Confidence band and Verdict."""

import json

import pytest

from stock_market_mcp.classifier import PROMPT_VERSION

from fakes import FakeClassifier, FakeFetcher, call_tool, seed_signals, text_of


async def _track_record(server_url, tmp_path, seeds, args=None) -> dict:
    """get_track_record reply after seeding seed_signals kwargs; the fakes fail loudly on any Yahoo call."""
    db = tmp_path / "s.db"
    for seed in seeds:
        seed_signals(db, **seed)
    classifier = FakeClassifier()
    result = await call_tool(server_url(FakeFetcher(), classifier=classifier, db_path=db), "get_track_record", args or {})
    assert not result.is_error, text_of(result)
    assert classifier.prompts == []
    return json.loads(text_of(result))


def _seed(outcomes: dict, **kwargs) -> dict:
    return {"outcomes": [outcome for outcome, n in outcomes.items() for _ in range(n)], **kwargs}


async def test_counts_signals_by_band_and_verdict_in_ascending_order(server_url, tmp_path):
    seeds = [
        _seed({"target": 2, "stop": 1, "expired": 1, None: 3}, confidence=0.8),
        _seed({"stop": 2}, verdict="sell", confidence=0.6, ticker="MSFT", interval="1h"),
        _seed({"quiet": 3, "missed": 1, None: 1}, verdict="hold", confidence=0.69),
    ]
    reply = await _track_record(server_url, tmp_path, seeds)

    empty_direction = {"target": 0, "stop": 0, "expired": 0, "pending": 0, "hit_rate": None}
    assert reply == {
        "prompt_version": PROMPT_VERSION,
        "ticker": None,
        "interval": None,
        "bands": [
            {
                "band": 0.6,
                "buy": empty_direction,
                "sell": {**empty_direction, "stop": 2},
                "hold": {"quiet": 3, "missed": 1, "pending": 1, "hit_rate": None},
            },
            {
                "band": 0.8,
                "buy": {"target": 2, "stop": 1, "expired": 1, "pending": 3, "hit_rate": None},
                "sell": empty_direction,
                "hold": {"quiet": 0, "missed": 0, "pending": 0, "hit_rate": None},
            },
        ],
    }


@pytest.mark.parametrize(
    ("verdict", "hit", "miss"), [("buy", "target", "stop"), ("sell", "target", "stop"), ("hold", "quiet", "missed")]
)
async def test_hit_rate_is_null_at_9_scored_and_set_at_10_with_expired_and_pending_left_out(
    server_url, tmp_path, verdict, hit, miss
):
    seeds = [
        _seed({hit: 3, miss: 6, "expired": 5, None: 5}, verdict=verdict, confidence=0.5),
        _seed({hit: 4, miss: 6, "expired": 5, None: 5}, verdict=verdict, confidence=0.7),
    ]
    low, high = (await _track_record(server_url, tmp_path, seeds))["bands"]

    assert low[verdict]["hit_rate"] is None
    assert high[verdict]["hit_rate"] == pytest.approx(0.4)
