"""Tests for get_track_record over MCP: Outcome counts and Hit Rates by Confidence band and Verdict."""

import json

import pytest

from stock_market_mcp.classifier import PROMPT_VERSION

from fakes import FakeClassifier, FakeFetcher, call_tool, flat_history, seed_signals, text_of

from stock_market_mcp.portfolio import PortfolioStore


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
                "buy_hit_rate": None,
            },
            {
                "band": 0.8,
                "buy": {"target": 2, "stop": 1, "expired": 1, "pending": 3, "hit_rate": None},
                "sell": empty_direction,
                "hold": {"quiet": 0, "missed": 0, "pending": 0, "hit_rate": None},
                "buy_hit_rate": None,
            },
        ],
        "min_confidence": 0.6,
        "cut_loss_hit_rate": None,
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


async def test_other_prompt_versions_and_signals_that_can_never_be_scored_are_left_out(server_url, tmp_path):
    seeds = [
        _seed({"target": 1}, confidence=0.8),
        _seed({"target": 5}, confidence=0.8, prompt_version=PROMPT_VERSION - 1),
        _seed({"target": 5}, confidence=0.8, long_plan=False),
        _seed({"target": 5}, verdict="sell", confidence=0.8, short_plan=False),
        _seed({"quiet": 5}, verdict="hold", confidence=0.8, short_plan=False),
        _seed({"quiet": 5}, verdict="hold", confidence=0.8, long_plan=False),
        _seed({"target": 5}, confidence=0.3, prompt_version=PROMPT_VERSION - 1),
        # Still scored: a buy only needs the long plan, a sell the short one.
        _seed({"target": 1}, verdict="sell", confidence=0.8, long_plan=False),
        _seed({"stop": 1}, confidence=0.8, short_plan=False),
    ]
    (band,) = (await _track_record(server_url, tmp_path, seeds))["bands"]

    assert band["band"] == 0.8
    assert (band["buy"]["target"], band["buy"]["stop"]) == (1, 1)
    assert band["sell"]["target"] == 1
    assert band["hold"]["quiet"] == 0


@pytest.mark.parametrize(
    ("args", "echo", "targets"),
    [
        ({"ticker": "aapl"}, {"ticker": "AAPL", "interval": None}, 1 + 2),
        ({"interval": "1h"}, {"ticker": None, "interval": "1h"}, 2 + 4),
        ({"ticker": "aapl", "interval": "1h"}, {"ticker": "AAPL", "interval": "1h"}, 2),
        ({"ticker": "nope"}, {"ticker": "NOPE", "interval": None}, None),
    ],
)
async def test_ticker_and_interval_narrow_the_bands_and_are_echoed(server_url, tmp_path, args, echo, targets):
    seeds = [
        _seed({"target": 1}, ticker="AAPL", interval="1d"),
        _seed({"target": 2}, ticker="AAPL", interval="1h"),
        _seed({"target": 4}, ticker="MSFT", interval="1h"),
        _seed({"target": 8}, ticker="MSFT", interval="1d"),
    ]
    reply = await _track_record(server_url, tmp_path, seeds, args)

    assert {"ticker": reply["ticker"], "interval": reply["interval"]} == echo
    assert [band["buy"]["target"] for band in reply["bands"]] == ([targets] if targets else [])


async def test_a_bad_interval_is_rejected(server_url):
    result = await call_tool(server_url(FakeFetcher()), "get_track_record", {"interval": "2d"})

    assert result.is_error


async def test_buy_hit_rate_leaves_out_sells_and_is_null_at_29_scored_and_set_at_30(server_url, tmp_path):
    seeds = [
        _seed({"target": 10, "stop": 19, "expired": 3, None: 3}, confidence=0.7),
        _seed({"target": 5, "stop": 9}, verdict="sell", confidence=0.7),
        _seed({"target": 15, "stop": 15}, confidence=0.8),
        _seed({"target": 30}, verdict="sell", confidence=0.8),
        _seed({"quiet": 30}, verdict="hold", confidence=0.8),
    ]
    low, high = (await _track_record(server_url, tmp_path, seeds))["bands"]

    assert low["buy_hit_rate"] is None
    assert high["buy_hit_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("args", "buy_hit_rates"),
    [
        ({}, [pytest.approx(0.2), pytest.approx(0.7)]),
        ({"ticker": "msft"}, [pytest.approx(0.2)]),
        ({"interval": "1d"}, [pytest.approx(0.7)]),
        ({"ticker": "nope"}, []),
    ],
)
async def test_min_confidence_and_cut_loss_hit_rate_are_server_wide_while_buy_hit_rate_follows_filters(
    server_url, tmp_path, args, buy_hit_rates
):
    seeds = [
        # Band 0.6 is under break-even, so the Minimum Confidence is band 0.7.
        _seed({"target": 6, "stop": 24}, confidence=0.6, ticker="MSFT", interval="1h"),
        _seed({"target": 15, "stop": 15}, verdict="sell", confidence=0.7, cut_loss=1),
        _seed({"target": 21, "stop": 9}, confidence=0.7, cut_loss=1),
    ]
    reply = await _track_record(server_url, tmp_path, seeds, args)

    assert reply["min_confidence"] == 0.7
    assert reply["cut_loss_hit_rate"] == pytest.approx(0.6)
    assert [band["buy_hit_rate"] for band in reply["bands"]] == buy_hit_rates


async def test_min_confidence_and_cut_loss_hit_rate_match_get_signal(server_url, tmp_path):
    db = tmp_path / "s.db"
    seed_signals(db, ["target"] * 6 + ["stop"] * 24, confidence=0.6)
    seed_signals(db, ["target"] * 18 + ["stop"] * 12, confidence=0.7, cut_loss=1)
    PortfolioStore(db).set("TSLA", "long", 10, 90.0)
    url = server_url(FakeFetcher(history=flat_history()), classifier=FakeClassifier(), db_path=db)

    signal = json.loads(text_of(await call_tool(url, "get_signal", {"ticker": "TSLA", "interval": "1d"})))
    track_record = json.loads(text_of(await call_tool(url, "get_track_record", {})))

    assert track_record["min_confidence"] == signal["min_confidence"] == 0.7
    assert track_record["cut_loss_hit_rate"] == signal["position"]["cut_loss_hit_rate"] == pytest.approx(0.6)
