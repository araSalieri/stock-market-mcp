"""Classifier tests: the Decisions answer mapping, retries and provider failures on the classifier module, plus get_signal tests of the classifier behind the tool (retries, provider failure, data_date from the latest bar)."""

import json
from datetime import UTC, datetime

import pytest
from fakes import (
    FakeClassifier,
    FakeFetcher,
    call_tool,
    daily_bars,
    fake_history,
    five_minute_bars,
    intraday_history,
    text_of,
)

from stock_market_mcp.classifier import (
    EnforcingClassifier,
    OpenRouterClassifier,
    SignalContractError,
    classifier_model_id,
)
from stock_market_mcp.portfolio import PortfolioStore


def test_classifier_model_id_default_and_env_override(monkeypatch):
    monkeypatch.delenv("SMM_CLASSIFIER_MODEL", raising=False)
    assert classifier_model_id() == "typesafe/jev-1.13"

    monkeypatch.setenv("SMM_CLASSIFIER_MODEL", "acme/other-2")
    assert classifier_model_id() == "acme/other-2"


async def test_malformed_reply_retried_once_then_loud_error(server_url):
    # Both attempts violate the contract; exactly one retry, then a loud tool
    # error naming the violation. Agents never see a fabricated success.
    bad_verdict = {"verdict": "maybe", "confidence": 0.7,
                   "rationale": "uptrend"}
    missing_key = {"verdict": "hold", "confidence": 0.5}
    classifier = FakeClassifier(replies=[bad_verdict, missing_key])
    result = await call_tool(
        server_url(fetcher := FakeFetcher(history=fake_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert result.is_error
    message = text_of(result)
    assert "verdict" in message  # names the violation
    assert len(classifier.prompts) == 2  # exactly one retry
    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False)]


async def test_single_formatting_slip_recovers_after_one_retry(server_url):
    # One malformed reply, then a valid one: the call succeeds on the retry.
    bad = {"verdict": "maybe", "confidence": 0.7,
           "rationale": "uptrend"}
    good = {"verdict": "hold", "confidence": 0.9,
            "rationale": "overbought"}
    classifier = FakeClassifier(replies=[bad, good])
    result = await call_tool(
        server_url(FakeFetcher(history=fake_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert not result.is_error
    payload = json.loads(text_of(result))
    assert payload.pop("trade_plan") is None
    payload.pop("price")
    assert payload.pop("hit_rate") is None
    assert payload.pop("min_confidence") == 0.6
    assert payload == {**good, "data_date": "2024-01-03T11:00:00"}
    assert len(classifier.prompts) == 2  # the retry happened, not a third call


async def test_cut_on_a_ticker_not_held_retried_once_then_loud(server_url):
    cut = {"verdict": "cut", "confidence": 0.9, "rationale": "breakdown"}
    classifier = FakeClassifier(replies=[cut, cut])
    result = await call_tool(
        server_url(FakeFetcher(history=fake_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert result.is_error
    assert "'cut'" in text_of(result)
    assert len(classifier.prompts) == 2


async def test_cut_on_a_held_position_is_accepted(server_url, tmp_path):
    PortfolioStore(tmp_path / "server.db").set("AAPL", 10, 90.0)
    cut = {"verdict": "cut", "confidence": 0.9, "rationale": "breakdown"}
    classifier = FakeClassifier(replies=[cut])
    result = await call_tool(
        server_url(FakeFetcher(history=fake_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert not result.is_error, text_of(result)
    assert json.loads(text_of(result))["verdict"] == "cut"
    assert len(classifier.prompts) == 1


async def test_provider_failure_is_loud_and_never_retried(server_url):
    # Transport errors are not formatting slips: loud immediately, no retry call.
    classifier = FakeClassifier(error=RuntimeError("openrouter timeout"))
    result = await call_tool(
        server_url(FakeFetcher(history=fake_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert result.is_error
    assert "openrouter timeout" in text_of(result)
    assert len(classifier.prompts) == 1


class FakeResponse:
    """Minimal requests.Response stub: raise_for_status no-op, json returns the body.

    Used only by the Decisions-transport tests below; the fake_post in
    _real_transport_with_bodies raises before a body is needed for HTTP errors.
    """

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


PROMPT = json.dumps(
    {"ticker": "AAPL", "period": "1y", "rows": [{"Date": "2024-01-03T00:00:00"}]}
)


def _decisions_body(answer):
    return {"answers": {"verdict": answer}, "usage": {"cost": 0.001}, "model": "typesafe/jev-1.13"}


def _choice_answer(
    choice="hold",
    probabilities=None,
    confidence: float | None = 0.9,
):
    answer = {
        "type": "choice",
        "choice": choice,
        "probabilities": probabilities
        if probabilities is not None
        else {"buy": 0.3, "hold": 0.7},
    }
    if confidence is not None:
        answer["confidence"] = confidence
    return answer


def _real_transport_with_bodies(monkeypatch, bodies):
    """Real OpenRouterClassifier transport with requests.post serving bodies in order."""
    posts = []

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return FakeResponse(bodies.pop(0))

    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", fake_post)
    return EnforcingClassifier(OpenRouterClassifier()), posts


def test_decisions_answer_maps_to_strict_signal(monkeypatch):
    # The Decisions answer becomes the settled verdict dict: confidence straight
    # from the answer, rationale citing the probabilities.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch, [_decisions_body(_choice_answer())]
    )

    verdict = classifier.classify(PROMPT, held=False)

    assert verdict == {
        "verdict": "hold",
        "confidence": 0.9,
        "rationale": "Decisions API chose hold (probabilities: buy=0.3, hold=0.7)",
    }
    assert len(posts) == 1
    url, kwargs = posts[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    request = kwargs["json"]
    assert request["model"] == classifier_model_id()
    assert request["state"] == PROMPT
    question = request["questions"]["verdict"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == ["buy", "hold"]
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_a_prompt_with_a_position_is_asked_buy_hold_or_cut(monkeypatch):
    prompt = json.dumps({**json.loads(PROMPT), "position": {"avg_cost": 90.0, "unrealised_pnl_pct": 5.0}})
    classifier, posts = _real_transport_with_bodies(
        monkeypatch, [_decisions_body(_choice_answer(choice="cut", probabilities={"buy": 0.1, "hold": 0.2, "cut": 0.7}))]
    )

    assert classifier.classify(prompt, held=True)["verdict"] == "cut"
    assert list(posts[0][1]["json"]["questions"]["verdict"]["criteria"]) == ["buy", "hold", "cut"]


def test_missing_confidence_falls_back_to_chosen_probability(monkeypatch):
    # confidence and probabilities are schema-optional; with confidence absent
    # the chosen option's probability stands in for it.
    classifier, _ = _real_transport_with_bodies(
        monkeypatch,
        [_decisions_body(_choice_answer(confidence=None))],
    )

    verdict = classifier.classify(PROMPT, held=False)

    assert verdict["verdict"] == "hold"
    assert verdict["confidence"] == 0.7


def test_missing_verdict_answer_retried_once_then_valid(monkeypatch):
    # A reply without answers.verdict is a contract violation like bad JSON:
    # exactly one retry, and the retry's valid answer succeeds.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [
            {"answers": {}, "usage": {}, "model": "typesafe/jev-1.13"},
            _decisions_body(_choice_answer()),
        ],
    )

    assert classifier.classify(PROMPT, held=False)["verdict"] == "hold"
    assert len(posts) == 2  # exactly one retry


def test_unexpected_answer_type_retried_once_then_loud(monkeypatch):
    # Per the Decisions contract the answer's type field is checked before
    # reading; an unexpected type is an error, never a default.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [
            _decisions_body({"type": "text", "text": "I think hold"}),
            _decisions_body({"type": "text", "text": "I still think hold"}),
        ],
    )

    with pytest.raises(SignalContractError) as exc_info:
        classifier.classify(PROMPT, held=False)

    assert "twice" in str(exc_info.value)
    assert "type" in str(exc_info.value)
    assert len(posts) == 2


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "maybe", "confidence": 0.9},
        {"type": "choice", "choice": "maybe", "probabilities": {"buy": 0.3, "hold": 0.7}},
        {"type": "choice", "choice": "maybe"},
        {"type": "choice", "choice": ["maybe"], "probabilities": {"buy": 0.3, "hold": 0.7}},
    ],
    ids=["confidence", "probabilities-only", "neither", "non-string"],
)
def test_choice_outside_buy_hold_retried_once_then_loud(monkeypatch, answer):
    # Called directly: a malformed Decisions reply is a transport detail no fake classifier behind a tool can send.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch, [_decisions_body(answer), _decisions_body(answer)]
    )

    with pytest.raises(SignalContractError) as exc_info:
        classifier.classify(PROMPT, held=False)

    assert "twice" in str(exc_info.value)
    assert "maybe" in str(exc_info.value)
    assert len(posts) == 2


def test_missing_confidence_and_probability_is_a_contract_error(monkeypatch):
    # With neither confidence nor a probability for the choice there is no
    # schema-valid confidence: a contract violation, retried once then loud.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [
            _decisions_body(_choice_answer(confidence=None, probabilities={})),
            _decisions_body(_choice_answer(confidence=None, probabilities={})),
        ],
    )

    with pytest.raises(SignalContractError) as exc_info:
        classifier.classify(PROMPT, held=False)

    assert "twice" in str(exc_info.value)
    assert len(posts) == 2


def test_provider_http_error_is_loud_and_never_retried(monkeypatch):
    # Non-200 responses are provider failures (#12 semantics): loud HTTPError,
    # never retried, never wrapped as a contract violation.
    import requests

    classifier, posts = _real_transport_with_bodies(monkeypatch, [])

    def failing_post(url, **kwargs):
        posts.append((url, kwargs))
        raise requests.HTTPError("503 service unavailable")

    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", failing_post)

    with pytest.raises(requests.HTTPError):
        classifier.classify(PROMPT, held=False)

    assert len(posts) == 1


async def test_intraday_data_date_is_the_latest_bars_timestamp(server_url, monkeypatch):
    # Real transport: data_date comes from the latest bar. The fake's 60th
    # 5m bar opens 2024-03-01 13:55 WIB, given in the exchange's offset as scan does.
    posts = []

    def fake_post(url, **kwargs):
        posts.append(kwargs)
        return FakeResponse(_decisions_body(_choice_answer(choice="hold")))

    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", fake_post)
    result = await call_tool(
        server_url(FakeFetcher(history=intraday_history())),
        "get_signal",
        {"ticker": "BBCA.JK", "interval": "5m"},
    )

    assert not result.is_error, text_of(result)
    data_date = json.loads(text_of(result))["data_date"]
    assert data_date == "2024-03-01T13:55:00+07:00"


async def test_daily_data_date_matches_scans_for_the_same_last_bar(server_url, monkeypatch):
    # Real transport: data_date comes from the latest bar; the 5m bars stop the day
    # before, so scan's data_date is the daily bar's too.
    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "stock_market_mcp.classifier.requests.post",
        lambda url, **kwargs: FakeResponse(_decisions_body(_choice_answer(choice="hold"))),
    )
    daily = daily_bars([100.0 + i for i in range(60)], last="2024-03-15")
    intraday = five_minute_bars({"2024-03-14 16:10": (158.0, 1000)}, [], tz="Asia/Jakarta")
    fetcher = FakeFetcher(history={"BBCA.JK": {"1d": daily, "5m": intraday}})
    # 2024-03-15 10:00 WIB, during the IDX session.
    url = server_url(fetcher, now=lambda: datetime(2024, 3, 15, 3, 0, tzinfo=UTC))

    signal = await call_tool(url, "get_signal", {"ticker": "BBCA.JK", "interval": "1d"})
    scan = await call_tool(url, "scan", {"tickers": ["BBCA.JK"]})

    assert not signal.is_error, text_of(signal)
    assert scan.structured_content is not None
    assert json.loads(text_of(signal))["data_date"] == scan.structured_content["result"][0]["data_date"]
    assert json.loads(text_of(signal))["data_date"] == "2024-03-15T00:00:00+07:00"
