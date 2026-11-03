"""Golden prompt test: what OpenRouter receives for a Signal is pinned to the Prompt Version."""

import hashlib
import json
from types import SimpleNamespace

from fakes import FakeFetcher, daily_bars, seed_signals

from stock_market_mcp.classifier import PROMPT_VERSION, EnforcingClassifier, OpenRouterClassifier
from stock_market_mcp.portfolio import PortfolioStore

# One fingerprint per Prompt Version. Never edit an existing one: add the new version's instead.
FINGERPRINTS = {
    3: "c9c7ffa3ffa594021d200f27cd11062ebb91eefc51cd4210ceebaea4a2e562b5",
}


def test_what_the_classifier_sees_is_pinned_to_the_prompt_version(signal_maker, tmp_path, monkeypatch):
    monkeypatch.delenv("SMM_CLASSIFIER_MODEL", raising=False)
    monkeypatch.delenv("SMM_CLASSIFIER_MAX_STATE_CHARS", raising=False)
    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    requests = []

    def fake_post(url, **kwargs):
        requests.append(kwargs["json"])
        answer = {"type": "choice", "choice": "hold", "confidence": 0.5}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"answers": {"verdict": answer}})

    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", fake_post)
    db = tmp_path / "server.db"
    # Uneven closes, so every indicator column carries values past its warmup.
    bars = daily_bars([100.0 + (i % 7) * 1.5 - i * 0.2 for i in range(60)])
    maker = signal_maker(FakeFetcher(history=bars), EnforcingClassifier(OpenRouterClassifier()), db_path=db)
    PortfolioStore(db).set("MSFT", 10, 95.0)
    seed_signals(db, ["target"], verdict="buy", confidence=0.7, ticker="MSFT")

    maker.make("AAPL", interval="1d")
    maker.make("MSFT", interval="1d")

    seen = json.dumps([{"state": r["state"], "questions": r["questions"]} for r in requests], sort_keys=True)
    fingerprint = hashlib.sha256(seen.encode()).hexdigest()
    assert FINGERPRINTS.get(PROMPT_VERSION) == fingerprint, (
        f"What the classifier sees has changed: bump the Prompt Version and add its fingerprint {fingerprint!r}."
    )
