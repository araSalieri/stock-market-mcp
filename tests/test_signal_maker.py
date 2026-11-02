"""Tests for SignalMaker: step order, the Signal cache and the Trade Plan, called directly with fakes and a SQLite file."""

from fakes import FakeClassifier, FakeFetcher, flat_history


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


def test_second_signal_within_the_ttl_is_served_from_the_cache(signal_maker):
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier)

    first = maker.make("AAPL")
    second = maker.make("AAPL")

    assert second == first
    assert len(classifier.prompts) == 1  # the stored Signal doesn't change the cache key
