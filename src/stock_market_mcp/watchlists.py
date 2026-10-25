"""Watchlist storage: one JSON file mapping name to tickers (ADR-0002)."""

import json
import os
import tempfile
import threading
from pathlib import Path

DEFAULT_WATCHLIST_PATH = "watchlists.json"
# Keeps one Scan within what free Yahoo tolerates (ADR-0002).
WATCHLIST_MAX_TICKERS = 15


class WatchlistStore:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> dict[str, list[str]]:
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return {}
        try:
            lists = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Watchlist file {self.path} is not valid JSON: {exc}") from exc
        if not isinstance(lists, dict) or not all(
            isinstance(v, list) and all(isinstance(t, str) for t in v) for v in lists.values()
        ):
            raise ValueError(f"Watchlist file {self.path} is not a {{name: [tickers]}} object")
        return lists

    def set(self, name: str, tickers: list[str]) -> None:
        """Replace one Watchlist; an empty list deletes it. A bad file raises before any write."""
        with self._lock:
            lists = self.load()
            if tickers:
                lists[name] = tickers
            else:
                lists.pop(name, None)
            # Temp file in the same directory, so the rename is atomic.
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".watchlists-")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(lists, f, indent=2)
                    # On disk before the rename, so a crash can't leave an empty file in place.
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                os.unlink(tmp)
                raise
