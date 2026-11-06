# Coding standards

Judgement calls for review. `scripts/check.sh` (run by CI) already enforces tests, types and dashes.

- **Loud errors.** Bad input, unknown tickers and missing data raise a `ToolError` naming the cause; a tool never returns empty or made-up data. In a multi-ticker tool the error is per ticker and the rest still return.
- **Null, not NaN.** Any number that can be non-finite goes through the finite check and comes back as `null`.
- **One source per limit.** A domain limit (ticker cap, period limits, TTLs) is one constant or table, used everywhere it applies; code and docstrings name it rather than restate the value. `GLOSSARY.md` and the README may state the value, since readers need it there.
- **Test through the MCP surface.** Tests start the server via the `server_url` fixture with fakes from `tests/fakes.py` and call tools. Signal logic (Verdicts, scoring, cut_loss, the Buyback Price and Cut Stop, Minimum Confidence) is tested through the `signal_maker` fixture with `FakeFetcher` and `FakeClassifier` instead. State is seeded through the stores (`PortfolioStore`, `WatchlistStore`) or `seed_signals`. Pure modules (`prompt`, `bars`, Position pricing) take table tests on plain values. Reaching into any other module needs a reason no tool can give, written in the test or the file's docstring.
- **Prompt Version.** Anything the classifier sees changes only with a Prompt Version bump. The golden prompt test enforces it; an existing version's fingerprint is never edited, only a new one added.
- **Injectable seams.** Anything a test must control (fetcher, classifier, TTLs, file paths, clock) is a `build_server` keyword argument. Tests of the real OpenRouter transport patch `requests.post`; that is the one patched call.
- **Glossary terms.** Code, tool descriptions and docs use `GLOSSARY.md` terms and avoid each term's _Avoid_ list; a new domain concept gets a glossary entry in the same change.
- **Docs follow the surface.** A change to a tool's parameters or reply, an env var, or the compose setup updates the tool description, README and compose comments in the same change.
- **No private library APIs.** A `pyright: ignore[reportPrivateUsage]` needs a `shortcut:` comment saying why no public API works.
- **Comments give the why.** A comment says what the code can't show, not what the next line does. No issue or spec numbers in code; ADR references and `shortcut:` comments are fine.
