# Coding standards

Judgement calls for review. CI already enforces tests, types and dashes.

- **Loud errors.** Bad input, unknown tickers and missing data raise a `ToolError` naming the cause; a tool never returns empty or made-up data. In a multi-ticker tool the error is per ticker and the rest still return.
- **Null, not NaN.** Any number that can be non-finite goes through the finite check and comes back as `null`.
- **One source per limit.** A domain limit (ticker cap, period limits, TTLs) is one constant or table, used everywhere it applies; docstrings name it rather than restate the value.
- **Test through the MCP surface.** Tests start the server via the `server_url` fixture with fakes from `tests/fakes.py` and call tools. Reaching into a module directly needs a reason no tool can give, written in the test.
- **Injectable seams.** Anything a test must control (fetcher, classifier, TTLs, file paths, clock) is a `build_server` keyword argument, not a monkeypatch.
- **Glossary terms.** Code, tool descriptions and docs use `GLOSSARY.md` terms and avoid each term's _Avoid_ list; a new domain concept gets a glossary entry in the same change.
