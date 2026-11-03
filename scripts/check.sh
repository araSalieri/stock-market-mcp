#!/bin/sh
# Every check CI runs; run it before committing.
set -e
uv run pytest -q
uv run pyright
if git grep -n '—' -- '*.py' '*.md'; then echo 'use en dashes (–), not em dashes'; exit 1; fi
