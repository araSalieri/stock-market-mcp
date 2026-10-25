# Multi-stage build: uv sync --frozen from the lockfile, runtime ships only
# the venv (dev dependency group excluded). python:3.12 is the hard constraint
# for pandas-ta/numba (they do not build on 3.14).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src

# --no-editable: the project lands in the venv's site-packages, so the runtime
# stage needs only the copied venv. --no-dev drops pytest/pyright from runtime.
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim

# Healthcheck uses a stdlib TCP connect; no curl/wget installed.
# The server binds SMM_HOST (0.0.0.0 in compose) on SMM_PORT (default 8000).
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import socket; socket.create_connection(('127.0.0.1', int(__import__('os').environ.get('SMM_PORT', '8000'))), timeout=4).close()"]

# Non-root runtime user with a writable home (yfinance caches there).
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser
# Owned by appuser, so a fresh named volume mounted here starts writable.
RUN mkdir /home/appuser/data

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

ENTRYPOINT ["stock-market-mcp"]
