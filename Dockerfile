# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage build
#   builder : compiles/installs dependencies into a virtualenv
#   test    : builder + dev deps + tests  -> `docker build --target test .`
#   runtime : slim image with ONLY the venv + app code, running as non-root
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv
WORKDIR /build
# Copy ONLY the dependency manifest first: this layer stays cached until
# requirements.txt changes, so code edits don't trigger a full reinstall.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

FROM builder AS test
COPY requirements-dev.txt pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-dev.txt
COPY api ./api
COPY tests ./tests
RUN ruff check . && mypy && pytest -q

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"
RUN groupadd --system app && useradd --system --gid app --no-create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app api ./api
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2)" || exit 1
# Exec form (JSON array): uvicorn is PID 1 and receives SIGTERM directly,
# so `docker stop` triggers a graceful shutdown (lifespan teardown runs).
CMD ["uvicorn", "api.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--no-access-log", "--timeout-graceful-shutdown", "20"]
