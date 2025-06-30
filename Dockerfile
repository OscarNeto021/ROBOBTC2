FROM python:3.11-slim-bookworm AS builder
WORKDIR /app
ENV POETRY_VERSION=1.8.2
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:$PATH"
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-root

FROM python:3.11-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY src ./src
COPY config ./config
COPY trading_api ./trading_api
CMD ["python", "trading_api/src/main.py"]
