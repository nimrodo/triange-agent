FROM python:3.13-slim-bookworm

# curl+unzip: required by `reflex init` to fetch its bundled Bun toolchain
# for the frontend build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY . .
RUN uv sync --frozen --no-dev

RUN uv run reflex init

ENV CHAT_PROVIDER=openai
ENV EMBEDDING_PROVIDER=openai
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 3000 8000

CMD ["reflex", "run", "--env", "prod", "--backend-host", "0.0.0.0"]
