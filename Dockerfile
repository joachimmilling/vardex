FROM python:3.12-slim

# Copy the uv binary from its official image (pin the version you use locally)
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

WORKDIR /app

# Dependencies first, so Docker can cache this layer when only code changes. The image
# carries every connector but the Mac-only trainer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --extra anthropic --extra openai \
    --extra duckdb --extra postgresql

# Then the engine and the packs
COPY src ./src
COPY packs ./packs
RUN uv sync --locked --no-dev --extra anthropic --extra openai --extra duckdb --extra postgresql

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["vardex"]
CMD ["--help"]