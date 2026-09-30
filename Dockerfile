FROM python:3.12-slim

# Copy the uv binary from its official image (pin the version you use locally)
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv

WORKDIR /app

# Dependencies first, so Docker can cache this layer when only code changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Then the engine and the packs
COPY src ./src
COPY packs ./packs
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["vardex"]
CMD ["--help"]