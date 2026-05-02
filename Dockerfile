FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system nonroot \
 && adduser --system --home /home/nonroot --ingroup nonroot nonroot

ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin

WORKDIR /app


COPY pyproject.toml uv.lock /app/
RUN uv sync --locked --no-install-project

COPY src /app/src
RUN uv sync --locked


USER nonroot
ENV UV_NO_CACHE=1
ENV PYTHONPATH="${PYTHONPATH}:/app/src"
CMD ["uv", "run", "python", "src/service/main.py"]
