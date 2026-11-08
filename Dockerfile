FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
ENV PORT=8080
CMD ["uv", "run", "--frozen", "--no-sync", "app.py"]
