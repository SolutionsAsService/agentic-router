FROM ghcr.io/astral-sh/uv:0.9.28 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY sidecars/task-domain/pyproject.toml sidecars/task-domain/uv.lock ./
RUN uv sync --locked --extra qwen --no-dev
COPY sidecars/task-domain/app.py sidecars/task-domain/batching.py sidecars/task-domain/contract.py sidecars/task-domain/server.py ./
# Torch otherwise JIT-compiles Triton overrides (e.g. Qwen's rotary bmm) at request
# time, which needs a C compiler and adds per-shape compile latency.
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1
EXPOSE 8095
CMD ["/app/.venv/bin/python", "server.py"]
