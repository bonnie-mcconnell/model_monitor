# Model Monitor
#
# Multi-stage build: install deps in builder, copy only what is needed
# into the final image to keep the image lean.
#
# The base image is slim but includes the C extensions scikit-learn needs.
# sentence-transformers downloads ~90MB on first run; the model cache is
# kept outside the image so it survives rebuilds.

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps for numpy / scikit-learn C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Install the package + runtime deps (no dev tools)
RUN pip install --no-cache-dir --prefix=/install ".[llm]"


FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY data/reference/ data/reference/
COPY config/ config/ 2>/dev/null || true
COPY src/model_monitor/config/ src/model_monitor/config/
COPY contracts/ contracts/ 2>/dev/null || true

# Runtime directories - populated by volumes or make train/sim
RUN mkdir -p data/metrics models/archive

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "model_monitor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

