# Model Monitor - main branch
#
# Multi-stage build: install deps in the builder stage, copy only what is
# needed into the final image to keep it lean.

FROM python:3.11-slim AS builder

WORKDIR /build

# Build deps for numpy/scikit-learn C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Install runtime deps only (no dev/test tools)
RUN pip install --no-cache-dir --prefix=/install .


FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Application source and static assets
COPY src/ src/
COPY data/reference/ data/reference/

# Runtime directories - populated via bind mounts (see docker-compose.yml)
RUN mkdir -p data/metrics models/archive

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "model_monitor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
