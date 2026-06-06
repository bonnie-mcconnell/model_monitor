"""FastAPI application entry point and lifespan setup."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from model_monitor.api import dashboard, health, ingest, metrics
from model_monitor.api.startup import start_background_loops


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    start_background_loops()
    yield


app = FastAPI(
    title="Model Monitor",
    description=(
        "Production ML monitoring: PSI drift detection, trust score, "
        "automated retraining and rollback. "
        "POST /metrics/ingest to push batch results from your inference pipeline."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(dashboard.router)
app.include_router(ingest.router)
app.include_router(metrics.router)
