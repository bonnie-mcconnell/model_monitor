from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from model_monitor.api import dashboard, health
from model_monitor.monitoring.aggregation import start_aggregation_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(start_aggregation_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    title="Model Monitor",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(dashboard.router)
