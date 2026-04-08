"""FastAPI application entry point and lifespan setup."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from model_monitor.api import dashboard, health
from model_monitor.api.startup import start_background_loops


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_background_loops()
    yield


app = FastAPI(title="Model Monitor", lifespan=lifespan)

app.include_router(health.router)
app.include_router(dashboard.router)