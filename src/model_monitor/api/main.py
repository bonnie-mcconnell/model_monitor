from __future__ import annotations

from fastapi import FastAPI

from model_monitor.api import dashboard, health
from model_monitor.api.startup import start_background_aggregation_loop

app = FastAPI(title="Model Monitor")

start_background_aggregation_loop()

app.include_router(health.router)
app.include_router(dashboard.router)
