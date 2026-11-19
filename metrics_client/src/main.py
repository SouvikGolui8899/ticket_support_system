from fastapi import FastAPI, Response

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.metrics import update_metrics
from src.schemas import PipelineResult


VERSION = "1.0.0"


app = FastAPI(
    title="Prometheus Client",
    description=("Prometheus Client"),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.post("/log_metrics")
async def log_metrics(result: PipelineResult) -> None:
    update_metrics(result)


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
