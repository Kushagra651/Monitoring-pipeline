"""
api/main.py — FastAPI application

Matched exactly to actual public APIs:
  api/predict.py : predict(), reload_if_stale(), force_reload(), get_model_info(), PredictionResult
  api/logger.py  : log_prediction_from_result(), flush(), shutdown(), query_logs()
  api/metrics.py : record_prediction_from_result(), record_error(), to_prometheus_text(), get_snapshot()
  api/schemas.py : PredictionInput, PredictionOutput, PredictionLog, HealthResponse
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# predict.py — module-level functions
import api.predict as predictor_module
from api.predict import (
    predict,
    reload_if_stale,
    force_reload,
    get_model_info,
    PredictionResult,
)

# logger.py — module-level functions
from api.logger import (
    log_prediction_from_result,
    flush,
    shutdown,
    query_logs,
)

# metrics.py — module-level functions
from api.metrics import (
    record_prediction_from_result,
    record_error,
    to_prometheus_text,
    get_snapshot,
)

# schemas.py — Pydantic models
from api.schemas import (
    PredictionInput,
    PredictionOutput,
    # PredictionLog,
    HealthResponse,
)

log = logging.getLogger(__name__)
_START_TIME = time.time()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting ML Monitoring API...")
    try:
        predictor_module._ensure_loaded()
        info = get_model_info()
        log.info("Model loaded: version=%s alias=%s", info.get("version"), info.get("alias"))
    except FileNotFoundError as e:
        log.warning("Model not found at startup (run training first): %s", e)
    except Exception as e:
        log.error("Unexpected startup error: %s", e)

    yield

    log.info("Shutdown — flushing prediction logs...")
    flush()
    shutdown()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Pipeline Monitor — UCI Adult Income",
    version="1.0.0",
    description="Predicts whether income >$50K. Drift detection + Prometheus monitoring.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware — latency header + hot-reload check ────────────────────────────

@app.middleware("http")
async def track_latency(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    if request.url.path == "/predict":
        reload_if_stale()   # cheap stat() check on every predict request
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Health / readiness ────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    info = get_model_info()
    return HealthResponse(
        status="ok",
        model_version=info.get("version") or "unknown",
        model_stage=info.get("alias") or "unknown",
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.get("/ready", tags=["ops"])
def ready():
    info = get_model_info()
    if info.get("status") != "loaded":
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready", "model_version": info.get("version")}


# ── Inference ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionOutput, tags=["inference"])
def predict_endpoint(req: PredictionInput):
    t_start = time.perf_counter()

    # Validated Pydantic model → plain dict for predict()
    features = req.model_dump()

    try:
        result: PredictionResult = predict(features)
    except (ValueError, RuntimeError) as e:
        record_error(kind="prediction_error")
        log.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t_start) * 1000

    # result.prediction is int 0/1 — map to label strings schema expects
    predicted_label = ">50K" if result.prediction == 1 else "<=50K"

    output = PredictionOutput(
        predicted_label=predicted_label,
        probability_above_50k=result.probability_class_1,
        model_version=result.model_version,
    )

    # Track metrics using module-level function
    record_prediction_from_result(result)

    # Log full record asynchronously (never blocks the response)
    try:
        log_prediction_from_result(
            features=features,
            result=result,
            request_id=str(output.prediction_id),
        )
    except Exception as e:
        log.warning("Prediction log enqueue failed (non-fatal): %s", e)

    return output


# ── Ops ───────────────────────────────────────────────────────────────────────

@app.post("/model/reload", tags=["ops"])
def reload_model():
    """Force hot-reload of model from registry without container restart."""
    try:
        force_reload()
        info = get_model_info()
        return {"reloaded": True, "model_version": info.get("version")}
    except Exception as e:
        log.error("Force reload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["ops"])
def model_info():
    """Metadata about the currently loaded model bundle."""
    return get_model_info()


# ── Monitoring ────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["monitoring"])
def prometheus_metrics():
    """Prometheus scrape endpoint (text/plain 0.0.4)."""
    return Response(
        content=to_prometheus_text(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/metrics/summary", tags=["monitoring"])
def metrics_summary():
    """Human-readable metrics snapshot."""
    snapshot = get_snapshot()
    return snapshot.__dict__ if hasattr(snapshot, "__dict__") else str(snapshot)


@app.get("/logs", tags=["monitoring"])
def get_logs(hours: int = 1, limit: int = 100):
    """
    Time-windowed prediction log retrieval.
    Returns last `hours` hours of predictions (default 1h).
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    try:
        df = query_logs(start=start, end=now, limit=limit)
        records = df.to_dict(orient="records") if not df.empty else []
        return {"count": len(records), "logs": records}
    except Exception as e:
        log.error("query_logs failed: %s", e)
        return {"count": 0, "logs": [], "error": str(e)}