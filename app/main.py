from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import numpy as np

from app.models import PredictionRequest, PredictionResponse, HealthResponse
from app.ml_model import model_holder
from app.utils import REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_DISTRIBUTION, MODEL_VERSION, log_request_middleware, prometheus_metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model at startup"""
    model_holder.load()
    MODEL_VERSION.set(1)
    yield
    print("Shutting down...")

app = FastAPI(title="ML Inference Service", version="1.0.0", lifespan=lifespan)
app.middleware("http")(log_request_middleware)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    REQUEST_COUNT.labels(status='success', endpoint='health').inc()
    return HealthResponse(status="healthy", model_loaded=model_holder.model is not None, model_version=model_holder.version)

@app.get("/ready", response_model=HealthResponse)
async def ready_check():
    if model_holder.model is None:
        REQUEST_COUNT.labels(status='error', endpoint='ready').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    REQUEST_COUNT.labels(status='success', endpoint='ready').inc()
    return HealthResponse(status="ready", model_loaded=True, model_version=model_holder.version)

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    with REQUEST_LATENCY.time():
        if model_holder.model is None:
            REQUEST_COUNT.labels(status='error', endpoint='predict').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        features = np.array(req.features).reshape(1, -1)
        pred = int(model_holder.predict(features)[0])
        try:
            prob = float(max(model_holder.predict_proba(features)[0]))
            PREDICTION_DISTRIBUTION.observe(prob)
        except TypeError:
            prob = 1.0
        REQUEST_COUNT.labels(status='success', endpoint='predict').inc()
        return PredictionResponse(prediction=pred, confidence=prob, model_version=model_holder.version, request_id=req.request_id)

@app.get("/metrics")
async def metrics():
    return prometheus_metrics()

@app.get("/")
async def root():
    return {"service":"ML Inference Service","version":"1.0.0","endpoints":["/predict","/health","/ready","/metrics","/docs"]}
