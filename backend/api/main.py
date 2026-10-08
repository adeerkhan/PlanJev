"""FastAPI application for SpatialDecide Engine."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from engine.decision_engine import get_engine
from engine.schemas import (
    DecisionRequest,
    DecisionResponse,
    ValidationRequest,
    ValidationResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model. Shutdown: cleanup."""
    logger.info("Starting SpatialDecide Engine...")
    engine = get_engine()
    # Pre-load the model on startup
    try:
        engine._model.load()
        logger.info("Model pre-loaded successfully")
    except Exception as exc:
        logger.warning("Model pre-load failed (will retry on first request): %s", exc)
    yield
    logger.info("Shutting down SpatialDecide Engine...")


# ── App creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="SpatialDecide Engine",
    description="Inference & Decision Core for spatial layout scoring and validation",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict[str, str | bool]:
    """Health check endpoint."""
    engine = get_engine()
    return {
        "status": "ok",
        "model_loaded": engine._model.is_loaded,
        "model_id": engine._model.model_id,
    }


# ── Decision endpoints ────────────────────────────────────────────────────────

@app.post("/api/v1/score", response_model=DecisionResponse)
async def score_layouts(request: DecisionRequest) -> DecisionResponse:
    """Score and rank layout candidates."""
    if not request.candidates:
        raise HTTPException(status_code=400, detail="At least one candidate is required")
    engine = get_engine()
    return engine.score_candidates(request)


@app.post("/api/v1/validate", response_model=ValidationResponse)
async def validate_layout(request: ValidationRequest) -> ValidationResponse:
    """Validate a single layout candidate."""
    engine = get_engine()
    return engine.validate_layout(request)


# ── Model info ───────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models() -> dict:
    """List available models (OpenAI-compatible endpoint)."""
    engine = get_engine()
    model = engine._model
    return {
        "object": "list",
        "data": [
            {
                "id": model.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "convaiinnovations",
            }
        ],
    }


@app.get("/api/v1/model/info")
async def model_info() -> dict[str, str | bool | list[str]]:
    """Get information about the loaded model."""
    engine = get_engine()
    model = engine._model
    info: dict[str, str | bool | list[str]] = {
        "model_id": model.model_id,
        "loaded": model.is_loaded,
    }
    if model.is_loaded and model._model is not None:
        config = model._model.config
        info["labels"] = list(config.id2label.values()) if config.id2label else []
        info["num_labels"] = config.num_labels
    return info


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
