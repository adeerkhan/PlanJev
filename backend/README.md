# SpatialDecide Engine - Backend

Inference & Decision Core for spatial layout scoring and validation.

## Architecture

```
backend/
├── api/
│   ├── main.py              # FastAPI application (port 8000)
│   └── litserve_api.py      # LitServe API for model serving (port 8001)
├── engine/
│   ├── model_loader.py      # Laya model wrapper
│   ├── decision_engine.py   # Core scoring & validation logic
│   └── schemas.py           # Pydantic request/response models
├── tests/
│   └── test_decision_engine.py
├── jev/                     # uv virtual environment
├── pyproject.toml
└── requirements.txt
```

## Quick Start

```bash
# Activate the uv environment
source jev/bin/activate

# Run FastAPI server (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run LitServe server (port 8001)
python -m api.litserve_api
```

## API Endpoints

### FastAPI (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check & model status |
| GET | `/api/v1/model/info` | Model information |
| POST | `/api/v1/score` | Score & rank layout candidates |
| POST | `/api/v1/validate` | Validate a single layout |

### LitServe (port 8001)

| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Single prompt inference |
| `POST /predict` (batch) | Batch inference with `prompts` field |

## Model

- **Model:** [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- **Type:** Text-classification (calibrated decision model)
- **Parameters:** 421M (ModernBERT-large backbone)
- **Context:** 512 tokens
- **License:** Apache 2.0

## Development

```bash
# Run tests
pytest tests/ -v

# Install dev dependencies
uv pip install -e ".[dev]"
```
