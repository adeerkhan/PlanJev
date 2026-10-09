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
| POST | `/api/v1/decide` | Batched probabilities for every candidate (one forward pass) |

### LitServe (port 8001)

| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Single prompt inference |
| `POST /predict` (batch) | Batch inference with `prompts` field |

## Model

Two interchangeable backends behind the same typed-question API (state +
questions in, calibrated probabilities out). Select with `DECISION_MODEL`
in `.env` — default `laya`:

| `DECISION_MODEL` | Model | Params | State budget | Device |
|---|---|---|---|---|
| `laya` (default) | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | 421M | 512 tokens | CPU or CUDA |
| `decider-2b` | [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) | 2B | 32k tokens | **CUDA (bf16)** by default, CPU fallback |

- **laya** — the original production path; ModernBERT-large backbone,
  Apache 2.0, lightest of the two.
- **decider-2b** — an open reproduction of the same System One model class
  on Qwen3.5-2B: same answer shapes (noul / score+legend / choice with
  probabilities), 64× the state budget, and CUDA shape-bucketed graphs so a
  full portfolio is scored in one batched pass. The `decider` inference
  package ships inside the model repo and is loaded automatically.

Weights come from `MODEL_CACHE_DIR` (default `.model_cache`), so a populated
cache means no re-download.

### CUDA

`decider-2b` needs a CUDA torch build (the cu128+ index for RTX 40/50-class
GPUs) — CPU torch works but is ~100× slower per batch:

```bash
uv pip install --python jev/bin/python torch==2.14.1+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

`flash-linear-attention` (pulled in with the cu130 wheel's triton) supplies
the fused Qwen3.5 kernels; without them the model runs but several times
slower.

### Measured on an 8 GB laptop GPU (RTX 5060, shared with the desktop)

| candidates per request | latency (warm) | VRAM peak |
|---|---|---|
| 2 | ~430 ms | ~4.4 GB |
| 4 | ~900 ms | ~5.0 GB |
| 5+ | OOM → HTTP 503; the solve keeps its deterministic ranking | — |

The model itself (bf16 weights + CUDA-graph pools) is ~5.8 GB, so the batch
is the only free variable: the solver client therefore sends only the
portfolio's **top 4** plans (`DecisionModelConfig.maxCandidates` — raise it
on a bigger card). An OOM releases the caching allocator before the error
propagates, so one oversized request never poisons the process. In-process
fp8 conversion is not an option on an 8 GB card: quantising needs 2x the
weights in VRAM.

### A/B the backends

```bash
MODEL_CACHE_DIR=.model_cache ./jev/bin/python scripts/ab_models.py
```

Loads each backend in turn, scores the same four layout candidates with the
same questions, and prints load time, batched latency, and the per-candidate
probabilities side by side.

## Development

```bash
# Run tests
pytest tests/ -v

# Install dev dependencies
uv pip install -e ".[dev]"
```
