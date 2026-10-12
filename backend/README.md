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
source .venv/bin/activate

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
uv pip install --python .venv/bin/python torch==2.14.1+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

`flash-linear-attention` (pulled in with the cu130 wheel's triton) supplies
the fused Qwen3.5 kernels; without them the model runs but several times
slower.

### VRAM: the batch is a hard limit

The batch is one GPU forward pass, and the card is shared with the desktop,
so the number of candidates a request may carry is a property of the
machine. `DECISION_MAX_CANDIDATES` (default **2**, measured on an 8 GB
laptop card) is the limit; `GET /health` reports it as `max_candidates`,
and the solver client asks for it and caps its batch automatically.

A request above the limit is refused with **413** and a message naming the
limit — before the GPU is touched. That is deliberate: a batch that fails
mid-forward reserves CUDA-graph pools, which makes even a small batch fail
afterwards. If a batch that was within the limit still runs out (the desktop
took memory back), the limit lowers itself and the allocator's blocks are
reclaimed, so the next request succeeds.

Raise it on a dedicated GPU:

    DECISION_MAX_CANDIDATES=8

```
$ curl -s localhost:8000/health
{"status":"ok","model_loaded":true,"model_id":"Mapika/decider-2b","max_candidates":2}
```

#### One-candidate requests are pinned to the single-state path

Guidance traffic (the solver asking which room to place next) sends exactly
**one** candidate per request, with the options inside the question — and the
question text changes on every call, because it states the residual free area
and the room ids on offer. The schema path used to compile a fresh schema and
capture fresh CUDA graphs per distinct question set, so a corpus run's hundreds
of guidance requests grew the graph pool until even one-candidate requests
503'd, and the service only recovered on a restart. Requests carrying a single
candidate now score through the engine's shape-keyed graphs (one bounded set,
shared by every request) instead. Multi-candidate batches — the portfolio
re-rank, whose questions are fixed — keep the one-pass batched path.

### A/B the backends

```bash
MODEL_CACHE_DIR=.model_cache ./.venv/bin/python scripts/ab_models.py
```

Loads each backend in turn, scores the same four layout candidates with the
same questions, and prints load time, batched latency, and the per-candidate
probabilities side by side.

## What the model can and cannot do (measured)

The decision model is a judge, not a generator: it answers questions about
states you hand it and cannot emit coordinates. Three measurements on this
service, with the numbers they produced:

| question | result | verdict |
|---|---|---|
| Can it rank a **complete** layout? | good `0.698` vs bad `0.568` (Δ0.13) | yes |
| Can it rank a **partial** layout? | promising `0.500` vs doomed `0.515` (Δ0.015); with the residual void described in the input, **backwards** (`0.333` vs `0.433`) | no — it cannot see the residual space we do not put in the state |
| Does its ranking agree with the solver's exact score? | mean Spearman **0.188** over 7 corpus programmes (range +0.895 to −0.684); 6/7 winners differ, always by near-tie margins | no — uncorrelated |
| If it drives the solver's room ORDER (`packages/solver/bench/guided-ordering.ts`) | model-greedy `0.7297` vs score-greedy `0.7781` mean over 5 programmes, 116 decisions: −0.048 on average, one case +0.050 | signal, but not reliable |

Consequences for how this service is used: keep it on the cold path, judging
plans the solver has already built (fused ranking, tie-breaking,
explainability). Do **not** let it prune the search or steer generation — it
is asked about partial states there, which it cannot judge, and the exact
geometry it would be paraphrasing is cheaper and exact.

Scripts: `scripts/probe_partial.py`, `scripts/probe_void.py`,
`scripts/ste_state_check.py`, `scripts/ab_models.py`. TS benches:
`packages/solver/bench/judge-calibration.ts` and
`packages/solver/bench/guided-ordering.ts`.

## The state text obeys Simplified Technical English

The model reads the state as plain text, and it is small (2B). Its
comprehension is therefore a function of how the text is written, so the
state is checked against the STE writing rules — short paragraphs, one idea
per sentence, no stacked clauses — rather than treated as prose:

- `tools/ste/ste_check.py` is the checker, vendored from
  [simplified-technical-english](https://github.com/adeerkhan/simplified-technical-english)
  (MIT for the tool; the word list is ASD's, see `tools/ste/NOTICE.md`).
- `tests/test_state_text.py` fails when a state or a default question
  breaks a rule, so the text cannot drift into long sentences.
- `scripts/ste_state_check.py` prints a state and its rule report:

      ./.venv/bin/python scripts/ste_state_check.py

Only the structural rules are enforced: room and zone names are technical
nouns the approved-word list does not carry, so a full dictionary check
would fail on `residential` and `utility`.

## Development

```bash
# Run tests
pytest tests/ -v

# Install dev dependencies
uv pip install -e ".[dev]"
```
