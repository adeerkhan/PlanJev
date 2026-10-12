"""Model loader for the decision model.

Two interchangeable backends, both speaking the System One typed-question
contract (state + questions in, calibrated probabilities out, one forward
pass per batch):

- `laya` (default) — convaiinnovations/laya, the original production path.
- `decider-2b` — Mapika/decider-2b, an open reproduction of the same model
  class on Qwen3.5-2B: identical question/answer shapes, a 32k-token state
  budget instead of Laya's 512, and CUDA (bf16) batched scoring where a
  whole portfolio of candidates is judged in one pass.

Select with DECISION_MODEL=laya|decider-2b (default laya).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID = "convaiinnovations/laya"
DECIDER_MODEL_ID = "Mapika/decider-2b"

# The question `score()` asks when the caller supplies none, and the collapse
# it applies to the answer — shared by both backends, which differ only in how
# they run inference.
DEFAULT_SCORE_QUESTION: dict[str, Any] = {
    "type": "score",
    "instructions": "How good is this layout?",
    "criteria": ["poor", "fair", "good", "excellent"],
}


def _collapse_to_score(result: dict[str, Any], question: dict[str, Any]) -> float:
    """Collapse a predict() result to one scalar: the score answer normalized
    by its criteria count, else the noul probability, else neutral 0.5."""
    overall = result.get("answers", {}).get("overall", {})

    if "score" in overall:
        raw_score = overall["score"]
        criteria_count = len(question.get("criteria", []))
        return raw_score / (criteria_count - 1) if criteria_count > 1 else raw_score

    if "noul" in overall:
        return overall["noul"]

    return 0.5


def _ensure_gpu_scratch_space() -> None:
    """Keep Triton's compile scratch space off a full system temp disk.

    decider-2b's fused Qwen3.5 kernels are compiled by Triton on first use.
    Triton defaults `TMPDIR`/`TRITON_CACHE_DIR` to the system temp dir, which
    on many desktops is a small tmpfs; when it fills, compilation fails and
    the kernels SILENTLY fall back to slow CPU ones (the log says "roll back
    to CPU"). Both point at backend/.tmp on the project disk instead, so the
    fast kernels compile and stay cached across restarts.

    `os.environ.setdefault` lets an explicit setting win. TMPDIR is read by
    the C compiler Triton spawns as a child of this process, so setting it
    here is early enough for the load below.
    """
    scratch = Path(__file__).resolve().parent.parent / ".tmp"
    os.environ.setdefault("TMPDIR", str(scratch))
    os.environ.setdefault("TRITON_CACHE_DIR", str(scratch / "triton"))
    (scratch / "triton").mkdir(parents=True, exist_ok=True)


def _apply_cache_dir() -> None:
    """Honour MODEL_CACHE_DIR (see .env.example) for the HF cache.

    huggingface_hub reads HF_HOME at load time; without this the server
    would re-download the weights to ~/.cache even when the repo ships a
    populated cache directory. An explicit HF_HOME/HF_HUB_CACHE wins.
    """
    cache_dir = os.environ.get("MODEL_CACHE_DIR")
    if not cache_dir or os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
        return
    path = Path(cache_dir)
    if not path.is_absolute():
        # Resolve against the backend root, not the process CWD.
        path = Path(__file__).resolve().parent.parent / path
    os.environ["HF_HOME"] = str(path)


class LayaModel:
    """Wrapper around the Laya decision model.

    Laya is a multilingual, non-autoregressive System 1 decision model.
    It takes a state (text) and typed questions, returning calibrated
    probabilities in a single forward pass.
    """

    backend = "laya"

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self._agent: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def max_batch_candidates(self) -> int:
        """Laya is 421M on CPU: batches far larger than any portfolio."""
        return MAX_BATCH_CANDIDATES

    def load(self) -> None:
        """Load the Laya model."""
        if self._loaded:
            return

        _apply_cache_dir()
        logger.info("Loading Laya model from %s", self.model_id)

        import laya

        self._agent = laya.load(self.model_id)
        self._loaded = True
        logger.info("Laya model loaded successfully")

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        """Run inference with state and typed questions."""
        if not self._loaded:
            self.load()

        assert self._agent is not None
        return self._agent.predict(state, questions)

    def predict_batch(
        self, states: list[str], questions: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Evaluate the same questions over many states in one forward pass.

        The throughput path: N layout candidates take ceil(N / batch_size)
        passes instead of N, which is what makes re-ranking a portfolio
        cheap enough to run inside a solve.
        """
        if not self._loaded:
            self.load()

        assert self._agent is not None
        return self._agent.predict_batch(states, questions)

    def score(self, state: str, question: dict[str, Any] | None = None) -> float:
        """Return a single scalar score for the given state."""
        question = question or DEFAULT_SCORE_QUESTION
        return _collapse_to_score(self.predict(state, {"overall": question}), question)


class DeciderModel:
    """Mapika/decider-2b: an open System One decision model (Qwen3.5-2B).

    Drop-in for LayaModel from the caller's side — `predict`/`predict_batch`
    take the same states and questions and return the same answer shapes
    (noul / score+legend / choice with probabilities) — so `DecisionEngine`
    and the HTTP contract need no changes to switch backends.

    Two differences that matter for floorplans:

    - **State budget**: 32k tokens instead of Laya's 512, so a large brief
      with its adjacency requirements is never silently truncated.
    - **Speed**: device and dtype default to CUDA + bfloat16 when a GPU is
      present (CPU otherwise), with shape-bucketed CUDA graphs; a portfolio
      of candidates is scored in one batched pass.

    The `decider` inference package ships inside the model repository and is
    put on `sys.path` at load time, so no extra install step is needed.
    """

    backend = "decider-2b"

    def __init__(self) -> None:
        self.model_id = DECIDER_MODEL_ID
        self._decider: Any = None
        self._batch_limit = MAX_BATCH_CANDIDATES

    @property
    def is_loaded(self) -> bool:
        return self._decider is not None

    def max_batch_candidates(self) -> int:
        """Candidates the model can judge in one GPU pass right now.

        Starts at `DECISION_MAX_CANDIDATES` (default %d, measured on a
        shared 8 GB laptop card: the weights, the CUDA-graph pools and the
        Triton workspace take most of it) and only ever LOWERS itself, when
        a real batch runs out of memory. It never probes upward: a batch
        that happens to fit while the desktop is idle captures graph pools
        that then make the next, larger request fail.
        """ % MAX_BATCH_CANDIDATES
        return self._batch_limit

    def load(self) -> None:
        if self._decider is not None:
            return

        _apply_cache_dir()
        _ensure_gpu_scratch_space()
        logger.info("Loading decider model from %s", self.model_id)

        import sys

        from huggingface_hub import snapshot_download

        # Weights plus the bundled `decider` package land in one snapshot.
        local_dir = snapshot_download(self.model_id)
        if local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        from decider.infer import Decider

        # Decider picks cuda+bf16 when available, else cpu; use_graphs
        # follows the device automatically.
        self._decider = Decider(local_dir)
        logger.info(
            "Decider model loaded (%s, device=%s)",
            getattr(self._decider, "name", self.model_id),
            getattr(self._decider, "dev", "unknown"),
        )

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        """Score one state against the typed questions."""
        if self._decider is None:
            self.load()
        assert self._decider is not None
        return self._decider.system_one(state, questions)

    def predict_batch(
        self, states: list[str], questions: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Score every state against the same questions.

        On CUDA the questions are compiled once into a schema and every
        state is scored in a single batched pass — a portfolio re-rank is
        one forward pass. The schema cache needs the compiled engine, which
        only exists on CUDA; on CPU each state takes its own pass. The
        answers are identical in shape either way.
        """
        if self._decider is None:
            self.load()
        assert self._decider is not None
        if getattr(self._decider, "eng", None) is None:
            return [self._decider.system_one(state, questions) for state in states]
        # A one-candidate request is guidance traffic: the solver asks about a
        # single partial plan, with the options inside the question. The schema
        # path compiles a fresh schema and captures fresh CUDA graphs per
        # distinct question set, and the question text changes on every call
        # (it states the residual free area and the room ids on offer) — so a
        # corpus run's hundreds of guidance requests grow the graph pool until
        # even this shape 503s, and the service only recovers on a restart.
        # The single-state path scores through the engine's shape-keyed graphs
        # instead: one bounded set, shared by every request. (This also pins
        # single-candidate /api/v1/score requests to the same path — same
        # answer shapes, so the endpoint contract is unchanged.)
        if len(states) == 1:
            return [self._decider.system_one(states[0], questions)]
        import torch  # already imported by load(); needed for the OOM guard

        try:
            # The schema compile captures CUDA graphs and allocates, so a
            # batch too big for the card can fail here as well as in batch().
            schema = self._decider.schema(questions)
            out = schema.batch(list(states))
        except torch.cuda.OutOfMemoryError:
            # VRAM is shared and moves. Report the smaller size from now on,
            # and reclaim the allocator's blocks before the error travels,
            # or every later request fails the same way.
            torch.cuda.empty_cache()
            self._batch_limit = min(self._batch_limit, max(1, len(states) - 1))
            raise
        # Hand the allocator's idle blocks back between requests. On a card
        # shared with a desktop this is the difference between a small
        # request fitting after a big one and failing on fragmentation.
        torch.cuda.empty_cache()
        return out

    def score(self, state: str, question: dict[str, Any] | None = None) -> float:
        """Return a single scalar score for the given state."""
        question = question or DEFAULT_SCORE_QUESTION
        return _collapse_to_score(self.predict(state, {"overall": question}), question)


# Candidates judged in one GPU pass. Two is what a shared 8 GB card holds
# alongside a desktop; raise DECISION_MAX_CANDIDATES on a dedicated GPU.
MAX_BATCH_CANDIDATES = max(1, int(os.getenv("DECISION_MAX_CANDIDATES", "2")))

# Registry of backends, keyed by the DECISION_MODEL value.
BACKENDS: dict[str, type[LayaModel] | type[DeciderModel]] = {
    "laya": LayaModel,
    "decider-2b": DeciderModel,
}

# Singleton instance (lazily created, rebuilt when the selection changes).
_model: LayaModel | DeciderModel | None = None


def get_model() -> LayaModel | DeciderModel:
    """Get or create the singleton for the configured backend."""
    global _model
    backend = os.getenv("DECISION_MODEL", "laya").strip().lower()
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown DECISION_MODEL {backend!r}; expected one of {sorted(BACKENDS)}"
        )
    if _model is not None and _model.backend != backend:
        # Drop the old backend and hand its VRAM back before the new one
        # loads: two resident models on one card is an OOM.
        _release(_model)
        _model = None
    if _model is None:
        _model = BACKENDS[backend]()
    return _model


def _release(model: LayaModel | DeciderModel) -> None:
    """Drop a model's tensors and return the allocator's blocks to the driver."""
    agent = getattr(model, "_agent", None)
    decider = getattr(model, "_decider", None)
    if agent is not None:
        model._agent = None
        model._loaded = False
    if decider is not None:
        model._decider = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch is already imported on the GPU path
        pass
