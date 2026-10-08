"""Model loader for the Laya decision model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID = "convaiinnovations/laya"
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"


class ModelLoadError(Exception):
    """Raised when the model fails to load."""


class LayaModel:
    """Wrapper around the Laya decision model.

    Laya is a multilingual, non-autoregressive System 1 decision model.
    It takes a state (text) and typed questions, returning calibrated
    probabilities in a single forward pass.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_id = MODEL_ID
        self._model_path = str(model_path) if model_path else MODEL_ID
        self._agent: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the Laya model."""
        if self._loaded:
            return

        logger.info("Loading Laya model from %s", self._model_path)

        import laya

        self._agent = laya.load(self._model_path)
        self._loaded = True
        logger.info("Laya model loaded successfully")

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        """Run inference with state and typed questions.

        Returns the full result dict with answers and routing info.
        """
        if not self._loaded:
            self.load()

        assert self._agent is not None
        return self._agent.predict(state, questions)

    def score(self, state: str, question: dict[str, Any] | None = None) -> float:
        """Return a single scalar score for the given state.

        Uses a default scoring question if none provided.
        """
        if question is None:
            question = {
                "type": "score",
                "instructions": "How good is this layout?",
                "criteria": ["poor", "fair", "good", "excellent"],
            }

        result = self.predict(state, {"overall": question})
        answers = result.get("answers", {})
        overall = answers.get("overall", {})

        # For score type, return the normalized score
        if "score" in overall:
            raw_score = overall["score"]
            # Normalize to 0-1 range (assuming 0-max criteria range)
            criteria_count = len(question.get("criteria", []))
            if criteria_count > 1:
                return raw_score / (criteria_count - 1)
            return raw_score

        # For noul type, return the probability
        if "noul" in overall:
            return overall["noul"]

        # Fallback
        return 0.5

    def noul(self, state: str, instructions: str) -> float:
        """Ask a yes/no question and return the probability."""
        question = {"type": "noul", "instructions": instructions}
        result = self.predict(state, {"q": question})
        answers = result.get("answers", {})
        return answers.get("q", {}).get("noul", 0.5)


# Singleton instance
_laya_model: LayaModel | None = None


def get_model() -> LayaModel:
    """Get or create the singleton model instance."""
    global _laya_model
    if _laya_model is None:
        _laya_model = LayaModel()
    return _laya_model
