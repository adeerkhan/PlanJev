"""Model loader for the Laya decision model."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID = "convaiinnovations/laya"


class LayaModel:
    """Wrapper around the Laya decision model.

    Laya is a multilingual, non-autoregressive System 1 decision model.
    It takes a state (text) and typed questions, returning calibrated
    probabilities in a single forward pass.
    """

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self._agent: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the Laya model."""
        if self._loaded:
            return

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

    def score(self, state: str, question: dict[str, Any] | None = None) -> float:
        """Return a single scalar score for the given state."""
        if question is None:
            question = {
                "type": "score",
                "instructions": "How good is this layout?",
                "criteria": ["poor", "fair", "good", "excellent"],
            }

        result = self.predict(state, {"overall": question})
        answers = result.get("answers", {})
        overall = answers.get("overall", {})

        if "score" in overall:
            raw_score = overall["score"]
            criteria_count = len(question.get("criteria", []))
            return raw_score / (criteria_count - 1) if criteria_count > 1 else raw_score

        if "noul" in overall:
            return overall["noul"]

        return 0.5


# Singleton instance
_laya_model: LayaModel | None = None


def get_model() -> LayaModel:
    """Get or create the singleton model instance."""
    global _laya_model
    if _laya_model is None:
        _laya_model = LayaModel()
    return _laya_model
