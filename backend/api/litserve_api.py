"""LitServe API for the Laya decision model."""

from __future__ import annotations

import logging
from typing import Any

import litserve as ls

from engine.model_loader import get_model

logger = logging.getLogger(__name__)


class LayaDecisionAPI(ls.LitAPI):
    """LitServe API wrapping the Laya decision model."""

    def setup(self, device: str) -> None:
        """Load the model on startup."""
        logger.info("Setting up LayaDecisionAPI on device: %s", device)
        self._model = get_model()
        self._model.load()
        logger.info("Model ready for inference")

    def decode_request(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Extract state and questions from the incoming request."""
        state = request.get("state", "")
        questions = request.get("questions", {})
        if not state:
            raise ValueError("Request must contain a 'state' field")
        if not questions:
            raise ValueError("Request must contain a 'questions' field")
        return state, questions

    def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        """Run inference on the state with questions."""
        try:
            result = self._model.predict(state, questions)
            return {
                "answers": result.get("answers", {}),
                "routing": result.get("routing", {}),
            }
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return {
                "answers": {},
                "routing": {},
                "error": str(exc),
            }

    def encode_response(self, output: dict[str, Any]) -> dict[str, Any]:
        """Return the response as-is (already JSON-serializable)."""
        return output


def main() -> None:
    """Run the LitServe server."""
    import os

    port = int(os.getenv("LITSERVE_PORT", "8001"))

    server = ls.LitServer(
        LayaDecisionAPI(),
        accelerator="auto",
        devices=1,
        max_batch_size=8,
        batch_timeout=0.01,
    )
    server.run(port=port)


if __name__ == "__main__":
    main()
