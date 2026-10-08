"""LitServe API for the Laya decision model.

This module exposes the model as a high-throughput LitServe endpoint
for integration with the FastAPI app or external consumers.
"""

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

    def decode_request(self, request: dict[str, Any]) -> str:
        """Extract the prompt from the incoming request."""
        prompt = request.get("prompt", "")
        if not prompt:
            raise ValueError("Request must contain a 'prompt' field")
        return prompt

    def predict(self, prompt: str) -> dict[str, Any]:
        """Run inference on the prompt."""
        try:
            probabilities = self._model.predict(prompt)
            score = self._model.score(prompt)
            return {
                "probabilities": probabilities,
                "score": score,
            }
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            return {
                "probabilities": {},
                "score": 0.5,
                "error": str(exc),
            }

    def encode_response(self, output: dict[str, Any]) -> dict[str, Any]:
        """Return the response as-is (already JSON-serializable)."""
        return output


class LayaBatchAPI(ls.LitAPI):
    """LitServe API for batch inference."""

    def setup(self, device: str) -> None:
        """Load the model on startup."""
        logger.info("Setting up LayaBatchAPI on device: %s", device)
        self._model = get_model()
        self._model.load()

    def decode_request(self, request: dict[str, Any]) -> list[str]:
        """Extract prompts from the incoming request."""
        prompts = request.get("prompts", [])
        if not prompts:
            raise ValueError("Request must contain a 'prompts' field with a non-empty list")
        return prompts

    def predict(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Run inference on a batch of prompts."""
        results: list[dict[str, Any]] = []
        for prompt in prompts:
            try:
                probabilities = self._model.predict(prompt)
                score = self._model.score(prompt)
                results.append({
                    "probabilities": probabilities,
                    "score": score,
                })
            except Exception as exc:
                logger.error("Inference failed for prompt: %s", exc)
                results.append({
                    "probabilities": {},
                    "score": 0.5,
                    "error": str(exc),
                })
        return results

    def encode_response(self, output: list[dict[str, Any]]) -> dict[str, Any]:
        """Wrap the batch results."""
        return {"results": output}


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
