"""Benchmark one decision-model backend on the SAME layout states.

    cd backend
    MODEL_CACHE_DIR=.model_cache ./jev/bin/python scripts/ab_models.py laya
    MODEL_CACHE_DIR=.model_cache ./jev/bin/python scripts/ab_models.py decider-2b

Loads the chosen backend (laya or decider-2b; default laya), scores the same
four candidates with the same typed questions, and reports load time,
batched latency (one warmed-up pass, then three timed passes) and the
per-candidate probabilities. Run it twice and diff the per-candidate lines
to compare backends.

One backend per invocation on purpose: on an 8 GB card keeping both models
resident (laya caches its weights module-wide) exhausts VRAM.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

# Runnable as a plain script: put the backend root on sys.path (pytest gets
# this from pyproject's pythonpath setting).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.decision_engine import DEFAULT_DECIDE_QUESTIONS, DecisionEngine
from engine.model_loader import BACKENDS, _apply_cache_dir
from engine.schemas import LayoutCandidate, ProjectBrief, RoomSpec, SiteBrief, ZoneType


def build_case() -> tuple[ProjectBrief, list[LayoutCandidate]]:
    """A 5-room apartment brief and four candidate arrangements."""
    brief = ProjectBrief(
        project_id="ab-001",
        site=SiteBrief(width_m=10.0, depth_m=12.0),
        rooms=[
            RoomSpec(id="living", name="Living", zone_type=ZoneType.RESIDENTIAL, target_area_m2=24.0,
                     must_be_adjacent=["kitchen"]),
            RoomSpec(id="kitchen", name="Kitchen", zone_type=ZoneType.RESIDENTIAL, target_area_m2=10.0),
            RoomSpec(id="bedroom-1", name="Bedroom 1", zone_type=ZoneType.RESIDENTIAL, target_area_m2=12.0,
                     must_be_adjacent=["bathroom"], must_not_be_adjacent=["kitchen"]),
            RoomSpec(id="bedroom-2", name="Bedroom 2", zone_type=ZoneType.RESIDENTIAL, target_area_m2=12.0,
                     must_be_adjacent=["bathroom"]),
            RoomSpec(id="bathroom", name="Bathroom", zone_type=ZoneType.UTILITY, target_area_m2=5.0),
        ],
    )
    candidates = [
        LayoutCandidate(candidate_id="compact", room_placements={
            "living": {"x": 0, "y": 0, "w": 6, "h": 4}, "kitchen": {"x": 6, "y": 0, "w": 4, "h": 2.5},
            "bedroom-1": {"x": 0, "y": 4, "w": 3, "h": 4}, "bedroom-2": {"x": 3, "y": 4, "w": 3, "h": 4},
            "bathroom": {"x": 6, "y": 2.5, "w": 2.5, "h": 2.5}}),
        LayoutCandidate(candidate_id="spread", room_placements={
            "living": {"x": 0, "y": 0, "w": 4, "h": 5}, "kitchen": {"x": 4, "y": 0, "w": 3, "h": 3},
            "bedroom-1": {"x": 7, "y": 0, "w": 3, "h": 4}, "bedroom-2": {"x": 0, "y": 5, "w": 4, "h": 4},
            "bathroom": {"x": 4, "y": 3, "w": 3, "h": 3}}),
        LayoutCandidate(candidate_id="long-hall", room_placements={
            "living": {"x": 0, "y": 0, "w": 8, "h": 3}, "kitchen": {"x": 8, "y": 0, "w": 2, "h": 3},
            "bedroom-1": {"x": 0, "y": 3, "w": 2, "h": 6}, "bedroom-2": {"x": 2, "y": 3, "w": 2, "h": 6},
            "bathroom": {"x": 4, "y": 3, "w": 4, "h": 4}}),
        LayoutCandidate(candidate_id="walled-kitchen", room_placements={
            "living": {"x": 0, "y": 0, "w": 5, "h": 6}, "kitchen": {"x": 5, "y": 0, "w": 5, "h": 3},
            "bedroom-1": {"x": 0, "y": 6, "w": 5, "h": 4}, "bedroom-2": {"x": 5, "y": 3, "w": 2.5, "h": 7},
            "bathroom": {"x": 7.5, "y": 3, "w": 2.5, "h": 7}}),
    ]
    return brief, candidates


def main() -> None:
    _apply_cache_dir()
    backend = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("DECISION_MODEL", "laya")).strip().lower()
    if backend not in BACKENDS:
        raise SystemExit(f"unknown backend {backend!r}; expected one of {sorted(BACKENDS)}")

    brief, candidates = build_case()
    engine = DecisionEngine()
    states = [engine._build_model_state(brief, c) for c in candidates]

    model = BACKENDS[backend]()
    t0 = time.perf_counter()
    model.load()
    load_s = time.perf_counter() - t0
    device = getattr(getattr(model, "_decider", None), "dev", "cpu")

    # Warm-up: compiles kernels / captures CUDA graphs.
    model.predict_batch(states, DEFAULT_DECIDE_QUESTIONS)

    latencies: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = model.predict_batch(states, DEFAULT_DECIDE_QUESTIONS)
        latencies.append((time.perf_counter() - t0) * 1000)

    median = statistics.median(latencies)
    print(f"== {backend} (device={device}) ==")
    print(f"   load       : {load_s:8.1f} s")
    print(f"   batch({len(states)})   : median {median:8.1f} ms   "
          f"min {min(latencies):.1f} ms   ({median / len(states):.0f} ms/state)")
    for candidate, answers in zip(candidates, out):
        raw = answers.get("answers", {})
        score = engine._answers_to_score(raw, DEFAULT_DECIDE_QUESTIONS)
        quality = raw.get("layout_quality", {})
        buildable = raw.get("is_buildable", {})
        print(f"   {candidate.candidate_id:<15} model={score:.3f}  "
              f"q={quality.get('score', float('nan')):.2f}  "
              f"P(ok)={buildable.get('noul', float('nan')):.3f}")


if __name__ == "__main__":
    main()
