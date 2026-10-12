"""Print the state text the decision model reads, then check it against the
vendored STE rule checker (tools/ste/ste_check.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from engine.decision_engine import DecisionEngine  # noqa: E402
from engine.schemas import (  # noqa: E402
    LayoutCandidate,
    ProjectBrief,
    RoomSpec,
    SiteBrief,
    ZoneType,
)


def main() -> None:
    brief = ProjectBrief(
        project_id="ste",
        site=SiteBrief(width_m=10.0, depth_m=12.0),
        rooms=[
            RoomSpec(id="living", name="Living", zone_type=ZoneType.RESIDENTIAL,
                     target_area_m2=24.0, must_be_adjacent=["kitchen"]),
            RoomSpec(id="kitchen", name="Kitchen", zone_type=ZoneType.RESIDENTIAL,
                     target_area_m2=10.0),
            RoomSpec(id="bedroom-1", name="Bedroom 1", zone_type=ZoneType.RESIDENTIAL,
                     target_area_m2=12.0, must_be_adjacent=["bathroom"],
                     must_not_be_adjacent=["kitchen"]),
            RoomSpec(id="bathroom", name="Bathroom", zone_type=ZoneType.UTILITY,
                     target_area_m2=5.0),
        ],
    )
    candidate = LayoutCandidate(
        candidate_id="c",
        room_placements={
            "living": {"x": 0, "y": 0, "w": 6, "h": 4},
            "kitchen": {"x": 6, "y": 0, "w": 4, "h": 2.5},
            "bedroom-1": {"x": 0, "y": 4, "w": 5, "h": 6},
            "bathroom": {"x": 5, "y": 4, "w": 2, "h": 2},
        },
    )
    state = DecisionEngine()._build_model_state(brief, candidate)
    print(state)
    print("=== STE rules check ===")
    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable, str(BACKEND / "tools/ste/ste_check.py"),
         "--no-vocab", "--mode", "descriptive"],
        input=state, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print(f"exit={result.returncode}")


if __name__ == "__main__":
    main()
