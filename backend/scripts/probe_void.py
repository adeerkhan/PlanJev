"""Does the model discriminate a PARTIAL layout once the residual space is
actually in its input?

Probe 1 (server-built state, placements only): no discrimination.
Probe 2 (this file): the free region is stated inside the question's
instructions, so the model really does receive it. One request per
candidate, because the wire contract sends one question set per request.
"""

from __future__ import annotations

import json
import urllib.request

URL = "http://127.0.0.1:8000/api/v1/decide"

ROOMS = [
    {"id": "living", "name": "Living", "zone_type": "residential", "target_area_m2": 24,
     "must_be_adjacent": ["kitchen"]},
    {"id": "kitchen", "name": "Kitchen", "zone_type": "residential", "target_area_m2": 10},
    {"id": "bedroom-1", "name": "Bedroom 1", "zone_type": "residential", "target_area_m2": 12,
     "must_be_adjacent": ["bathroom"]},
    {"id": "bedroom-2", "name": "Bedroom 2", "zone_type": "residential", "target_area_m2": 12,
     "must_be_adjacent": ["bathroom"]},
    {"id": "bathroom", "name": "Bathroom", "zone_type": "utility", "target_area_m2": 5},
]

CASES = [
    ("promising", "one free rectangle 12.0m x 6.0m along the whole south edge",
     {"living": {"x": 0, "y": 0, "w": 6, "h": 4}, "kitchen": {"x": 6, "y": 0, "w": 4, "h": 3}}),
    ("doomed", "two separated free regions, 3.0m x 10.0m and 6.0m x 10.0m, with the "
               "three remaining rooms (including two bedrooms that each need an "
               "exterior wall) to place in them",
     {"living": {"x": 0, "y": 3.5, "w": 3, "h": 3}, "kitchen": {"x": 9, "y": 3.5, "w": 3, "h": 3}}),
]


def ask(name: str, free_space: str, placements: dict) -> tuple[float, float]:
    questions = {
        "layout_quality": {
            "type": "score",
            "instructions": "How good is this floorplan arrangement so far: room proportions, adjacency relationships, daylight access, and use of the site",
            "criteria": ["poor", "below average", "average", "good", "excellent"],
        },
        "still_feasible": {
            "type": "noul",
            "instructions": (
                "Two of five rooms are placed. The free space that is left is "
                f"{free_space}. The remaining rooms can all still get a good "
                "position, so this layout can still become a good apartment."
            ),
        },
    }
    body = json.dumps({
        "brief": {"project_id": "partial", "site": {"width_m": 12, "depth_m": 10, "setbacks_m": {}},
                  "rooms": ROOMS},
        "candidates": [{"candidate_id": name,
                        "room_placements": {k: dict(v) for k, v in placements.items()}}],
        "questions": questions,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))["decisions"][0]
    a = d["answers"]
    return d["model_score"], a["still_feasible"]["noul"], a["layout_quality"]["score"]


print("--- partial layouts, residual space stated in the input ---")
for name, free_space, placements in CASES:
    model, feasible, quality = ask(name, free_space, placements)
    print(f"  {name:<11} model={model:.3f}  q={quality:.2f}  P(still feasible)={feasible:.3f}")
