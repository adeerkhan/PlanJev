"""Can the decision model judge an INCOMPLETE layout — if told about the void?

The solver only ever scores finished plans (evaluateCandidate needs a full
rect set). A judge that ranks PARTIAL layouts could prune the search long
before it finishes — but the model only knows what the state text says.

Groups compared here:
  partial  — 2 of 5 rooms placed, with the free region described
  full     — all 5 rooms placed, a good and a bad arrangement
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

QUESTIONS = {
    "layout_quality": {
        "type": "score",
        "instructions": "How good is this floorplan arrangement so far: room proportions, adjacency relationships, daylight access, and use of the site",
        "criteria": ["poor", "below average", "average", "good", "excellent"],
    },
    "still_feasible": {
        "type": "noul",
        "instructions": "The rooms that are not placed yet can all get a good position and an exterior wall, so this layout can still become a good apartment",
    },
}

# Free space spelled out, because the model only knows the state text.
PARTIALS = [
    {
        "candidate_id": "promising",
        "room_placements": {
            "living": {"x": 0, "y": 0, "w": 6, "h": 4},
            "kitchen": {"x": 6, "y": 0, "w": 4, "h": 3},
        },
        "free_space": "One free rectangle, 12.0m x 6.0m, along the whole south edge",
    },
    {
        "candidate_id": "doomed",
        "room_placements": {
            "living": {"x": 0, "y": 3.5, "w": 3, "h": 3},
            "kitchen": {"x": 9, "y": 3.5, "w": 3, "h": 3},
        },
        "free_space": "Two separated free regions, 3.0m x 10.0m and 6.0m x 10.0m",
    },
]

FULL = [
    {
        "candidate_id": "good",
        "room_placements": {
            "living": {"x": 0, "y": 0, "w": 6, "h": 4},
            "kitchen": {"x": 6, "y": 0, "w": 4, "h": 3},
            "bedroom-1": {"x": 0, "y": 4, "w": 3, "h": 6},
            "bedroom-2": {"x": 3, "y": 4, "w": 3, "h": 6},
            "bathroom": {"x": 6, "y": 3, "w": 2.5, "h": 2.5},
        },
    },
    {
        "candidate_id": "bad",
        "room_placements": {
            "living": {"x": 0, "y": 0, "w": 3, "h": 3},
            "kitchen": {"x": 9, "y": 9, "w": 2, "h": 1},
            "bedroom-1": {"x": 8, "y": 8, "w": 2, "h": 2},
            "bedroom-2": {"x": 10, "y": 2, "w": 2, "h": 2},
            "bathroom": {"x": 0, "y": 9, "w": 1.5, "h": 1},
        },
    },
]


def ask(candidates: list[dict], free_space: bool = False) -> list[dict]:
    """Render each candidate as a state string and send them as one batch."""
    states = []
    for c in candidates:
        lines = [
            "Site: 12.0m x 10.0m",
            f"Rooms: {len(ROOMS)}",
            f"Placed: {len(c['room_placements'])} of {len(ROOMS)}",
            "",
            "Requirements:",
            "- Living must adjoin Kitchen",
            "",
            "Layout:",
        ]
        lines += [
            f"- {room_id} : ({p['x']:.1f}, {p['y']:.1f}) {p['w']:.1f}m x {p['h']:.1f}m"
            for room_id, p in c["room_placements"].items()
        ]
        if free_space and "free_space" in c:
            lines += ["", f"Free space: {c['free_space']}"]
        states.append("\n".join(lines))

    # The service scores STATES; reuse its own serializer by putting the
    # placements through the normal wire shape and reading back the scores.
    body = json.dumps({
        "brief": {"project_id": "partial", "site": {"width_m": 12, "depth_m": 10, "setbacks_m": {}},
                  "rooms": ROOMS},
        "candidates": [{k: v for k, v in c.items() if k != "free_space"} for c in candidates],
        "questions": QUESTIONS,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["decisions"]


def show(label: str, decisions: list[dict]) -> None:
    print(f"--- {label} ---")
    for d in decisions:
        a = d["answers"]
        print(
            f"  {d['candidate_id']:<11} model={d['model_score']:.3f}  "
            f"q={a['layout_quality']['score']:.2f}  "
            f"P(still feasible)={a['still_feasible']['noul']:.3f}"
        )


show("partial layouts (void NOT described)", ask(PARTIALS))
show("partial layouts (void described in the state)", ask(PARTIALS))
show("full layouts (control)", ask(FULL))
