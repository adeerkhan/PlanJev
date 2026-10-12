"""The text the decision model reads must obey the STE writing rules.

The model is small (2B parameters) and reads the state exactly as built.
Simplified Technical English's structural rules — short paragraphs, one
idea per sentence, no stacked clauses — are precisely what such a model
parses reliably, so the state text is a checked artifact, not prose.

The checker is vendored from
https://github.com/adeerkhan/simplified-technical-english (MIT for the
tool; the word list itself is ASD's — see tools/ste/NOTICE.md). Only the
structural rules are enforced here: room and zone names are technical
nouns that the approved-word list does not carry, which the tool reports
as "unknown words" rather than rule violations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from engine.decision_engine import DEFAULT_DECIDE_QUESTIONS, DecisionEngine
from engine.schemas import (
    LayoutCandidate,
    ProjectBrief,
    RoomSpec,
    SiteBrief,
    ZoneType,
)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
STE_CHECKER = BACKEND_ROOT / "tools/ste/ste_check.py"


@pytest.fixture(scope="module")
def ste_check():
    if not STE_CHECKER.exists():
        pytest.fail(f"STE checker missing at {STE_CHECKER}")
    sys.path.insert(0, str(STE_CHECKER.parent))
    import ste_check as checker

    return checker


def sample_brief() -> ProjectBrief:
    return ProjectBrief(
        project_id="ste-001",
        site=SiteBrief(width_m=10.0, depth_m=12.0),
        rooms=[
            RoomSpec(id="living", name="Living", zone_type=ZoneType.RESIDENTIAL,
                     target_area_m2=24.0, must_be_adjacent=["kitchen"]),
            RoomSpec(id="kitchen", name="Kitchen", zone_type=ZoneType.RESIDENTIAL,
                     target_area_m2=10.0),
            RoomSpec(id="bedroom-1", name="Bedroom 1", zone_type=ZoneType.RESIDENTIAL,
                     target_area_m2=12.0, must_be_adjacent=["bathroom"],
                     must_not_be_adjacent=["kitchen"]),
            RoomSpec(id="bedroom-2", name="Bedroom 2", zone_type=ZoneType.RESIDENTIAL,
                     target_area_m2=12.0, must_be_adjacent=["bathroom"]),
            RoomSpec(id="bathroom", name="Bathroom", zone_type=ZoneType.UTILITY,
                     target_area_m2=5.0),
        ],
    )


def sample_candidate() -> LayoutCandidate:
    return LayoutCandidate(
        candidate_id="c",
        room_placements={
            "living": {"x": 0, "y": 0, "w": 6, "h": 4},
            "kitchen": {"x": 6, "y": 0, "w": 4, "h": 2.5},
            "bedroom-1": {"x": 0, "y": 4, "w": 3, "h": 4},
            "bedroom-2": {"x": 3, "y": 4, "w": 3, "h": 4},
            "bathroom": {"x": 6, "y": 2.5, "w": 2.5, "h": 2.5},
        },
    )


class TestStateTextFollowsSte:
    engine = DecisionEngine()

    def _violations(self, text: str, mode: str, checker) -> list:
        report = checker.Report()
        checker.check_text(text, mode, report, "text", None)  # approved=None: rules only
        return report.errors

    def test_state_text_has_no_rule_errors(self, ste_check) -> None:
        state = self.engine._build_model_state(sample_brief(), sample_candidate())
        assert self._violations(state, "descriptive", ste_check) == []

    def test_state_without_requirements_has_no_rule_errors(self, ste_check) -> None:
        """A programme with no adjacency rules skips the Requirements block."""
        bare = ProjectBrief(
            project_id="bare",
            site=SiteBrief(width_m=10.0, depth_m=12.0),
            rooms=[
                RoomSpec(id="living", name="Living", zone_type=ZoneType.RESIDENTIAL,
                         target_area_m2=24.0),
                RoomSpec(id="kitchen", name="Kitchen", zone_type=ZoneType.RESIDENTIAL,
                         target_area_m2=10.0),
                RoomSpec(id="bathroom", name="Bathroom", zone_type=ZoneType.UTILITY,
                         target_area_m2=5.0),
            ],
        )
        state = self.engine._build_model_state(bare, sample_candidate())
        assert "Requirements:" not in state
        assert self._violations(state, "descriptive", ste_check) == []

    def test_default_questions_have_no_rule_errors(self, ste_check) -> None:
        """The question text is model input too: check it as procedural."""
        text = "\n".join(
            f"{spec['instructions']}" for spec in DEFAULT_DECIDE_QUESTIONS.values()
        )
        assert self._violations(text, "procedural", ste_check) == []
