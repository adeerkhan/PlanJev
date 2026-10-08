"""Tests for the decision engine."""

from __future__ import annotations

import pytest

from engine.decision_engine import DecisionEngine
from engine.schemas import (
    DecisionType,
    DecisionRequest,
    LayoutCandidate,
    ProjectBrief,
    RoomSpec,
    SiteBrief,
    ValidationRequest,
    ZoneType,
)


@pytest.fixture
def sample_brief() -> ProjectBrief:
    """Create a sample project brief."""
    return ProjectBrief(
        project_id="test-001",
        site=SiteBrief(
            width_m=20.0,
            depth_m=15.0,
            setbacks_m={"north": 2.0, "south": 2.0, "east": 2.0, "west": 2.0},
            floor_count=1,
        ),
        rooms=[
            RoomSpec(
                id="living",
                name="Living Room",
                zone_type=ZoneType.RESIDENTIAL,
                target_area_m2=30.0,
                preferred_adjacent=["kitchen"],
            ),
            RoomSpec(
                id="kitchen",
                name="Kitchen",
                zone_type=ZoneType.RESIDENTIAL,
                target_area_m2=15.0,
                preferred_adjacent=["living"],
            ),
            RoomSpec(
                id="bedroom",
                name="Bedroom",
                zone_type=ZoneType.RESIDENTIAL,
                target_area_m2=20.0,
                must_not_be_adjacent=["kitchen"],
            ),
        ],
    )


@pytest.fixture
def sample_candidate() -> LayoutCandidate:
    """Create a sample layout candidate."""
    return LayoutCandidate(
        candidate_id="cand-001",
        room_placements={
            "living": {"x": 2.0, "y": 2.0, "w": 6.0, "h": 5.0, "floor": 0},
            "kitchen": {"x": 8.0, "y": 2.0, "w": 4.0, "h": 4.0, "floor": 0},
            "bedroom": {"x": 2.0, "y": 7.0, "w": 5.0, "h": 4.0, "floor": 0},
        },
    )


@pytest.fixture
def engine() -> DecisionEngine:
    """Create a decision engine (without loading the model)."""
    return DecisionEngine()


class TestDecisionEngine:
    """Test the decision engine."""

    def test_score_candidates(
        self, engine: DecisionEngine, sample_brief: ProjectBrief, sample_candidate: LayoutCandidate
    ) -> None:
        """Test scoring candidates."""
        request = DecisionRequest(
            brief=sample_brief,
            candidates=[sample_candidate],
            decision_type=DecisionType.OVERALL,
        )
        response = engine.score_candidates(request)
        assert response.best_candidate_id == "cand-001"
        assert len(response.ranked_candidates) == 1
        assert 0 <= response.ranked_candidates[0].overall_score <= 1

    def test_validate_layout(
        self, engine: DecisionEngine, sample_brief: ProjectBrief, sample_candidate: LayoutCandidate
    ) -> None:
        """Test layout validation."""
        request = ValidationRequest(brief=sample_brief, candidate=sample_candidate)
        response = engine.validate_layout(request)
        assert isinstance(response.is_valid, bool)
        assert isinstance(response.violations, list)
        assert isinstance(response.scores, dict)

    def test_overlap_detection(self, engine: DecisionEngine) -> None:
        """Test that overlapping rooms are detected."""
        a = {"x": 0, "y": 0, "w": 5, "h": 5}
        b = {"x": 3, "y": 3, "w": 5, "h": 5}
        assert engine._rects_overlap(a, b) is True

        c = {"x": 10, "y": 10, "w": 5, "h": 5}
        assert engine._rects_overlap(a, c) is False

    def test_adjacency_detection(self, engine: DecisionEngine) -> None:
        """Test adjacency detection."""
        candidate = LayoutCandidate(
            candidate_id="test",
            room_placements={
                "a": {"x": 0, "y": 0, "w": 5, "h": 5},
                "b": {"x": 5, "y": 0, "w": 5, "h": 5},  # Adjacent to a
                "c": {"x": 20, "y": 20, "w": 5, "h": 5},  # Not adjacent
            },
        )
        assert engine._are_rooms_adjacent("a", "b", candidate) is True
        assert engine._are_rooms_adjacent("a", "c", candidate) is False

    def test_boundary_violation(self, engine: DecisionEngine) -> None:
        """Test boundary violation detection."""
        site = SiteBrief(width_m=10, depth_m=10)
        candidate = LayoutCandidate(
            candidate_id="test",
            room_placements={
                "big": {"x": 0, "y": 0, "w": 15, "h": 15},  # Exceeds boundaries
            },
        )
        violations = engine._check_boundaries(site, candidate)
        assert len(violations) > 0
