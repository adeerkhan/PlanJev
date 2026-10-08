"""Pydantic schemas for SpatialDecide Engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class DecisionType(str, Enum):
    ADJACENCY = "adjacency"
    ZONING = "zoning"
    SETBACK = "setback"
    AREA = "area"
    ORIENTATION = "orientation"
    OVERALL = "overall"


class ZoneType(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    GREEN = "green"
    CIRCULATION = "circulation"
    UTILITY = "utility"
    MIXED = "mixed"


# ── Input Schemas ────────────────────────────────────────────────────────────

class RoomSpec(BaseModel):
    """A single room/zone specification."""
    id: str
    name: str
    zone_type: ZoneType
    target_area_m2: float = Field(gt=0)
    min_dimension_m: float | None = Field(default=None, gt=0)
    max_dimension_m: float | None = Field(default=None, gt=0)
    preferred_adjacent: list[str] = Field(default_factory=list)
    must_be_adjacent: list[str] = Field(default_factory=list)
    must_not_be_adjacent: list[str] = Field(default_factory=list)
    floor_preference: int | None = Field(default=None, ge=0)
    aspect_ratio: float | None = Field(default=None, gt=0)


class SiteBrief(BaseModel):
    """Site dimensions and constraints."""
    width_m: float = Field(gt=0)
    depth_m: float = Field(gt=0)
    setbacks_m: dict[str, float] = Field(
        default_factory=lambda: {"north": 0, "south": 0, "east": 0, "west": 0}
    )
    max_height_m: float | None = None
    floor_count: int = Field(default=1, ge=1)


class ProjectBrief(BaseModel):
    """Full architectural project brief."""
    project_id: str
    site: SiteBrief
    rooms: list[RoomSpec]
    global_constraints: dict[str, Any] = Field(default_factory=dict)


class LayoutCandidate(BaseModel):
    """A candidate layout to be scored."""
    candidate_id: str
    room_placements: dict[str, dict[str, float]]  # room_id -> {x, y, w, h, floor}
    adjacency_matrix: dict[str, list[str]] | None = None


class DecisionRequest(BaseModel):
    """Request to the decision engine."""
    brief: ProjectBrief
    candidates: list[LayoutCandidate]
    decision_type: DecisionType = DecisionType.OVERALL
    top_k: int = Field(default=3, ge=1, le=10)


# ── Output Schemas ───────────────────────────────────────────────────────────

class RoomDecision(BaseModel):
    """Decision for a single room."""
    room_id: str
    score: float = Field(ge=0, le=1)
    reasoning: str
    violations: list[str] = Field(default_factory=list)


class CandidateScore(BaseModel):
    """Score for a single layout candidate."""
    candidate_id: str
    overall_score: float = Field(ge=0, le=1)
    room_decisions: list[RoomDecision]
    adjacency_score: float = Field(ge=0, le=1)
    zoning_score: float = Field(ge=0, le=1)
    area_score: float = Field(ge=0, le=1)
    violations: list[str] = Field(default_factory=list)


class DecisionResponse(BaseModel):
    """Response from the decision engine."""
    request_id: str
    ranked_candidates: list[CandidateScore]
    best_candidate_id: str
    model_used: str
    inference_time_ms: float


class ValidationRequest(BaseModel):
    """Request to validate a single layout."""
    brief: ProjectBrief
    candidate: LayoutCandidate


class ValidationResponse(BaseModel):
    """Validation result for a single layout."""
    is_valid: bool
    violations: list[str]
    warnings: list[str]
    scores: dict[str, float]
