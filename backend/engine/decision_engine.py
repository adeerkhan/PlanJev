"""Decision engine that uses the Laya model for spatial scoring and validation."""

from __future__ import annotations

import logging
import time
import uuid

from .model_loader import LayaModel, get_model
from .schemas import (
    CandidateDecision,
    CandidateScore,
    DecideRequest,
    DecideResponse,
    DecisionRequest,
    DecisionResponse,
    LayoutCandidate,
    ProjectBrief,
    RoomDecision,
    RoomSpec,
    SiteBrief,
    ValidationRequest,
    ValidationResponse,
    ZoneType,
)

logger = logging.getLogger(__name__)

INCOMPATIBLE_ZONES = {
    (ZoneType.INDUSTRIAL, ZoneType.RESIDENTIAL),
    (ZoneType.INDUSTRIAL, ZoneType.GREEN),
    (ZoneType.UTILITY, ZoneType.RESIDENTIAL),
}

# The "logic" half of a decide call: typed questions the model answers for
# every candidate state. Callers may override per request; these defaults
# cover the two judgments a floorplan portfolio is ranked by.
DEFAULT_DECIDE_QUESTIONS: dict[str, dict] = {
    "layout_quality": {
        "type": "score",
        "instructions": (
            "How good is this floorplan arrangement overall: room proportions, "
            "adjacency relationships, daylight access, and use of the site"
        ),
        "criteria": ["poor", "below average", "average", "good", "excellent"],
    },
    "is_buildable": {
        "type": "noul",
        "instructions": (
            "The layout is buildable and usable as drawn: rooms do not overlap, "
            "every room is reachable, and service rooms are practically placed"
        ),
    },
}


class DecisionEngine:
    """Core decision engine for spatial layout scoring and validation."""

    def __init__(self) -> None:
        self._model = get_model()

    def score_candidates(self, request: DecisionRequest) -> DecisionResponse:
        """Score and rank layout candidates."""
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]

        scored: list[CandidateScore] = []
        for candidate in request.candidates:
            score = self._score_candidate(request.brief, candidate)
            scored.append(score)

        scored.sort(key=lambda c: c.overall_score, reverse=True)
        top_k = scored[: request.top_k]

        elapsed = (time.perf_counter() - start) * 1000

        return DecisionResponse(
            request_id=request_id,
            ranked_candidates=top_k,
            best_candidate_id=top_k[0].candidate_id if top_k else "",
            model_used=self._model.model_id,
            inference_time_ms=round(elapsed, 2),
        )

    def validate_layout(self, request: ValidationRequest) -> ValidationResponse:
        """Validate a single layout candidate."""
        violations: list[str] = []
        warnings: list[str] = []
        scores: dict[str, float] = {}

        brief = request.brief
        candidate = request.candidate

        violations.extend(self._check_boundaries(brief.site, candidate))
        violations.extend(self._check_overlaps(candidate))

        adjacency_score, adjacency_violations = self._check_adjacency(brief, candidate)
        scores["adjacency"] = adjacency_score
        violations.extend(adjacency_violations)

        area_score, area_warnings = self._check_areas(brief, candidate)
        scores["area"] = area_score
        warnings.extend(area_warnings)

        zoning_score, zoning_violations = self._check_zoning(brief, candidate)
        scores["zoning"] = zoning_score
        violations.extend(zoning_violations)

        scores["model"] = self._model_score(brief, candidate)

        return ValidationResponse(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            scores=scores,
        )

    def decide(self, request: DecideRequest) -> DecideResponse:
        """Return calibrated probabilities for every candidate in one batch.

        This is the System One path: no ranking heuristics on our side, just
        a state per candidate and the caller's typed questions. Inference
        failures propagate (the HTTP layer maps them to 503) so a caller
        never mistakes a fallback constant for a real probability.
        """
        start = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]
        questions = request.questions or DEFAULT_DECIDE_QUESTIONS

        states = [self._build_model_state(request.brief, c) for c in request.candidates]
        results = self._model.predict_batch(states, questions)

        decisions = [
            CandidateDecision(
                candidate_id=candidate.candidate_id,
                model_score=self._answers_to_score(result.get("answers", {}), questions),
                answers=result.get("answers", {}),
            )
            for candidate, result in zip(request.candidates, results, strict=True)
        ]
        decisions.sort(key=lambda d: d.model_score, reverse=True)

        elapsed = (time.perf_counter() - start) * 1000
        return DecideResponse(
            request_id=request_id,
            decisions=decisions,
            model_used=self._model.model_id,
            inference_time_ms=round(elapsed, 2),
        )

    @staticmethod
    def _answers_to_score(answers: dict, questions: dict) -> float:
        """Collapse typed answers to one 0..1 scalar: prefer a normalized
        score answer (expected criteria index / (k-1)), else the mean noul
        probability, else neutral 0.5."""
        for qid, answer in answers.items():
            if not isinstance(answer, dict) or answer.get("type") != "score":
                continue
            criteria = questions.get(qid, {}).get("criteria") or answer.get("legend") or {}
            k = len(criteria)
            if k > 1 and "score" in answer:
                return max(0.0, min(1.0, float(answer["score"]) / (k - 1)))

        nouls = [
            float(answer["noul"])
            for answer in answers.values()
            if isinstance(answer, dict) and "noul" in answer
        ]
        if nouls:
            return max(0.0, min(1.0, sum(nouls) / len(nouls)))
        return 0.5

    # ── Private scoring methods ────────────────────────────────────────────

    def _score_candidate(
        self,
        brief: ProjectBrief,
        candidate: LayoutCandidate,
    ) -> CandidateScore:
        """Score a single candidate layout."""
        room_decisions = [self._score_room(room, brief.site, candidate) for room in brief.rooms]

        adjacency_score = self._compute_adjacency_score(brief, candidate)
        zoning_score = self._compute_zoning_score(brief, candidate)
        area_score = self._compute_area_score(brief, candidate)
        model_score = self._model_score(brief, candidate)

        overall = (
            0.30 * adjacency_score
            + 0.25 * zoning_score
            + 0.20 * area_score
            + 0.25 * model_score
        )

        all_violations: list[str] = []
        for rd in room_decisions:
            all_violations.extend(rd.violations)

        return CandidateScore(
            candidate_id=candidate.candidate_id,
            overall_score=round(overall, 4),
            room_decisions=room_decisions,
            adjacency_score=round(adjacency_score, 4),
            zoning_score=round(zoning_score, 4),
            area_score=round(area_score, 4),
            violations=all_violations,
        )

    def _score_room(
        self,
        room: RoomSpec,
        site: SiteBrief,
        candidate: LayoutCandidate,
    ) -> RoomDecision:
        """Score a single room's placement."""
        placement = candidate.room_placements.get(room.id)
        if placement is None:
            return RoomDecision(
                room_id=room.id,
                score=0.0,
                reasoning="Room not placed in layout",
                violations=[f"Room '{room.id}' has no placement"],
            )

        violations: list[str] = []
        score = 1.0

        x, y = placement.get("x", 0), placement.get("y", 0)
        w, h = placement.get("w", 0), placement.get("h", 0)

        effective_width = site.width_m - site.setbacks_m.get("east", 0) - site.setbacks_m.get("west", 0)
        effective_depth = site.depth_m - site.setbacks_m.get("north", 0) - site.setbacks_m.get("south", 0)

        if x < 0 or y < 0 or x + w > effective_width or y + h > effective_depth:
            violations.append(f"Room '{room.id}' exceeds site boundaries")
            score -= 0.3

        actual_area = w * h
        target_area = room.target_area_m2
        area_ratio = actual_area / target_area if target_area > 0 else 0
        if area_ratio < 0.8:
            violations.append(f"Room '{room.id}' area {actual_area:.1f}m² is below target {target_area:.1f}m²")
            score -= 0.2
        elif area_ratio > 1.5:
            violations.append(f"Room '{room.id}' area {actual_area:.1f}m² exceeds target {target_area:.1f}m²")
            score -= 0.1

        if room.min_dimension_m and (w < room.min_dimension_m or h < room.min_dimension_m):
            violations.append(f"Room '{room.id}' dimension below minimum {room.min_dimension_m}m")
            score -= 0.15

        if room.max_dimension_m and (w > room.max_dimension_m or h > room.max_dimension_m):
            violations.append(f"Room '{room.id}' dimension exceeds maximum {room.max_dimension_m}m")
            score -= 0.15

        if room.aspect_ratio and h > 0:
            actual_ratio = w / h
            ratio_diff = abs(actual_ratio - room.aspect_ratio) / room.aspect_ratio
            if ratio_diff > 0.3:
                violations.append(f"Room '{room.id}' aspect ratio {actual_ratio:.2f} deviates from preferred {room.aspect_ratio:.2f}")
                score -= 0.1

        score = max(0.0, min(1.0, score))

        reasoning = f"Area ratio: {area_ratio:.2f}, Dimensions: {w:.1f}x{h:.1f}m"
        if violations:
            reasoning += f", {len(violations)} violation(s)"

        return RoomDecision(
            room_id=room.id,
            score=round(score, 4),
            reasoning=reasoning,
            violations=violations,
        )

    def _compute_adjacency_score(
        self, brief: ProjectBrief, candidate: LayoutCandidate
    ) -> float:
        """Compute adjacency satisfaction score."""
        if not brief.rooms:
            return 1.0

        total_weight = 0
        satisfied_weight = 0

        for room in brief.rooms:
            for adj_id in room.must_be_adjacent:
                total_weight += 3.0
                if self._are_rooms_adjacent(room.id, adj_id, candidate):
                    satisfied_weight += 3.0

            for adj_id in room.preferred_adjacent:
                total_weight += 1.0
                if self._are_rooms_adjacent(room.id, adj_id, candidate):
                    satisfied_weight += 1.0

            for adj_id in room.must_not_be_adjacent:
                total_weight += 2.0
                if not self._are_rooms_adjacent(room.id, adj_id, candidate):
                    satisfied_weight += 2.0

        return satisfied_weight / total_weight if total_weight > 0 else 1.0

    def _compute_zoning_score(
        self, brief: ProjectBrief, candidate: LayoutCandidate
    ) -> float:
        """Compute zoning compatibility score."""
        if not brief.rooms:
            return 1.0

        violations = 0
        total_checks = 0

        for room in brief.rooms:
            for other in brief.rooms:
                if room.id >= other.id:
                    continue
                total_checks += 1
                pair = (room.zone_type, other.zone_type)
                if pair in INCOMPATIBLE_ZONES or (pair[1], pair[0]) in INCOMPATIBLE_ZONES:
                    if self._are_rooms_adjacent(room.id, other.id, candidate):
                        violations += 1

        return max(0.0, 1.0 - (violations / total_checks)) if total_checks > 0 else 1.0

    def _compute_area_score(
        self, brief: ProjectBrief, candidate: LayoutCandidate
    ) -> float:
        """Compute area utilization score."""
        if not brief.rooms:
            return 1.0

        total_target = sum(r.target_area_m2 for r in brief.rooms)
        if total_target == 0:
            return 1.0

        total_placed = sum(
            p.get("w", 0) * p.get("h", 0)
            for p in candidate.room_placements.values()
        )

        ratio = total_placed / total_target
        if ratio < 0.5:
            return ratio * 2
        elif ratio <= 1.2:
            return 1.0
        else:
            return max(0.0, 1.0 - (ratio - 1.2) * 2)

    def _model_score(self, brief: ProjectBrief, candidate: LayoutCandidate) -> float:
        """Get a score from the Laya model."""
        state = self._build_model_state(brief, candidate)
        question = {
            "type": "score",
            "instructions": "How good is this spatial layout?",
            "criteria": ["poor", "below average", "average", "good", "excellent"],
        }
        try:
            return self._model.score(state, question)
        except Exception as exc:
            logger.warning("Model inference failed: %s", exc)
            return 0.5

    def _build_model_state(self, brief: ProjectBrief, candidate: LayoutCandidate) -> str:
        """Build a text state description for the model.

        The state carries both the data (site, placements) and the logic the
        layout must satisfy (the brief's adjacency requirements), so one
        forward pass can judge how well the arrangement meets the programme
        rather than only how tidy the geometry looks.
        """
        lines = [
            f"Site: {brief.site.width_m}m x {brief.site.depth_m}m",
            f"Rooms: {len(brief.rooms)}",
        ]
        requirements = self._requirement_lines(brief)
        if requirements:
            lines.append("Requirements:")
            lines.extend(requirements)
        lines.append("Layout:")
        for room in brief.rooms:
            placement = candidate.room_placements.get(room.id)
            if placement:
                lines.append(
                    f"  {room.name} ({room.zone_type.value}): "
                    f"({placement.get('x', 0):.1f}, {placement.get('y', 0):.1f}) "
                    f"{placement.get('w', 0):.1f}m x {placement.get('h', 0):.1f}m"
                )
            else:
                lines.append(f"  {room.name} ({room.zone_type.value}): NOT PLACED")
        return "\n".join(lines)

    @staticmethod
    def _requirement_lines(brief: ProjectBrief) -> list[str]:
        """One line per stated adjacency requirement, room ids named as rooms."""
        names = {room.id: room.name for room in brief.rooms}
        lines: list[str] = []
        for room in brief.rooms:
            label = names.get(room.id, room.id)
            for target in room.must_be_adjacent:
                lines.append(f"  {label} must adjoin {names.get(target, target)}")
            for target in room.preferred_adjacent:
                lines.append(f"  {label} should adjoin {names.get(target, target)}")
            for target in room.must_not_be_adjacent:
                lines.append(f"  {label} must NOT adjoin {names.get(target, target)}")
        return lines

    # ── Private validation methods ─────────────────────────────────────────

    def _check_boundaries(
        self, site: SiteBrief, candidate: LayoutCandidate
    ) -> list[str]:
        """Check if all rooms are within site boundaries."""
        violations: list[str] = []
        effective_width = site.width_m - site.setbacks_m.get("east", 0) - site.setbacks_m.get("west", 0)
        effective_depth = site.depth_m - site.setbacks_m.get("north", 0) - site.setbacks_m.get("south", 0)

        for room_id, placement in candidate.room_placements.items():
            x, y = placement.get("x", 0), placement.get("y", 0)
            w, h = placement.get("w", 0), placement.get("h", 0)

            if x < 0:
                violations.append(f"Room '{room_id}' extends past west boundary")
            if y < 0:
                violations.append(f"Room '{room_id}' extends past north boundary")
            if x + w > effective_width:
                violations.append(f"Room '{room_id}' extends past east boundary")
            if y + h > effective_depth:
                violations.append(f"Room '{room_id}' extends past south boundary")

        return violations

    def _check_overlaps(self, candidate: LayoutCandidate) -> list[str]:
        """Check for overlapping room placements."""
        violations: list[str] = []
        placements = list(candidate.room_placements.items())

        for i, (id_a, pos_a) in enumerate(placements):
            for id_b, pos_b in placements[i + 1 :]:
                if self._rects_overlap(pos_a, pos_b):
                    violations.append(f"Rooms '{id_a}' and '{id_b}' overlap")

        return violations

    def _check_adjacency(
        self, brief: ProjectBrief, candidate: LayoutCandidate
    ) -> tuple[float, list[str]]:
        """Check adjacency constraints."""
        violations: list[str] = []
        total = 0
        satisfied = 0

        for room in brief.rooms:
            for adj_id in room.must_be_adjacent:
                total += 1
                if self._are_rooms_adjacent(room.id, adj_id, candidate):
                    satisfied += 1
                else:
                    violations.append(f"Room '{room.id}' must be adjacent to '{adj_id}' but is not")

            for adj_id in room.must_not_be_adjacent:
                total += 1
                if not self._are_rooms_adjacent(room.id, adj_id, candidate):
                    satisfied += 1
                else:
                    violations.append(f"Room '{room.id}' must NOT be adjacent to '{adj_id}' but is")

        return (satisfied / total if total > 0 else 1.0), violations

    def _check_areas(
        self, brief: ProjectBrief, candidate: LayoutCandidate
    ) -> tuple[float, list[str]]:
        """Check area constraints."""
        warnings: list[str] = []
        if not brief.rooms:
            return 1.0, warnings

        scores: list[float] = []
        for room in brief.rooms:
            placement = candidate.room_placements.get(room.id)
            if not placement:
                scores.append(0.0)
                continue

            actual = placement.get("w", 0) * placement.get("h", 0)
            target = room.target_area_m2
            ratio = actual / target if target > 0 else 0

            if ratio < 0.5:
                warnings.append(f"Room '{room.id}' is significantly undersized ({actual:.1f}m² vs {target:.1f}m² target)")
            elif ratio > 2.0:
                warnings.append(f"Room '{room.id}' is significantly oversized ({actual:.1f}m² vs {target:.1f}m² target)")

            scores.append(max(0.0, 1.0 - abs(1.0 - ratio)))

        return (sum(scores) / len(scores) if scores else 1.0), warnings

    def _check_zoning(
        self, brief: ProjectBrief, candidate: LayoutCandidate
    ) -> tuple[float, list[str]]:
        """Check zoning compatibility."""
        violations: list[str] = []

        for room in brief.rooms:
            for other in brief.rooms:
                if room.id >= other.id:
                    continue
                pair = (room.zone_type, other.zone_type)
                if pair in INCOMPATIBLE_ZONES or (pair[1], pair[0]) in INCOMPATIBLE_ZONES:
                    if self._are_rooms_adjacent(room.id, other.id, candidate):
                        violations.append(
                            f"Incompatible zoning: '{room.name}' ({room.zone_type.value}) "
                            f"is adjacent to '{other.name}' ({other.zone_type.value})"
                        )

        return (1.0 if not violations else max(0.0, 1.0 - len(violations) * 0.2)), violations

    # ── Geometry helpers ───────────────────────────────────────────────────

    def _are_rooms_adjacent(
        self, room_a: str, room_b: str, candidate: LayoutCandidate
    ) -> bool:
        """Check if two rooms are adjacent (share a wall or are very close)."""
        pos_a = candidate.room_placements.get(room_a)
        pos_b = candidate.room_placements.get(room_b)
        if not pos_a or not pos_b:
            return False

        tolerance = 0.5

        a_left, a_right = pos_a.get("x", 0), pos_a.get("x", 0) + pos_a.get("w", 0)
        a_top, a_bottom = pos_a.get("y", 0), pos_a.get("y", 0) + pos_a.get("h", 0)
        b_left, b_right = pos_b.get("x", 0), pos_b.get("x", 0) + pos_b.get("w", 0)
        b_top, b_bottom = pos_b.get("y", 0), pos_b.get("y", 0) + pos_b.get("h", 0)

        horizontal_adj = (
            abs(a_right - b_left) <= tolerance or abs(b_right - a_left) <= tolerance
        ) and (a_top < b_bottom and a_bottom > b_top)

        vertical_adj = (
            abs(a_bottom - b_top) <= tolerance or abs(b_bottom - a_top) <= tolerance
        ) and (a_left < b_right and a_right > b_left)

        return horizontal_adj or vertical_adj

    def _rects_overlap(self, a: dict[str, float], b: dict[str, float]) -> bool:
        """Check if two rectangles overlap."""
        a_left, a_right = a.get("x", 0), a.get("x", 0) + a.get("w", 0)
        a_top, a_bottom = a.get("y", 0), a.get("y", 0) + a.get("h", 0)
        b_left, b_right = b.get("x", 0), b.get("x", 0) + b.get("w", 0)
        b_top, b_bottom = b.get("y", 0), b.get("y", 0) + b.get("h", 0)

        return a_left < b_right and a_right > b_left and a_top < b_bottom and a_bottom > b_top


# Singleton
_engine: DecisionEngine | None = None


def get_engine() -> DecisionEngine:
    """Get or create the singleton decision engine."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine
