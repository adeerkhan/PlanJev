"""Tests for the batched probability endpoint (`/api/v1/decide`)."""

from __future__ import annotations

import os

import pytest

from engine.decision_engine import DEFAULT_DECIDE_QUESTIONS, DecisionEngine
from engine.model_loader import (
    DeciderModel,
    LayaModel,
    _apply_cache_dir,
    get_model,
)
from engine.schemas import (
    DecideRequest,
    LayoutCandidate,
    ProjectBrief,
    RoomSpec,
    SiteBrief,
    ZoneType,
)


@pytest.fixture
def brief() -> ProjectBrief:
    return ProjectBrief(
        project_id="decide-001",
        site=SiteBrief(width_m=10.0, depth_m=12.0),
        rooms=[
            RoomSpec(
                id="living",
                name="Living Room",
                zone_type=ZoneType.RESIDENTIAL,
                target_area_m2=20.0,
            ),
            RoomSpec(
                id="kitchen",
                name="Kitchen",
                zone_type=ZoneType.RESIDENTIAL,
                target_area_m2=10.0,
            ),
        ],
    )


@pytest.fixture
def candidates() -> list[LayoutCandidate]:
    return [
        LayoutCandidate(
            candidate_id="good",
            room_placements={
                "living": {"x": 0.0, "y": 0.0, "w": 5.0, "h": 4.0},
                "kitchen": {"x": 5.0, "y": 0.0, "w": 4.0, "h": 2.5},
            },
        ),
        LayoutCandidate(
            candidate_id="weak",
            room_placements={
                "living": {"x": 0.0, "y": 0.0, "w": 3.0, "h": 3.0},
                "kitchen": {"x": 3.0, "y": 0.0, "w": 2.0, "h": 2.0},
            },
        ),
    ]


@pytest.fixture
def fake_batch(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch the batched model call; record what it received."""
    calls: list[dict] = []

    def _predict_batch(self: LayaModel, states: list[str], questions: dict) -> list[dict]:
        calls.append({"states": states, "questions": questions})
        # First state looks excellent, second mediocre — the ranking under test.
        answers_per_state = [
            {
                "layout_quality": {"type": "score", "score": 3.6, "confidence": 0.9,
                                   "legend": {"0": "poor", "1": "below average", "2": "average",
                                              "3": "good", "4": "excellent"}},
                "is_buildable": {"type": "noul", "noul": 0.97, "confidence": 0.97},
            },
            {
                "layout_quality": {"type": "score", "score": 1.2, "confidence": 0.8,
                                   "legend": {"0": "poor", "1": "below average", "2": "average",
                                              "3": "good", "4": "excellent"}},
                "is_buildable": {"type": "noul", "noul": 0.41, "confidence": 0.59},
            },
        ]
        return [{"answers": answers_per_state[i % len(answers_per_state)]} for i in range(len(states))]

    monkeypatch.setattr(LayaModel, "predict_batch", _predict_batch)
    return calls


class TestDecide:
    def test_returns_probability_per_candidate(
        self, brief: ProjectBrief, candidates: list[LayoutCandidate], fake_batch: list[dict]
    ) -> None:
        response = DecisionEngine().decide(
            DecideRequest(brief=brief, candidates=candidates)
        )
        assert response.model_used == "convaiinnovations/laya"
        assert {d.candidate_id for d in response.decisions} == {"good", "weak"}
        for decision in response.decisions:
            assert 0.0 <= decision.model_score <= 1.0
            assert decision.answers["layout_quality"]["type"] == "score"
            assert decision.answers["is_buildable"]["noul"] in (0.97, 0.41)

    def test_ranked_best_first(
        self, brief: ProjectBrief, candidates: list[LayoutCandidate], fake_batch: list[dict]
    ) -> None:
        response = DecisionEngine().decide(
            DecideRequest(brief=brief, candidates=candidates)
        )
        # score 3.6 / (5 - 1) = 0.9 beats 1.2 / 4 = 0.3
        assert response.decisions[0].candidate_id == "good"
        assert response.decisions[0].model_score == pytest.approx(0.9)
        assert response.decisions[1].model_score == pytest.approx(0.3)

    def test_questions_are_passed_through(
        self, brief: ProjectBrief, candidates: list[LayoutCandidate], fake_batch: list[dict]
    ) -> None:
        custom = {
            "has_daylight": {"type": "noul", "instructions": "Habitable rooms get daylight"},
        }
        DecisionEngine().decide(
            DecideRequest(brief=brief, candidates=candidates, questions=custom)
        )
        assert fake_batch[0]["questions"] == custom
        # One state per candidate, and the state names the rooms.
        assert len(fake_batch[0]["states"]) == 2
        assert "Living Room" in fake_batch[0]["states"][0]

    def test_default_questions_cover_score_and_noul(
        self, brief: ProjectBrief, candidates: list[LayoutCandidate], fake_batch: list[dict]
    ) -> None:
        DecisionEngine().decide(DecideRequest(brief=brief, candidates=candidates))
        assert fake_batch[0]["questions"] == DEFAULT_DECIDE_QUESTIONS

    def test_answers_to_score_noul_mean(self) -> None:
        answers = {"a": {"type": "noul", "noul": 0.8}, "b": {"type": "noul", "noul": 0.4}}
        assert DecisionEngine._answers_to_score(answers, {}) == pytest.approx(0.6)

    def test_answers_to_score_neutral_when_empty(self) -> None:
        assert DecisionEngine._answers_to_score({}, {}) == 0.5

    def test_model_failure_propagates(
        self, brief: ProjectBrief, candidates: list[LayoutCandidate], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No fabricated probabilities: a dead model raises, not 0.5."""
        def _boom(self: LayaModel, states: list[str], questions: dict) -> list[dict]:
            raise RuntimeError("model offline")

        monkeypatch.setattr(LayaModel, "predict_batch", _boom)
        with pytest.raises(RuntimeError, match="model offline"):
            DecisionEngine().decide(DecideRequest(brief=brief, candidates=candidates))


class TestModelState:
    def test_state_carries_the_briefs_adjacency_logic(self) -> None:
        """The state must state the rules, not only the geometry — the model
        judges whether the arrangement satisfies the programme."""
        brief = ProjectBrief(
            project_id="state-001",
            site=SiteBrief(width_m=10.0, depth_m=12.0),
            rooms=[
                RoomSpec(
                    id="living", name="Living Room", zone_type=ZoneType.RESIDENTIAL,
                    target_area_m2=24.0, must_be_adjacent=["kitchen"],
                ),
                RoomSpec(
                    id="kitchen", name="Kitchen", zone_type=ZoneType.RESIDENTIAL,
                    target_area_m2=10.0, preferred_adjacent=["living"],
                ),
                RoomSpec(
                    id="bedroom", name="Bedroom", zone_type=ZoneType.RESIDENTIAL,
                    target_area_m2=12.0, must_not_be_adjacent=["kitchen"],
                ),
            ],
        )
        candidate = LayoutCandidate(
            candidate_id="c",
            room_placements={
                "living": {"x": 0.0, "y": 0.0, "w": 5.0, "h": 4.0},
                "kitchen": {"x": 5.0, "y": 0.0, "w": 3.0, "h": 3.0},
                "bedroom": {"x": 0.0, "y": 4.0, "w": 4.0, "h": 3.0},
            },
        )

        state = DecisionEngine()._build_model_state(brief, candidate)

        assert "Requirements:" in state
        assert "Living Room must adjoin Kitchen" in state
        assert "Kitchen should adjoin Living Room" in state
        assert "Bedroom must NOT adjoin Kitchen" in state
        assert "Layout:" in state
        assert "Living Room (residential): (0.0, 0.0) 5.0m x 4.0m" in state


class TestModelCacheDir:
    def test_cache_dir_is_honoured(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))

        _apply_cache_dir()

        assert os.environ["HF_HOME"] == str(tmp_path)

    def test_relative_cache_dir_resolves_against_the_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HF_HUB_CACHE", raising=False)
        monkeypatch.setenv("MODEL_CACHE_DIR", ".model_cache")

        _apply_cache_dir()

        assert os.environ["HF_HOME"].endswith("/backend/.model_cache")

    def test_an_explicit_hf_home_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("HF_HOME", "/explicit/cache")
        monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))

        _apply_cache_dir()

        assert os.environ["HF_HOME"] == "/explicit/cache"


class TestModelSelection:
    """DECISION_MODEL picks the backend; nothing else changes."""

    def test_defaults_to_laya(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DECISION_MODEL", raising=False)
        assert get_model().__class__ is LayaModel

    def test_decider_2b_is_selectable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DECISION_MODEL", "decider-2b")
        assert get_model().__class__ is DeciderModel

    def test_unknown_backend_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DECISION_MODEL", "nope")
        with pytest.raises(ValueError, match="unknown DECISION_MODEL"):
            get_model()

    def test_switching_backends_rebuilds_the_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DECISION_MODEL", "laya")
        first = get_model()
        monkeypatch.setenv("DECISION_MODEL", "decider-2b")
        second = get_model()
        assert first is not second
        assert second.__class__ is DeciderModel


class _FakeDecider:
    """Stands in for decider.infer.Decider so no weights are loaded."""

    def __init__(self) -> None:
        self.schema_calls: list[dict] = []
        self.system_one_calls: list[str] = []
        self.name = "decider-2b-v11"
        self.dev = "cuda"
        # The compiled engine exists only on CUDA; it is what the schema
        # cache needs. Set to None to exercise the CPU path.
        self.eng = object()

    def _answers(self, questions: dict | None = None) -> dict:
        """Answers keyed by the question ids actually asked, as the model's
        own shapes: noul -> {noul}, score -> {score, legend}."""
        questions = questions or DEFAULT_DECIDE_QUESTIONS
        out: dict = {}
        for qid, spec in questions.items():
            if spec.get("type") == "noul":
                out[qid] = {"type": "noul", "noul": 0.9}
            else:
                levels = len(spec.get("criteria", []))
                out[qid] = {
                    "type": "score",
                    "score": 3.0,
                    "confidence": 0.8,
                    "legend": {str(i): f"level {i}" for i in range(levels)},
                    "probabilities": {str(i): 1.0 / max(levels, 1) for i in range(levels)},
                }
        return out

    def schema(self, questions: dict) -> object:
        self.schema_calls.append(questions)
        answers = self._answers(questions)
        fake = self

        class _Schema:
            def batch(self, states: list[str]) -> list[dict]:
                return [
                    {"model": fake.name, "answers": dict(answers)} for _ in states
                ]

        return _Schema()

    def system_one(self, state: str, questions: dict) -> dict:
        self.system_one_calls.append(state)
        return {"model": self.name, "answers": self._answers(questions)}


class TestDeciderModel:
    """decider-2b is a drop-in: same questions in, same answer shapes out."""

    def _model(self, engine: object = object()) -> tuple[DeciderModel, _FakeDecider]:
        fake = _FakeDecider()
        fake.eng = engine
        model = DeciderModel()
        model._decider = fake
        return model, fake

    def test_predict_batch_compiles_the_questions_once(self) -> None:
        model, fake = self._model()

        out = model.predict_batch(["state a", "state b", "state c"], DEFAULT_DECIDE_QUESTIONS)

        assert len(out) == 3
        # The schema is built a single time and reused for every state.
        assert fake.schema_calls == [DEFAULT_DECIDE_QUESTIONS]
        assert out[0]["answers"]["is_buildable"]["noul"] == 0.9

    def test_cpu_path_scores_every_state_on_its_own(self) -> None:
        """schema() needs the compiled engine (CUDA only); without it each
        state takes its own system_one pass and the answers are the same."""
        model, fake = self._model(engine=None)

        out = model.predict_batch(["state a", "state b"], DEFAULT_DECIDE_QUESTIONS)

        assert len(out) == 2
        assert fake.schema_calls == []
        assert fake.system_one_calls == ["state a", "state b"]
        assert out[0]["answers"]["is_buildable"]["noul"] == 0.9

    def test_answers_feed_the_engines_score_collapse(self) -> None:
        """_answers_to_score already reads this shape: 3.0 / (5 - 1)."""
        model, _ = self._model()
        out = model.predict_batch(["state a"], DEFAULT_DECIDE_QUESTIONS)
        assert DecisionEngine._answers_to_score(
            out[0]["answers"], DEFAULT_DECIDE_QUESTIONS
        ) == pytest.approx(0.75)

    def test_predict_scores_a_single_state(self) -> None:
        model, _ = self._model()
        out = model.predict("state a", DEFAULT_DECIDE_QUESTIONS)
        assert out["answers"]["layout_quality"]["type"] == "score"

    def test_score_uses_the_shared_collapse(self) -> None:
        """score() is the same helper for both backends: 3.0 / (5 - 1)."""
        model, _ = self._model()
        question = {
            "type": "score",
            "instructions": "How good is this layout?",
            "criteria": ["poor", "below average", "average", "good", "excellent"],
        }
        assert model.score("state a", question) == pytest.approx(0.75)
