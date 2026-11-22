"""Unit tests for verify.py and repair.py payloads, overlays, plots, and gating (no mocks)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from google.genai._gaos.types.decisions import (
    ChoiceAnswer,
    CreateDecisionRequest,
    CreateDecisionResponse,
    NoulAnswer,
)
from google.genai._gaos.types.interactions import TextResponseFormat

from foresight import config
from physics.fit import fit_scene
from physics.repair import (
    ObjectRepairGuess,
    RepairProposal,
    build_repair_prompt_blocks,
    plot_trajectory_diagnostics,
    proposal_to_fit_overrides,
)
from physics.tests.test_simulate_render import _make_render_scene
from physics.verify import (
    StructuredVerification,
    build_decision_request_payload,
    build_structured_verify_payload,
    parse_decision_response,
    parse_structured_verify_response,
    render_verification_overlays,
)
from physics.world import BodyParams, CoordTransform, PhysicsWorld


def _scene_with_keyframes(tmp_path: Path):
    scene = _make_render_scene(tmp_path)
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(300.0, 750.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=1.2, vy0=0.0, friction=0.5, elasticity=0.65, damping=0.1)
    ts = [2.5 + i * 0.05 for i in range(10)]
    w = PhysicsWorld(scene, {"obj0": true_p}, t_start_s=ts[0], motion_model="roll")

    bg = cv2.imread(scene["background_png"])
    keyframes = []
    for idx, t in enumerate(ts):
        w.step_to(t)
        st = w.get_state_px()["obj0"]
        scene["objects"][0]["track"].append({"t_s": t, "x": st["x"], "y": st["y"]})
        if idx in (0, 3, 6, 9):
            frame = bg.copy()
            cv2.circle(frame, (int(round(st["x"])), int(round(st["y"]))), 50, (80, 220, 190), -1)
            kf_path = tmp_path / f"kf_{idx:03d}.png"
            cv2.imwrite(str(kf_path), frame)
            keyframes.append({"t_s": t, "png": str(kf_path)})
    scene["keyframes"] = keyframes
    return scene


def test_verify_overlays_and_sdk_decision_request_schema(tmp_path: Path) -> None:
    scene = _scene_with_keyframes(tmp_path)
    fit_res = fit_scene(scene)
    overlays = render_verification_overlays(scene, fit_res, attempt=0)

    assert len(overlays) == 4
    for ov in overlays:
        assert Path(ov["path"]).exists()
        assert len(ov["jpeg_b64"]) > 1000

    input_blocks, questions = build_decision_request_payload(scene, fit_res, overlays)
    # Validate strictly against the real google-genai 3.6.1 SDK Pydantic model
    req = CreateDecisionRequest(
        model=config.DECISION,
        input=input_blocks,
        questions=questions,
    )
    assert req.model == config.DECISION
    assert set(req.questions.keys()) == {"outline_on_object", "motion"}
    assert "other" in req.questions["motion"].criteria
    assert set(req.questions["motion"].criteria.keys()) == {"roll", "slide", "fall", "bounce", "other"}


def test_parse_decision_response_confidence_and_answer_gating() -> None:
    # 1. Passing case: outline yes (0.92), motion roll (0.85) matching fitted "roll"
    resp_ok = CreateDecisionResponse(
        answers={
            "outline_on_object": NoulAnswer(noul=0.92, confidence=0.92),
            "motion": ChoiceAnswer(
                choice="roll",
                confidence=0.85,
                probabilities={"roll": 0.85, "slide": 0.05, "fall": 0.03, "bounce": 0.04, "other": 0.03},
            ),
        }
    )
    v_ok = parse_decision_response(resp_ok, "roll")
    assert v_ok.passed is True
    assert v_ok.answers == {"outline_on_object": "yes", "motion": "roll"}
    assert v_ok.confidences == {"outline_on_object": 0.92, "motion": 0.85}

    # 2. Low confidence (< 0.7) fails
    resp_low = CreateDecisionResponse(
        answers={
            "outline_on_object": NoulAnswer(noul=0.62, confidence=0.62),
            "motion": ChoiceAnswer(
                choice="roll",
                confidence=0.80,
                probabilities={"roll": 0.80, "slide": 0.10, "fall": 0.03, "bounce": 0.03, "other": 0.04},
            ),
        }
    )
    v_low = parse_decision_response(resp_low, "roll")
    assert v_low.passed is False
    assert any("confidence" in r for r in v_low.failure_reasons)

    # 3. High-confidence "no" (noul=0.05 -> confidence=0.95) MUST FAIL!
    resp_no = CreateDecisionResponse(
        answers={
            "outline_on_object": NoulAnswer(noul=0.05, confidence=0.95),
            "motion": ChoiceAnswer(
                choice="roll",
                confidence=0.88,
                probabilities={"roll": 0.88, "slide": 0.04, "fall": 0.03, "bounce": 0.02, "other": 0.03},
            ),
        }
    )
    v_no = parse_decision_response(resp_no, "roll")
    assert v_no.passed is False
    assert v_no.answers["outline_on_object"] == "no"

    # 4. Motion mismatch (Decision 1 confident it's "bounce" when fitted "roll") MUST FAIL!
    resp_mm = CreateDecisionResponse(
        answers={
            "outline_on_object": NoulAnswer(noul=0.88, confidence=0.88),
            "motion": ChoiceAnswer(
                choice="bounce",
                confidence=0.90,
                probabilities={"roll": 0.03, "slide": 0.02, "fall": 0.03, "bounce": 0.90, "other": 0.02},
            ),
        }
    )
    v_mm = parse_decision_response(resp_mm, "roll")
    assert v_mm.passed is False
    assert any("mismatch" in r for r in v_mm.failure_reasons)


def test_repair_plot_schema_and_override_clamping(tmp_path: Path) -> None:
    scene = _scene_with_keyframes(tmp_path)
    fit_res = fit_scene(scene)
    overlays = render_verification_overlays(scene, fit_res, attempt=0)

    plot_path = plot_trajectory_diagnostics(scene, fit_res, attempt=1)
    assert plot_path.exists()
    img = cv2.imread(str(plot_path))
    assert img is not None and img.shape[0] > 200 and img.shape[1] > 400

    # Validate RepairProposal JSON schema against SDK TextResponseFormat
    rf = TextResponseFormat(
        type="text",
        mime_type="application/json",
        schema=RepairProposal.model_json_schema(),
    )
    assert rf.mime_type == "application/json"
    assert rf.schema_ is not None

    ver = parse_decision_response(
        CreateDecisionResponse(
            answers={
                "outline_on_object": NoulAnswer(noul=0.40, confidence=0.60),
                "motion": ChoiceAnswer(
                    choice="bounce",
                    confidence=0.75,
                    probabilities={"roll": 0.1, "slide": 0.05, "fall": 0.05, "bounce": 0.75, "other": 0.05},
                ),
            }
        ),
        "roll",
        overlay_paths=[ov["path"] for ov in overlays],
    )
    blocks = build_repair_prompt_blocks(scene, fit_res, ver, plot_path, ver.overlay_paths)
    img_blocks = [b for b in blocks if b["type"] == "image"]
    assert len(img_blocks) == 5  # 1 trajectory plot + 4 keyframes

    prop = RepairProposal(
        motion_model="bounce",
        reason="Ball rebounds after contact",
        object_guesses=[
            ObjectRepairGuess(
                object_id="obj0",
                vx0_m_s=1.5,
                vy0_m_s=-2.0,
                friction=3.0,  # out of bounds -> must clamp
                elasticity=1.5,  # out of bounds -> must clamp
                damping=-1.0,  # out of bounds -> must clamp
            )
        ],
    )
    init_g, priors = proposal_to_fit_overrides(prop)
    assert init_g["obj0"]["friction"] == pytest.approx(1.50)
    assert init_g["obj0"]["elasticity"] == pytest.approx(0.98)
    assert init_g["obj0"]["damping"] == pytest.approx(0.0)
    assert priors["obj0"]["elasticity"] == pytest.approx(0.98)


def test_structured_verify_payload_schema_and_gating(tmp_path: Path) -> None:
    scene = _scene_with_keyframes(tmp_path)
    fit_res = fit_scene(scene)
    overlays = render_verification_overlays(scene, fit_res, attempt=0)

    blocks = build_structured_verify_payload(scene, fit_res, overlays)
    assert sum(1 for b in blocks if b["type"] == "image") == 4
    assert any("outline_on_object" in b.get("text", "") and "other" in b.get("text", "") for b in blocks)

    rf = TextResponseFormat(
        type="text",
        mime_type="application/json",
        schema=StructuredVerification.model_json_schema(),
    )
    assert rf.mime_type == "application/json"
    assert rf.schema_ is not None

    # 1. Passing structured check (>= 0.7 on both, outline='yes', matching motion)
    v_ok = parse_structured_verify_response(
        '{"outline_on_object":"yes","outline_confidence":0.91,"motion":"roll","motion_confidence":0.87}',
        "roll",
    )
    assert v_ok.passed is True
    assert v_ok.answers == {"outline_on_object": "yes", "motion": "roll"}
    assert v_ok.confidences == {"outline_on_object": 0.91, "motion": 0.87}
    assert v_ok.noul_prob == pytest.approx(0.91)

    # 2. Confidence below 0.7 gate fails
    v_low = parse_structured_verify_response(
        '{"outline_on_object":"yes","outline_confidence":0.65,"motion":"roll","motion_confidence":0.88}',
        "roll",
    )
    assert v_low.passed is False
    assert any("confidence" in r for r in v_low.failure_reasons)

    # 3. High-confidence 'no' fails and flips noul_prob to 1 - conf
    v_no = parse_structured_verify_response(
        '{"outline_on_object":"no","outline_confidence":0.94,"motion":"roll","motion_confidence":0.90}',
        "roll",
    )
    assert v_no.passed is False
    assert v_no.answers["outline_on_object"] == "no"
    assert v_no.noul_prob == pytest.approx(0.06)
    assert v_ok.selection_score(5.0) > v_no.selection_score(5.0)

    # 4. Motion mismatch fails; 'other' >= 0.7 passes when outline is 'yes'
    v_mm = parse_structured_verify_response(
        '{"outline_on_object":"yes","outline_confidence":0.89,"motion":"bounce","motion_confidence":0.84}',
        "roll",
    )
    assert v_mm.passed is False
    assert any("mismatch" in r for r in v_mm.failure_reasons)

    v_other = parse_structured_verify_response(
        '{"outline_on_object":"yes","outline_confidence":0.89,"motion":"other","motion_confidence":0.80}',
        "roll",
    )
    assert v_other.passed is True
