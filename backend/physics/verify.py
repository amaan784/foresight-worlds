"""Verification stage using gemini-decision-1 on conditioning keyframes.

Draws the FITTED physics trajectory outline (circle or rotated box + conditioning path)
onto 3-4 conditioning keyframes (NEVER the 5 s future prediction), then queries
`gemini-decision-1` via `foresight.gemini.client_for(config.DECISION)` (v1beta) and
`foresight.gemini.call(...)` with:
  1. `outline_on_object` (noul): "Does the drawn outline sit on the real object?"
  2. `motion` (choice): keys roll / slide / fall / bounce / other.

Reads and checks `.confidence` (and noul truth) in code against CONFIDENCE_THRESHOLD (0.7).
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from foresight import config, gemini

from .fit import FitResult
from .simulate import evaluate_at_times

OnEvent = Callable[[str, dict[str, Any]], None]

CONFIDENCE_THRESHOLD = 0.7
MAX_KEYFRAMES = 4
VERIFY_MAX_WIDTH = 960

MOTION_CHOICE_CRITERIA: dict[str, str] = {
    "roll": "A round object rolling smoothly along a table, floor, ramp, or support surface.",
    "slide": "An object sliding across a surface without rolling.",
    "fall": "An object falling freely through the air under gravity.",
    "bounce": "An object impacting a surface or obstacle and rebounding.",
    "other": "Any other motion, stationary object, or unclear behavior.",
}


@dataclass
class VerificationResult:
    """Structured result from gemini-decision-1 keyframe verification."""

    confidences: dict[str, float]
    answers: dict[str, str]
    passed: bool
    noul_prob: float
    motion_probabilities: dict[str, float] = field(default_factory=dict)
    overlay_paths: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    def selection_score(self, fit_error_px: float) -> float:
        """Scalar ranking score when comparing multiple fit attempts."""
        outline_score = self.noul_prob if self.answers.get("outline_on_object") == "yes" else (self.noul_prob - 1.0)
        motion_conf = float(self.confidences.get("motion", 0.0))
        return (1.0 if self.passed else 0.0) * 10.0 + 2.0 * outline_score + motion_conf - 0.002 * fit_error_px


def _resolve_existing_path(path_str: str | None, work_dir: Path | None = None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    if work_dir is not None:
        alt = work_dir / p.name
        if alt.exists():
            return alt
    return None


def select_keyframes(
    scene: dict[str, Any],
    t_start_s: float,
    max_frames: int = MAX_KEYFRAMES,
) -> list[dict[str, Any]]:
    """Choose 3-4 existing conditioning keyframes inside the fitted track window."""
    work_dir = Path(scene["work_dir"]) if scene.get("work_dir") else None
    raw_kfs = scene.get("keyframes", [])

    existing: list[dict[str, Any]] = []
    for kf in raw_kfs:
        resolved = _resolve_existing_path(kf.get("png"), work_dir)
        if resolved is not None:
            existing.append({"t_s": float(kf["t_s"]), "png": str(resolved)})

    if not existing:
        last_p = _resolve_existing_path(scene.get("last_frame_png"), work_dir)
        if last_p is not None:
            existing.append({
                "t_s": float(scene.get("t_cond_end_s", t_start_s)),
                "png": str(last_p),
            })

    existing.sort(key=lambda item: item["t_s"])
    in_window = [kf for kf in existing if kf["t_s"] >= t_start_s - 1e-3]
    chosen_pool = in_window if in_window else existing
    if len(chosen_pool) <= max_frames:
        return chosen_pool

    indices = np.linspace(0, len(chosen_pool) - 1, max_frames).round().astype(int).tolist()
    seen: set[int] = set()
    selected: list[dict[str, Any]] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            selected.append(chosen_pool[idx])
    return selected


def _draw_object_outline(
    img: np.ndarray,
    obj: dict[str, Any],
    pose: dict[str, Any],
    scale_x: float,
    scale_y: float,
) -> None:
    """Draw a high-contrast fitted object outline (circle or rotated box) on `img`."""
    cx = int(round(float(pose["x"]) * scale_x))
    cy = int(round(float(pose["y"]) * scale_y))
    angle = float(pose.get("angle", 0.0))
    color_outer = (0, 0, 0)
    color_inner = (0, 255, 60)  # bright neon green

    if obj.get("shape") == "box":
        w = float(obj.get("w_px") or 40.0) * scale_x
        h = float(obj.get("h_px") or 40.0) * scale_y
        # Positive angle in y-up rotates CCW in image space -> (x*cos + y*sin, -x*sin + y*cos)
        ca, sa = math.cos(angle), math.sin(angle)
        hw, hh = 0.5 * w, 0.5 * h
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            rx = dx * ca + dy * sa
            ry = -dx * sa + dy * ca
            corners.append([int(round(cx + rx)), int(round(cy + ry))])
        pts = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color_outer, 6, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color_inner, 3, cv2.LINE_AA)
    else:
        r = max(4, int(round(float(obj.get("radius_px") or 20.0) * 0.5 * (scale_x + scale_y))))
        cv2.circle(img, (cx, cy), r, color_outer, 6, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, color_inner, 3, cv2.LINE_AA)

    cv2.drawMarker(img, (cx, cy), color_outer, cv2.MARKER_CROSS, 16, 4, cv2.LINE_AA)
    cv2.drawMarker(img, (cx, cy), color_inner, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)


def render_verification_overlays(
    scene: dict[str, Any],
    fit_result: FitResult,
    *,
    attempt: int = 0,
    max_width: int = VERIFY_MAX_WIDTH,
) -> list[dict[str, Any]]:
    """Draw the fitted trajectory outline at each conditioning keyframe time.

    Returns list of {"t_s", "path", "jpeg_b64"} ready for Decision 1 and Carbon repair.
    """
    keyframes = select_keyframes(scene, fit_result.t_start_s)
    if not keyframes:
        return []

    orig_w, orig_h = int(scene["orig_size"][0]), int(scene["orig_size"][1])
    kf_times = [kf["t_s"] for kf in keyframes]
    kf_states = evaluate_at_times(
        scene, fit_result.params, fit_result.t_start_s, kf_times, fit_result.motion_model
    )
    state_by_t = {round(s["t_s"], 6): s["objects"] for s in kf_states}

    # Also build dense conditioning polyline over [t_start_s, max(kf_times)]
    t_end_kf = max(kf_times)
    if t_end_kf > fit_result.t_start_s + 1e-4:
        dense_ts = np.linspace(fit_result.t_start_s, t_end_kf, 20).tolist()
        dense_states = evaluate_at_times(
            scene, fit_result.params, fit_result.t_start_s, dense_ts, fit_result.motion_model
        )
    else:
        dense_states = kf_states

    work_dir = Path(scene.get("work_dir", "."))
    work_dir.mkdir(parents=True, exist_ok=True)
    obj_by_id = {o["id"]: o for o in scene.get("objects", [])}

    overlays: list[dict[str, Any]] = []
    for idx, kf in enumerate(keyframes):
        img = cv2.imread(kf["png"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        if iw > max_width:
            tw = max_width
            th = max(2, int(round(ih * (tw / float(iw)))))
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            ih, iw = img.shape[:2]

        sx = iw / float(orig_w)
        sy = ih / float(orig_h)

        # Draw fitted path polyline up to this keyframe time
        for oid in fit_result.params:
            pts = [
                (int(round(st["objects"][oid]["x"] * sx)), int(round(st["objects"][oid]["y"] * sy)))
                for st in dense_states
                if st["t_s"] <= kf["t_s"] + 1e-4 and oid in st["objects"]
            ]
            if len(pts) >= 2:
                arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [arr], False, (255, 220, 0), 2, cv2.LINE_AA)

        # Draw fitted outline at exact keyframe timestamp
        objs_at_t = state_by_t.get(round(kf["t_s"], 6), {})
        for oid, pose in objs_at_t.items():
            if oid in obj_by_id:
                _draw_object_outline(img, obj_by_id[oid], pose, sx, sy)

        label = f"KF {idx + 1}/{len(keyframes)}  t={kf['t_s']:.3f}s  model={fit_result.motion_model}"
        cv2.putText(img, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 60), 2, cv2.LINE_AA)

        out_path = work_dir / f"verify_a{attempt}_kf{idx}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            continue
        out_path.write_bytes(buf.tobytes())
        overlays.append({
            "t_s": kf["t_s"],
            "path": str(out_path),
            "jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        })
    return overlays


def build_decision_request_payload(
    scene: dict[str, Any],
    fit_result: FitResult,
    overlays: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build SDK-validated `input` and `questions` payloads for `decisions.create`."""
    obj_names = ", ".join(o.get("name", o["id"]) for o in scene.get("objects", [])) or "moving object"

    input_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"You are verifying a 2D physics fit on {len(overlays)} sequential video keyframes "
                f"from the conditioning window. Target moving object(s): {obj_names}. "
                f"Fitted motion model: '{fit_result.motion_model}'. "
                "On each frame, the bright green circle/box outline with a center cross shows "
                "where the fitted physics model predicts the object is at that exact timestamp, "
                "and the cyan curve shows the fitted path up to that frame."
            ),
        }
    ]
    for idx, ov in enumerate(overlays):
        input_blocks.append({"type": "text", "text": f"Keyframe {idx + 1} (t = {ov['t_s']:.3f} s):"})
        input_blocks.append({
            "type": "image",
            "data": ov["jpeg_b64"],
            "mime_type": "image/jpeg",
        })

    questions: dict[str, dict[str, Any]] = {
        "outline_on_object": {
            "type": "noul",
            "instructions": (
                f"Does the drawn bright green outline sit on the real moving object ({obj_names}) "
                "across these keyframes?"
            ),
            "criteria": {
                "true_description": (
                    "The drawn green outline sits directly on the real object in the keyframes "
                    "(minor alignment offsets of a few pixels are acceptable)."
                ),
                "false_description": (
                    "The drawn green outline clearly misses, lags behind, leads ahead of, "
                    "or sits off the real object in one or more keyframes."
                ),
            },
        },
        "motion": {
            "type": "choice",
            "instructions": (
                f"Across these sequential keyframes, which motion best describes how the real "
                f"moving object ({obj_names}) is moving?"
            ),
            "criteria": dict(MOTION_CHOICE_CRITERIA),
        },
    }
    return input_blocks, questions


def parse_decision_response(
    resp: Any,
    motion_model: str,
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    overlay_paths: list[str] | None = None,
) -> VerificationResult:
    """Parse `CreateDecisionResponse` and evaluate `.confidence` and answer gates."""
    answers_map = getattr(resp, "answers", None) or {}
    outline_ans_obj = answers_map.get("outline_on_object")
    motion_ans_obj = answers_map.get("motion")

    if outline_ans_obj is None or motion_ans_obj is None:
        raise RuntimeError(f"Decision 1 response missing expected keys: got {list(answers_map.keys())}")

    # NoulAnswer: .noul = P(true) in [0.01, 0.99], .confidence = max(noul, 1 - noul)
    noul_prob = float(getattr(outline_ans_obj, "noul", 0.0))
    outline_conf = float(getattr(outline_ans_obj, "confidence", max(noul_prob, 1.0 - noul_prob)))
    outline_label = "yes" if noul_prob >= 0.5 else "no"

    # ChoiceAnswer: .choice (str), .confidence (float), .probabilities (dict[str, float])
    motion_choice = str(getattr(motion_ans_obj, "choice", "other"))
    motion_conf = float(getattr(motion_ans_obj, "confidence", 0.0))
    motion_probs = {
        str(k): float(v) for k, v in (getattr(motion_ans_obj, "probabilities", None) or {}).items()
    }

    reasons: list[str] = []
    if outline_label != "yes":
        reasons.append(f"outline_on_object={outline_label} (P(true)={noul_prob:.2f})")
    if outline_conf < confidence_threshold:
        reasons.append(f"outline_on_object confidence {outline_conf:.2f} < {confidence_threshold}")
    if motion_conf < confidence_threshold:
        reasons.append(f"motion confidence {motion_conf:.2f} < {confidence_threshold}")
    if motion_choice not in ("other", motion_model) and motion_conf >= confidence_threshold:
        reasons.append(
            f"motion mismatch: Decision 1 chose '{motion_choice}' ({motion_conf:.2f}) vs fitted '{motion_model}'"
        )

    passed = len(reasons) == 0
    return VerificationResult(
        confidences={
            "outline_on_object": round(outline_conf, 4),
            "motion": round(motion_conf, 4),
        },
        answers={
            "outline_on_object": outline_label,
            "motion": motion_choice,
        },
        passed=passed,
        noul_prob=round(noul_prob, 4),
        motion_probabilities=motion_probs,
        overlay_paths=list(overlay_paths or []),
        failure_reasons=reasons,
    )


def verify_fit(
    scene: dict[str, Any],
    fit_result: FitResult,
    *,
    attempt: int = 0,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    on_event: OnEvent | None = None,
) -> VerificationResult:
    """Render fitted trajectory outlines on conditioning keyframes and verify via gemini-decision-1."""
    overlays = render_verification_overlays(scene, fit_result, attempt=attempt)
    if not overlays:
        raise ValueError("No readable keyframes available for Decision 1 verification")

    input_blocks, questions = build_decision_request_payload(scene, fit_result, overlays)
    client = gemini.client_for(config.DECISION)
    clip_id = str(scene.get("clip_id", "clip"))

    resp = gemini.call(
        client.decisions.create,
        kind="decisions.create",
        model=config.DECISION,
        tag=f"verify:{clip_id}:a{attempt}",
        input=input_blocks,
        questions=questions,
    )
    result = parse_decision_response(
        resp,
        fit_result.motion_model,
        confidence_threshold=confidence_threshold,
        overlay_paths=[ov["path"] for ov in overlays],
    )

    if on_event is not None:
        on_event("verify", {
            "model": config.DECISION,
            "attempt": attempt,
            "confidences": result.confidences,
            "answers": result.answers,
            "passed": result.passed,
        })
    return result
