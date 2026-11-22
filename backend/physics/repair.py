"""Physics fit repair: fast local multi-start by default, with optional LLM diagnostics behind FORESIGHT_REPAIR_MODE=llm.

When fit verification fails (low confidence < 0.7, outline miss, or motion mismatch):
- Default (`REPAIR_MODE == "local"`): refits locally across alternative motion models (`roll`, `slide`,
  `fall`, `bounce`) and initial velocity / material seeds in under a second, ranking distinct fits by
  RMSE and verifier motion agreement before re-checking at most `MAX_REPAIRS` candidates.
- Optional (`REPAIR_MODE == "llm"`): renders a 2-panel diagnostic trajectory plot and queries
  `config.REPAIR` (`config.REPAIR_THINKING`, native structured output) for a `RepairProposal`.
"""
from __future__ import annotations

import base64
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, List, Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/foresight_mplconfig")

import cv2
import matplotlib
import numpy as np
from pydantic import BaseModel, Field

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foresight import config, gemini  # noqa: E402

from .fit import (  # noqa: E402
    DAMPING_BOUNDS,
    ELASTICITY_BOUNDS,
    FRICTION_BOUNDS,
    FitResult,
    _obj_vertical_half_m,
    _surface_top_y_at_x,
    fit_scene,
)
from .simulate import evaluate_at_times  # noqa: E402
from .verify import CONFIDENCE_THRESHOLD, VerificationResult, render_verification_overlays  # noqa: E402
from .world import CoordTransform  # noqa: E402

OnEvent = Callable[[str, dict[str, Any]], None]

MAX_REPAIRS = 2
LOCAL_REPAIR_MODEL = "local-multistart"


class ObjectRepairGuess(BaseModel):
    """Per-object initial velocity and physical prior overrides proposed by Carbon."""

    object_id: str = Field(description="Object id from the scene JSON, e.g. 'obj0'")
    vx0_m_s: float = Field(description="Initial horizontal velocity in m/s (right positive)")
    vy0_m_s: float = Field(description="Initial vertical velocity in m/s (UP positive)")
    friction: float = Field(description="Friction coefficient prior in [0.02, 1.5]")
    elasticity: float = Field(description="Coefficient of restitution prior in [0.05, 0.98]")
    damping: float = Field(description="Rolling resistance damping rate in [0.0, 5.0] s^-1")


class RepairProposal(BaseModel):
    """Structured repair proposal returned by gemini-4-carbon."""

    motion_model: Literal["roll", "slide", "fall", "bounce", "other"] = Field(
        description="Physics motion model to use for the next fit attempt"
    )
    reason: str = Field(
        description="Short diagnosis of why the previous fit failed verification and what changed"
    )
    object_guesses: List[ObjectRepairGuess] = Field(
        description="Per-object initial velocity and material prior guesses for refitting"
    )


def plot_trajectory_diagnostics(
    scene: dict[str, Any],
    fit_result: FitResult,
    *,
    attempt: int = 1,
) -> Path:
    """Render a 2-panel diagnostic PNG (2D scene view + x/y vs time) for Carbon repair."""
    work_dir = Path(scene.get("work_dir", "."))
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"repair_plot_a{attempt}.png"

    orig_w, orig_h = int(scene["orig_size"][0]), int(scene["orig_size"][1])
    t_cond_end = float(scene.get("t_cond_end_s", 3.0))

    # Sample fitted trajectory over conditioning + 1.5 s preview
    t_end_preview = max(t_cond_end + 1.5, fit_result.t_start_s + 1.0)
    dense_ts = np.linspace(fit_result.t_start_s, t_end_preview, 60).tolist()
    dense_traj = evaluate_at_times(
        scene, fit_result.params, fit_result.t_start_s, dense_ts, fit_result.motion_model
    )

    fig, (ax_xy, ax_t) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # Panel 1: 2D pixel space (y DOWN like the video)
    for surf in scene.get("surfaces", []):
        p0, p1 = surf["p0"], surf["p1"]
        ax_xy.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#8B5A2B", lw=3, label=f"surface: {surf.get('name', 's')}")
    for sobj in scene.get("static_objects", []):
        x0, y0, x1, y1 = sobj["bbox"]
        ax_xy.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#666666", lw=1.5, ls="--"))

    for obj in scene.get("objects", []):
        oid = obj["id"]
        obs = obj.get("track", [])
        if obs:
            ox = [p["x"] for p in obs]
            oy = [p["y"] for p in obs]
            ax_xy.scatter(ox, oy, c="#1f77b4", s=28, zorder=5, label=f"{oid} tracked")
        cond_pts = [
            (s["objects"][oid]["x"], s["objects"][oid]["y"])
            for s in dense_traj
            if s["t_s"] <= t_cond_end + 1e-4 and oid in s["objects"] and s["objects"][oid].get("active", True)
        ]
        fut_pts = [
            (s["objects"][oid]["x"], s["objects"][oid]["y"])
            for s in dense_traj
            if s["t_s"] >= t_cond_end - 1e-4 and oid in s["objects"] and s["objects"][oid].get("active", True)
        ]
        if cond_pts:
            ax_xy.plot([p[0] for p in cond_pts], [p[1] for p in cond_pts], c="#2ca02c", lw=2.5, label=f"{oid} fit")
        if fut_pts:
            ax_xy.plot([p[0] for p in fut_pts], [p[1] for p in fut_pts], c="#ff7f0e", lw=2.0, ls="--", label=f"{oid} pred +1.5s")

    ax_xy.set_xlim(0, orig_w)
    ax_xy.set_ylim(orig_h, 0)  # y DOWN
    ax_xy.set_xlabel("x (px)")
    ax_xy.set_ylabel("y (px, down)")
    ax_xy.set_title(f"2D View — model='{fit_result.motion_model}'  RMSE={fit_result.fit_error_px:.1f} px")
    ax_xy.grid(alpha=0.25)
    ax_xy.legend(loc="upper right", fontsize=8)

    # Panel 2: x(t) and y(t) during conditioning window
    for obj in scene.get("objects", []):
        oid = obj["id"]
        obs = obj.get("track", [])
        if obs:
            ot = [p["t_s"] for p in obs]
            ax_t.scatter(ot, [p["x"] for p in obs], c="#1f77b4", s=24, label=f"{oid} x obs")
            ax_t.scatter(ot, [p["y"] for p in obs], c="#d62728", s=24, marker="x", label=f"{oid} y obs")
        ft = [s["t_s"] for s in fit_result.fit_trajectory if oid in s["objects"]]
        if ft:
            ax_t.plot(ft, [s["objects"][oid]["x"] for s in fit_result.fit_trajectory if oid in s["objects"]],
                      c="#1f77b4", lw=2, label=f"{oid} x fit")
            ax_t.plot(ft, [s["objects"][oid]["y"] for s in fit_result.fit_trajectory if oid in s["objects"]],
                      c="#d62728", lw=2, ls="--", label=f"{oid} y fit")

    ax_t.set_xlabel("t_s (s)")
    ax_t.set_ylabel("position (orig px)")
    ax_t.set_title("Conditioning Track vs Fitted Model")
    ax_t.grid(alpha=0.25)
    ax_t.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)
    return out_path


def _img_file_to_jpeg_b64(path: str | Path, max_width: int = 960) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image for repair prompt: {path}")
    h, w = img.shape[:2]
    if w > max_width:
        nh = max(2, int(round(h * (max_width / float(w)))))
        img = cv2.resize(img, (max_width, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"JPEG encode failed for {path}")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def proposal_to_fit_overrides(
    proposal: RepairProposal,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Convert a Carbon `RepairProposal` into bounded `initial_guess` and `param_priors` dicts."""
    init_guess: dict[str, dict[str, float]] = {}
    priors: dict[str, dict[str, float]] = {}
    for g in proposal.object_guesses:
        vx0 = float(np.clip(g.vx0_m_s, -20.0, 20.0))
        vy0 = float(np.clip(g.vy0_m_s, -20.0, 20.0))
        fric = float(np.clip(g.friction, FRICTION_BOUNDS[0], FRICTION_BOUNDS[1]))
        elas = float(np.clip(g.elasticity, ELASTICITY_BOUNDS[0], ELASTICITY_BOUNDS[1]))
        damp = float(np.clip(g.damping, DAMPING_BOUNDS[0], DAMPING_BOUNDS[1]))
        init_guess[g.object_id] = {
            "vx0": vx0,
            "vy0": vy0,
            "friction": fric,
            "elasticity": elas,
            "damping": damp,
        }
        priors[g.object_id] = {
            "friction": fric,
            "elasticity": elas,
            "damping": damp,
        }
    return init_guess, priors


def build_repair_prompt_blocks(
    scene: dict[str, Any],
    fit_result: FitResult,
    verification: VerificationResult,
    plot_path: Path,
    overlay_paths: list[str],
    previous_attempts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build multimodal input blocks (diagnostic plot + keyframes + state summary) for Carbon."""
    obj_summary = [
        {
            "id": o["id"],
            "name": o.get("name"),
            "shape": o.get("shape"),
            "n_track_pts": len(o.get("track", [])),
            "fitted_params_m": fit_result.fit_params_dict.get(o["id"]),
        }
        for o in scene.get("objects", [])
    ]
    text_summary = (
        "You are the physics repair diagnostician for Foresight, a 2D pymunk rigid-body simulator.\n"
        f"Current motion_model: '{fit_result.motion_model}', fit_error_px: {fit_result.fit_error_px:.2f} px "
        f"(image {scene['orig_size'][0]}x{scene['orig_size'][1]}, px_per_m={scene['px_per_m']}, "
        f"fixed gravity={scene.get('gravity_m_s2', 9.81)} m/s^2).\n"
        f"Decision 1 verification FAILED with answers={verification.answers}, "
        f"confidences={verification.confidences}, reasons={verification.failure_reasons}.\n"
        f"Objects and current fitted parameters (y-UP metres, m/s): {obj_summary}.\n"
    )
    if previous_attempts:
        text_summary += f"Previous repair history (do not repeat failed proposals): {previous_attempts}.\n"
    text_summary += (
        "Inspect the 2-panel trajectory plot and the annotated keyframes (bright green outline = fitted model). "
        "Diagnose what went wrong and propose a better motion_model (roll|slide|fall|bounce|other) and "
        "improved per-object initial velocities (vx0_m_s right+, vy0_m_s UP+) and material priors "
        "(friction, elasticity, damping) for scipy least_squares refitting."
    )

    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": text_summary},
        {"type": "text", "text": "Diagnostic trajectory plot (2D scene view + x/y vs time):"},
        {"type": "image", "data": _img_file_to_jpeg_b64(plot_path, max_width=1200), "mime_type": "image/jpeg"},
    ]
    for idx, p in enumerate(overlay_paths[:4]):
        if Path(p).exists():
            blocks.append({"type": "text", "text": f"Annotated conditioning keyframe {idx + 1}:"})
            blocks.append({"type": "image", "data": _img_file_to_jpeg_b64(p), "mime_type": "image/jpeg"})
    return blocks


def propose_repair(
    scene: dict[str, Any],
    fit_result: FitResult,
    verification: VerificationResult,
    *,
    attempt: int = 1,
    previous_attempts: list[dict[str, Any]] | None = None,
    on_event: OnEvent | None = None,
) -> RepairProposal:
    """Ask config.REPAIR (config.REPAIR_THINKING, native structured output) for a repair proposal."""
    plot_path = plot_trajectory_diagnostics(scene, fit_result, attempt=attempt)
    overlay_paths = list(verification.overlay_paths)
    if not overlay_paths:
        ovs = render_verification_overlays(scene, fit_result, attempt=max(0, attempt - 1))
        overlay_paths = [ov["path"] for ov in ovs]

    blocks = build_repair_prompt_blocks(
        scene, fit_result, verification, plot_path, overlay_paths, previous_attempts
    )

    repair_model = config.REPAIR
    client = gemini.client_for(repair_model)
    clip_id = str(scene.get("clip_id", "clip"))

    resp = gemini.call(
        client.interactions.create,
        kind="interactions.create",
        model=repair_model,
        tag=f"repair:{clip_id}:a{attempt}",
        fallback=True,
        input=blocks,
        generation_config={"thinking_level": config.REPAIR_THINKING},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": RepairProposal.model_json_schema(),
        },
    )

    used_model = getattr(resp, "model", None) or repair_model
    proposal = RepairProposal.model_validate_json(resp.output_text)

    if on_event is not None:
        on_event("repair", {
            "model": used_model,
            "attempt": attempt,
            "proposal": {
                "motion_model": proposal.motion_model,
                "reason": proposal.reason,
            },
        })
    return proposal


def build_multistart_candidates(
    scene: dict[str, Any],
    fit_result: FitResult,
    verification: VerificationResult | None = None,
) -> list[tuple[str, dict[str, dict[str, float]] | None, dict[str, dict[str, float]] | None, str]]:
    """Build deterministic `(motion_model, initial_guess, param_priors, reason)` starts for local refitting."""
    objects = [o for o in scene.get("objects", []) if o.get("track")]
    if not objects:
        return []

    coord = CoordTransform.from_scene(scene)
    g = float(scene.get("gravity_m_s2", 9.81))
    all_boxes = all(o.get("shape") == "box" for o in objects)
    valid_models = ["slide", "fall", "bounce"] if all_boxes else ["roll", "slide", "fall", "bounce"]
    ver_motion = verification.answers.get("motion") if verification else None

    alt_models = [m for m in valid_models if m != fit_result.motion_model]
    if ver_motion in alt_models:
        alt_models = [ver_motion, *[m for m in alt_models if m != ver_motion]]

    specs: list[tuple[str, dict[str, dict[str, float]] | None, dict[str, dict[str, float]] | None, str]] = []
    for m in alt_models:
        note = f"refit with '{m}' dynamics"
        if m == ver_motion:
            note += f" matching verifier '{ver_motion}'"
        specs.append((m, None, None, note))

    bounce_init: dict[str, dict[str, float]] = {}
    bounce_priors: dict[str, dict[str, float]] = {}
    chord_init: dict[str, dict[str, float]] = {}
    chord_priors: dict[str, dict[str, float]] = {}
    fall_init: dict[str, dict[str, float]] = {}
    fall_priors: dict[str, dict[str, float]] = {}

    for obj in objects:
        oid = obj["id"]
        track = sorted(obj.get("track", []), key=lambda p: float(p["t_s"]))
        if len(track) < 2:
            continue
        ts = np.asarray([float(pt["t_s"]) for pt in track], dtype=float)
        pts_m = [coord.px_to_m(float(pt["x"]), float(pt["y"])) for pt in track]
        xs = np.asarray([p[0] for p in pts_m], dtype=float)
        ys = np.asarray([p[1] for p in pts_m], dtype=float)

        dt_all = max(1e-3, float(ts[-1] - ts[0]))
        vx_ch = float(np.clip((xs[-1] - xs[0]) / dt_all, -20.0, 20.0))
        vy_ch = float(np.clip((ys[-1] - ys[0]) / dt_all, -20.0, 20.0))
        chord_init[oid] = {
            "x0": float(xs[0]),
            "y0": float(ys[0]),
            "vx0": vx_ch,
            "vy0": vy_ch,
            "friction": 0.15,
            "damping": 0.0,
        }
        chord_priors[oid] = {"friction": 0.15, "damping": 0.0}

        idx_min = int(np.argmin(ys))
        e_seed = 0.82
        if ys[1] >= ys[0]:
            # Impact occurred between frame 0 and frame 1: infer pre-bounce downward speed from rebound apex height
            half_h = _obj_vertical_half_m(obj, coord)
            surf_ys = _surface_top_y_at_x(scene, coord, float(xs[0]), oid)
            y_contact = (max(surf_ys) + half_h) if surf_ys else float(np.min(ys))
            h_reb = max(0.02, float(np.max(ys)) - y_contact)
            vx_pre = vx_ch
            vy_pre = -math.sqrt(2.0 * g * h_reb) / e_seed
        elif 0 < idx_min < len(ts) and ts[idx_min] > ts[0] + 1e-4:
            dt_imp = float(ts[idx_min] - ts[0])
            vx_pre = float((xs[idx_min] - xs[0]) / dt_imp)
            vy_pre = float(min(-0.35, (ys[idx_min] - ys[0]) / dt_imp + 0.5 * g * dt_imp))
        else:
            dt1 = max(1e-3, float(ts[1] - ts[0]))
            vx_pre = vx_ch
            vy_pre = float(-max(0.8, abs((ys[1] - ys[0]) / dt1)))
        bounce_init[oid] = {
            "x0": float(xs[0]),
            "y0": float(ys[0]),
            "vx0": float(np.clip(vx_pre, -20.0, 20.0)),
            "vy0": float(np.clip(vy_pre, -20.0, 20.0)),
            "elasticity": e_seed,
        }
        bounce_priors[oid] = {"elasticity": e_seed, "damping": 0.02}

        k_end = min(len(ts) - 1, 3)
        dt_e = max(1e-3, float(ts[k_end] - ts[0]))
        vx_e = float(np.clip((xs[k_end] - xs[0]) / dt_e, -20.0, 20.0))
        vy_e = float(np.clip((ys[k_end] - ys[0]) / dt_e + 0.5 * g * dt_e, -20.0, 20.0))
        fall_init[oid] = {
            "x0": float(xs[0]),
            "y0": float(ys[0]),
            "vx0": vx_e,
            "vy0": vy_e,
            "damping": 0.0,
        }
        fall_priors[oid] = {"damping": 0.0}

    if bounce_init:
        bounce_spec = ("bounce", bounce_init, bounce_priors, "pre-impact bounce seed (elasticity 0.82)")
        if ver_motion == "bounce":
            specs.insert(min(1, len(specs)), bounce_spec)
        else:
            specs.append(bounce_spec)
    if chord_init:
        surf_model = "slide" if all_boxes else "roll"
        specs.append((surf_model, chord_init, chord_priors, f"low-friction chord velocity seed ('{surf_model}')"))
    if fall_init:
        specs.append(("fall", fall_init, fall_priors, "early-arc ballistic fall seed"))
    return specs


def local_multistart_repair(
    scene: dict[str, Any],
    fit_result: FitResult,
    verification: VerificationResult | None = None,
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    budget_s: float = 15.0,
) -> list[tuple[FitResult, str]]:
    """Evaluate local multi-start candidates and return distinct plausible fits ordered best-first."""
    specs = build_multistart_candidates(scene, fit_result, verification)
    if not specs:
        return []

    t0 = time.time()
    ver_motion = verification.answers.get("motion") if verification else None
    ver_conf = float((verification.confidences if verification else {}).get("motion", 0.0))
    ver_motion_strong = ver_motion not in (None, "other", fit_result.motion_model) and ver_conf >= confidence_threshold

    by_bucket: dict[tuple[str, int], tuple[FitResult, str]] = {}
    for idx, (model, init_g, priors, reason) in enumerate(specs):
        if idx > 0 and (time.time() - t0) >= budget_s:
            break
        cand = fit_scene(
            scene,
            motion_model=model,
            initial_guess=init_g,
            param_priors=priors,
            on_event=None,
        )
        # Skip duplicate of the attempt-0 fit or severely degraded fits
        if cand.motion_model == fit_result.motion_model and abs(cand.fit_error_px - fit_result.fit_error_px) < 0.25:
            continue
        if cand.fit_error_px > max(35.0, 1.5 * fit_result.fit_error_px) and cand.fit_error_px >= fit_result.fit_error_px:
            continue
        bucket = (cand.motion_model, int(round(cand.fit_error_px / 2.0)))
        prev = by_bucket.get(bucket)
        if prev is None or cand.fit_error_px < prev[0].fit_error_px:
            by_bucket[bucket] = (cand, reason)

    def _rank_key(item: tuple[FitResult, str]) -> tuple[int, float]:
        cand, _ = item
        matches_ver = (
            ver_motion_strong
            and cand.motion_model == ver_motion
            and cand.fit_error_px <= max(25.0, 1.5 * fit_result.fit_error_px)
        )
        return (0 if matches_ver else 1, cand.fit_error_px)

    return sorted(by_bucket.values(), key=_rank_key)
