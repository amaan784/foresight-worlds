"""Forward physics simulation from fitted parameters (5.000 s rollout at fps_out)."""
from __future__ import annotations

from typing import Any

from .fit import FitResult
from .world import BodyParams, PhysicsWorld


def evaluate_at_times(
    scene: dict[str, Any],
    params: dict[str, BodyParams | dict[str, Any]],
    t_start_s: float,
    timestamps: list[float],
    motion_model: str | None = None,
) -> list[dict[str, Any]]:
    """Roll out a fitted world and sample pixel-space poses at arbitrary sorted timestamps.

    Each returned item matches CONTRACT.md §2:
      {"t_s": float, "objects": {oid: {"x": float, "y": float, "angle": float, "visible": bool}}}
    Timestamps strictly earlier than `t_start_s` are omitted from stepping (not extrapolated backward).
    """
    if not timestamps:
        return []
    ts_sorted = sorted(float(t) for t in timestamps)
    t0 = float(t_start_s)
    world = PhysicsWorld(scene, params, t_start_s=t0, motion_model=motion_model)

    out: list[dict[str, Any]] = []
    for t in ts_sorted:
        if t + 1e-9 < t0:
            continue
        if t > world.time_s + 1e-12:
            world.step_to(t)
        out.append({
            "t_s": float(t),
            "objects": world.get_state_px(),
        })
    return out


def simulate_future(
    scene: dict[str, Any],
    fit_result: FitResult | dict[str, Any],
    *,
    t_start_s: float | None = None,
    motion_model: str | None = None,
) -> list[dict[str, Any]]:
    """Run exactly `pred_seconds` (default 5.0 s) from `t_cond_end_s` at `fps_out` (default 24 FPS).

    Output frame k in 0..N-1 (120 frames for 5.0 s @ 24 FPS) is evaluated at:
        t_k = t_cond_end_s + k / fps_out
    """
    fps_out = int(scene.get("fps_out", 24))
    t_cond_end_s = float(scene.get("t_cond_end_s", 3.0))
    pred_seconds = float(scene.get("pred_seconds", 5.0))
    n_frames = int(round(pred_seconds * fps_out))

    if isinstance(fit_result, FitResult):
        params = fit_result.params
        t0 = fit_result.t_start_s if t_start_s is None else float(t_start_s)
        model = motion_model or fit_result.motion_model
    else:
        params = fit_result
        t0 = t_cond_end_s if t_start_s is None else float(t_start_s)
        model = motion_model or scene.get("motion_type_hint", "roll")

    timestamps = [t_cond_end_s + (k / fps_out) for k in range(n_frames)]
    if not params:
        return [{"t_s": float(t), "objects": {}} for t in timestamps]

    # Guard against degenerate tracks starting after t_cond_end_s
    t_init = min(t0, t_cond_end_s)
    world = PhysicsWorld(scene, params, t_start_s=t_init, motion_model=model)

    trajectory: list[dict[str, Any]] = []
    for t in timestamps:
        if t > world.time_s + 1e-12:
            world.step_to(t)
        trajectory.append({
            "t_s": float(t),
            "objects": world.get_state_px(),
        })
    return trajectory
