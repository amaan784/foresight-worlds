"""Foresight physics simulation, fitting, rendering, verification, and repair."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .fit import FitResult, fit_scene
from .render import render_trajectory
from .simulate import evaluate_at_times, simulate_future
from .world import BodyParams, CoordTransform, PhysicsWorld, default_priors

OnEvent = Callable[[str, dict[str, Any]], None]


def predict(
    scene: dict[str, Any],
    on_event: OnEvent | None = None,
    *,
    out_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Fit physics to a Foresight scene JSON, simulate 5.000 s, and render MP4."""
    fit_res = fit_scene(scene, attempt=0, on_event=on_event)
    trajectory = simulate_future(scene, fit_res)

    work_dir = Path(scene.get("work_dir", "."))
    work_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = work_dir / "foresight.mp4"

    eff_out_size = out_size
    if eff_out_size is None and scene.get("out_size"):
        eff_out_size = (int(scene["out_size"][0]), int(scene["out_size"][1]))

    rendered = render_trajectory(
        scene, trajectory, out_mp4, out_size=eff_out_size, on_event=on_event
    )

    return {
        "prediction_mp4": str(rendered.resolve()),
        "fit_params": fit_res.fit_params_dict,
        "fit_error_px": fit_res.fit_error_px,
        "motion_model": fit_res.motion_model,
        "decision1_confidences": {},
        "decision1_answers": {},
        "repairs_used": 0,
        "trajectory": trajectory,
        "fit_trajectory": fit_res.fit_trajectory,
    }


__all__ = [
    "BodyParams",
    "CoordTransform",
    "FitResult",
    "PhysicsWorld",
    "default_priors",
    "evaluate_at_times",
    "fit_scene",
    "predict",
    "render_trajectory",
    "simulate_future",
]
