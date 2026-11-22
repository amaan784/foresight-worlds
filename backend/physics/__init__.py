"""Foresight physics simulation, fitting, rendering, verification, and repair."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable

from foresight import config, gemini

from .fit import FitResult, fit_scene
from .render import render_trajectory
from .repair import MAX_REPAIRS, RepairProposal, proposal_to_fit_overrides, propose_repair
from .simulate import evaluate_at_times, simulate_future
from .verify import CONFIDENCE_THRESHOLD, VerificationResult, select_keyframes, verify_fit
from .world import BodyParams, CoordTransform, PhysicsWorld, default_priors

OnEvent = Callable[[str, dict[str, Any]], None]


def implausible(scene: dict[str, Any], fit: FitResult, trajectory: list[dict[str, Any]]) -> str | None:
    """Why the simulated future cannot be right, or None.

    Contacts and damping only take energy away, so no tracked object may move faster than its starting speed plus
    what gravity can add over the height it has dropped (with 30% + 0.3 m/s slack for frame-to-frame jitter). A
    faster object means the simulation went wrong (bodies started inside each other, a solver blow-up).
    """
    ppm, g = float(scene["px_per_m"]), float(scene.get("gravity_m_s2", 9.81))
    fps = float(scene.get("fps_out", 24))
    for oid, bp in fit.params.items():
        pts = [
            (st["objects"][oid]["x"], st["objects"][oid]["y"])
            for st in trajectory
            if oid in st.get("objects", {}) and st["objects"][oid].get("active", True)
        ]
        if len(pts) < 3:
            continue
        v0 = max((bp.vx0 ** 2 + bp.vy0 ** 2) ** 0.5, ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5 * fps / ppm)
        for k in range(1, len(pts)):
            v = ((pts[k][0] - pts[k - 1][0]) ** 2 + (pts[k][1] - pts[k - 1][1]) ** 2) ** 0.5 * fps / ppm
            drop_m = max(0.0, (pts[k][1] - pts[0][1]) / ppm)  # image y grows downwards
            bound = (v0 ** 2 + 2 * g * drop_m) ** 0.5
            if v > 1.3 * bound + 0.3:
                return (f"{oid} reached {v:.1f} m/s at t={trajectory[k]['t_s']:.2f} s, more than its start speed plus "
                        f"free fall allows ({bound:.1f} m/s)")
    return None


def predict(
    scene: dict[str, Any],
    on_event: OnEvent | None = None,
    *,
    out_size: tuple[int, int] | None = None,
    verify: bool | None = None,
    max_repairs: int = MAX_REPAIRS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    repair_budget_s: float | None = None,
) -> dict[str, Any]:
    """Fit physics to a Foresight scene JSON, verify/repair via Gemini, simulate 5.000 s, render MP4.

    Verification via `gemini-decision-1` and repair via `gemini-4-carbon` run automatically
    when keyframes exist and GEMINI_API_KEY is configured (or always if `verify=True`).
    `SpendCapHit` (HTTP 429) always propagates immediately.
    """
    fit_res = fit_scene(scene, attempt=0, on_event=on_event)

    should_verify = verify
    if should_verify is None:
        has_kfs = bool(select_keyframes(scene, fit_res.t_start_s))
        if has_kfs and gemini.have_key():
            should_verify = True
        else:
            if has_kfs and not gemini.have_key():
                print(
                    "[physics] GEMINI_API_KEY not set; skipping fit verification/repair",
                    file=sys.stderr,
                )
            should_verify = False

    best_fit = fit_res
    best_ver: VerificationResult | None = None
    best_score = -1e9
    repairs_used = 0
    history: list[dict[str, Any]] = []

    t_loop = time.time()
    budget_s = config.REPAIR_BUDGET_S if repair_budget_s is None else repair_budget_s
    if should_verify and fit_res.n_points > 0:
        for attempt in range(max_repairs + 1):
            ver = verify_fit(
                scene,
                fit_res,
                attempt=attempt,
                confidence_threshold=confidence_threshold,
                on_event=on_event,
            )
            score = ver.selection_score(fit_res.fit_error_px)
            if best_ver is None or (ver.passed and not best_ver.passed) or (
                ver.passed == best_ver.passed and score > best_score
            ):
                best_fit = fit_res
                best_ver = ver
                best_score = score

            if ver.passed or attempt >= max_repairs:
                break
            left = budget_s - (time.time() - t_loop)
            if left <= 5:
                if on_event is not None:
                    on_event("repair_skipped", {"reason": f"check-and-repair budget of {budget_s:.0f} s used"})
                break

            # A repair that outruns the budget is abandoned (its call still finishes and is logged in the background).
            pool = ThreadPoolExecutor(max_workers=1)
            fut = pool.submit(propose_repair, scene, fit_res, ver, attempt=attempt + 1,
                              previous_attempts=history, on_event=on_event)
            pool.shutdown(wait=False)
            try:
                proposal = fut.result(timeout=left)
            except FuturesTimeout:
                if on_event is not None:
                    on_event("repair_skipped", {"reason": f"{config.REPAIR} repair still running after {left:.0f} s; keeping the best fit"})
                break
            history.append({
                "attempt": attempt,
                "motion_model": fit_res.motion_model,
                "fit_error_px": fit_res.fit_error_px,
                "confidences": ver.confidences,
                "answers": ver.answers,
                "proposal": proposal.model_dump(),
            })
            repairs_used += 1
            init_guess, priors_override = proposal_to_fit_overrides(proposal)
            fit_res = fit_scene(
                scene,
                motion_model=proposal.motion_model,
                initial_guess=init_guess,
                param_priors=priors_override,
                attempt=attempt + 1,
                on_event=on_event,
            )

    trajectory = simulate_future(scene, best_fit)
    dynamics_used, fallback_reason = "physics", implausible(scene, best_fit, trajectory)
    if fallback_reason:
        from foresight.pipeline import const_velocity_trajectory  # same baseline the eval reports

        trajectory = const_velocity_trajectory(scene)
        dynamics_used = "const_velocity_fallback"
        if on_event is not None:
            on_event("fallback", {"to": "constant velocity", "reason": fallback_reason})

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
        "fit_params": best_fit.fit_params_dict,
        "fit_error_px": best_fit.fit_error_px,
        "motion_model": best_fit.motion_model,
        "decision1_confidences": best_ver.confidences if best_ver else {},
        "decision1_answers": best_ver.answers if best_ver else {},
        "repairs_used": repairs_used,
        "dynamics_used": dynamics_used,
        "fallback_reason": fallback_reason,
        "trajectory": trajectory,
        "fit_trajectory": best_fit.fit_trajectory,
    }


__all__ = [
    "BodyParams",
    "CoordTransform",
    "FitResult",
    "MAX_REPAIRS",
    "PhysicsWorld",
    "RepairProposal",
    "VerificationResult",
    "default_priors",
    "evaluate_at_times",
    "fit_scene",
    "implausible",
    "predict",
    "propose_repair",
    "render_trajectory",
    "simulate_future",
    "verify_fit",
]
