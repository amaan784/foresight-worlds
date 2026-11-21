"""Foresight physics simulation, fitting, rendering, verification, and repair."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from foresight import gemini

from .fit import FitResult, fit_scene
from .render import render_trajectory
from .repair import MAX_REPAIRS, RepairProposal, proposal_to_fit_overrides, propose_repair
from .simulate import evaluate_at_times, simulate_future
from .verify import CONFIDENCE_THRESHOLD, VerificationResult, select_keyframes, verify_fit
from .world import BodyParams, CoordTransform, PhysicsWorld, default_priors

OnEvent = Callable[[str, dict[str, Any]], None]


def predict(
    scene: dict[str, Any],
    on_event: OnEvent | None = None,
    *,
    out_size: tuple[int, int] | None = None,
    verify: bool | None = None,
    max_repairs: int = MAX_REPAIRS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
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
                    "[physics] GEMINI_API_KEY not set; skipping Decision 1 verification/repair",
                    file=sys.stderr,
                )
            should_verify = False

    best_fit = fit_res
    best_ver: VerificationResult | None = None
    best_score = -1e9
    repairs_used = 0
    history: list[dict[str, Any]] = []

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

            proposal = propose_repair(
                scene,
                fit_res,
                ver,
                attempt=attempt + 1,
                previous_attempts=history,
                on_event=on_event,
            )
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
    "predict",
    "propose_repair",
    "render_trajectory",
    "simulate_future",
    "verify_fit",
]
