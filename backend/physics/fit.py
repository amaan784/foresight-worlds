"""Parameter fitting for Foresight physics worlds using scipy.optimize.least_squares.

Fits per-body initial state (x0, y0, vx0, vy0) and identifiable physical material
parameters (friction, elasticity, damping) against observed track points evaluated
at their real timestamps `t_s` (never uniform frame indices).

Key design choices for short conditioning tracks (5-14 points, 0.2-0.6 s):
- Gravity is fixed at `scene["gravity_m_s2"]`; scale is fixed at `scene["px_per_m"]`.
- Two-stage fitting with soft MAP priors and an identifiability/improvement gate:
  1. Stage 1 fits kinematics (x0, y0, vx0, vy0) -- plus elasticity if a bounce is
     already visible in the raw track -- using strong physical priors.
  2. Stage 2 checks which material parameters are actually exercised by the rollout
     (bounce -> elasticity; sliding contact -> friction; rolling contact -> damping)
     and refines them with weak MAP priors, accepting the refinement only when a
     bounce occurred or data RMSE improves meaningfully over Stage 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

from .world import BodyParams, CoordTransform, PhysicsWorld, default_priors

OnEvent = Callable[[str, dict[str, Any]], None]

# Bounds for fitted material parameters
FRICTION_BOUNDS = (0.02, 1.50)
ELASTICITY_BOUNDS = (0.05, 0.98)
DAMPING_BOUNDS = (0.0, 5.0)

# Prior standard deviations (soft MAP regularization in normalized residual units)
PRIOR_STD_FRICTION = 0.50
PRIOR_STD_ELASTICITY = 0.50
PRIOR_STD_DAMPING = 0.65


@dataclass
class FitResult:
    """Output of fitting a scene's dynamic bodies to their conditioning tracks."""

    params: dict[str, BodyParams]
    fit_params_dict: dict[str, dict[str, float]]
    fit_error_px: float
    fit_trajectory: list[dict[str, Any]]
    t_start_s: float
    motion_model: str
    n_points: int
    had_bounce: bool = False
    identifiable_params: dict[str, list[str]] = field(default_factory=dict)


def _detect_track_bounce(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    coord: CoordTransform,
) -> int | None:
    """Return the track index near a bounce (vertical or horizontal reversal), else None."""
    n = len(ts)
    if n < 3:
        return None
    dts = np.diff(ts)
    valid = dts > 1e-4
    if not np.any(valid):
        return None
    vys = np.where(valid, np.diff(ys) / np.where(valid, dts, 1.0), 0.0)
    vxs = np.where(valid, np.diff(xs) / np.where(valid, dts, 1.0), 0.0)

    # Vertical bounce: downward velocity followed by upward velocity
    for i in range(len(vys) - 1):
        if vys[i] < -0.20 and vys[i + 1] > 0.10:
            return i + 1

    # Also check a clear local y minimum where the ball drops then rebounds
    min_drop_m = max(0.008, coord.len_px_to_m(5.0))
    for i in range(1, n - 1):
        if (ys[i - 1] - ys[i]) > min_drop_m and (ys[i + 1] - ys[i]) > min_drop_m:
            return i

    # Horizontal wall/obstacle bounce: vx flips sign with significant speed change
    for i in range(len(vxs) - 1):
        if vxs[i] * vxs[i + 1] < 0 and abs(vxs[i] - vxs[i + 1]) > 0.40:
            return i + 1

    return None


def _surface_top_and_slope_at_x(
    scene: dict[str, Any],
    coord: CoordTransform,
    x_m: float,
) -> list[tuple[float, float]]:
    """Return (y_m, slope_dy_dx) of all support surfaces spanning x_m."""
    out: list[tuple[float, float]] = []
    for surf in scene.get("surfaces", []):
        p0 = coord.px_to_m(surf["p0"][0], surf["p0"][1])
        p1 = coord.px_to_m(surf["p1"][0], surf["p1"][1])
        lo, hi = (p0[0], p1[0]) if p0[0] <= p1[0] else (p1[0], p0[0])
        dx = p1[0] - p0[0]
        if lo - 1e-4 <= x_m <= hi + 1e-4 and abs(dx) > 1e-6:
            slope = (p1[1] - p0[1]) / dx
            y_m = p0[1] + slope * (x_m - p0[0])
            out.append((y_m, slope))
    return out


def _surface_top_y_at_x(
    scene: dict[str, Any],
    coord: CoordTransform,
    x_m: float,
) -> list[float]:
    return [y for y, _ in _surface_top_and_slope_at_x(scene, coord, x_m)]


def _obj_vertical_half_m(obj: dict[str, Any], coord: CoordTransform) -> float:
    if obj.get("shape") == "box":
        return 0.5 * coord.len_px_to_m(float(obj.get("h_px") or 20.0))
    return coord.len_px_to_m(float(obj.get("radius_px") or 10.0))


def _initial_guess_for_object(
    obj: dict[str, Any],
    scene: dict[str, Any],
    coord: CoordTransform,
    t_start_s: float,
    motion_model: str,
    override: dict[str, Any] | None = None,
    prior_override: dict[str, float] | None = None,
) -> tuple[BodyParams, bool]:
    """Estimate initial (x0, y0, vx0, vy0) and priors from real timestamps."""
    track = sorted(obj.get("track", []), key=lambda p: float(p["t_s"]))
    priors = default_priors(motion_model, obj.get("shape", "circle"))
    if prior_override:
        priors.update({k: float(v) for k, v in prior_override.items() if k in priors})

    if not track:
        cx, cy = coord.px_to_m(coord.width_px * 0.5, coord.height_px * 0.5)
        p = BodyParams(x0=cx, y0=cy, vx0=0.0, vy0=0.0, **priors)
        return p, False

    ts = np.array([float(pt["t_s"]) for pt in track], dtype=float)
    pts_m = [coord.px_to_m(float(pt["x"]), float(pt["y"])) for pt in track]
    xs = np.array([p[0] for p in pts_m], dtype=float)
    ys = np.array([p[1] for p in pts_m], dtype=float)
    g = float(scene.get("gravity_m_s2", 9.81))

    bounce_idx = _detect_track_bounce(ts, xs, ys, coord)
    had_raw_bounce = bounce_idx is not None

    if len(ts) == 1:
        x0, y0, vx0, vy0 = float(xs[0]), float(ys[0]), 0.0, 0.0
    else:
        # Use pre-bounce segment when a bounce occurs so initial velocity isn't averaged across rebound
        end_idx = bounce_idx if (bounce_idx is not None and bounce_idx >= 2) else len(ts)
        taus = ts[:end_idx] - t_start_s
        x_seg = xs[:end_idx]
        y_seg = ys[:end_idx]

        # X regression: prefer quadratic if >= 4 points and clean deceleration, else linear
        A_lin = np.column_stack([np.ones_like(taus), taus])
        c_x_lin, *_ = np.linalg.lstsq(A_lin, x_seg, rcond=None)
        x0, vx0 = float(c_x_lin[0]), float(c_x_lin[1])
        if len(taus) >= 4 and (taus[-1] - taus[0]) > 0.05:
            A_quad = np.column_stack([np.ones_like(taus), taus, 0.5 * taus ** 2])
            c_x_q, *_ = np.linalg.lstsq(A_quad, x_seg, rcond=None)
            ax = float(c_x_q[2])
            if abs(ax) < 20.0 and (ax * float(c_x_q[1]) <= 0.5):
                lin_err = float(np.sqrt(np.mean((A_lin @ c_x_lin - x_seg) ** 2)))
                quad_err = float(np.sqrt(np.mean((A_quad @ c_x_q - x_seg) ** 2)))
                if quad_err < 0.75 * lin_err:
                    x0, vx0 = float(c_x_q[0]), float(c_x_q[1])

        # Y regression: compare linear (on surface) vs fixed-gravity ballistic
        c_y_lin, *_ = np.linalg.lstsq(A_lin, y_seg, rcond=None)
        err_lin = float(np.sqrt(np.mean((A_lin @ c_y_lin - y_seg) ** 2)))

        y_plus_g = y_seg + 0.5 * g * (taus ** 2)
        c_y_bal, *_ = np.linalg.lstsq(A_lin, y_plus_g, rcond=None)
        y_bal_pred = c_y_bal[0] + c_y_bal[1] * taus - 0.5 * g * (taus ** 2)
        err_bal = float(np.sqrt(np.mean((y_bal_pred - y_seg) ** 2)))

        if had_raw_bounce or motion_model in ("fall", "bounce") or err_bal < 0.85 * err_lin:
            y0, vy0 = float(c_y_bal[0]), float(c_y_bal[1])
        else:
            y0, vy0 = float(c_y_lin[0]), float(c_y_lin[1])

    # Avoid spawning slightly buried inside a support surface (which causes a spurious hop)
    half_h = _obj_vertical_half_m(obj, coord)
    if motion_model in ("roll", "slide", "other") and not had_raw_bounce:
        for sy in _surface_top_y_at_x(scene, coord, x0):
            target_y = sy + half_h
            if target_y - 0.6 * half_h <= y0 < target_y:
                y0 = target_y + 1e-4
                if vy0 < 0.0 and abs(vy0) < 0.5:
                    vy0 = 0.0

    params = BodyParams(x0=x0, y0=y0, vx0=vx0, vy0=vy0, **priors)
    if override:
        for k in ("x0", "y0", "vx0", "vy0", "friction", "elasticity", "damping"):
            if k in override and override[k] is not None:
                setattr(params, k, float(override[k]))
    return params, had_raw_bounce


def _rollout_track(
    scene: dict[str, Any],
    params_by_id: dict[str, BodyParams],
    t_start_s: float,
    timestamps: list[float],
    motion_model: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, bool]]]:
    """Simulate `scene` over sorted `timestamps` and collect positions + contact flags."""
    world = PhysicsWorld(scene, params_by_id, t_start_s=t_start_s, motion_model=motion_model)
    contacts: dict[str, dict[str, bool]] = {
        oid: {"surface_contact": False, "bounce": False, "oblique_bounce": False}
        for oid in params_by_id
    }
    prev_vel: dict[str, tuple[float, float]] = {
        oid: (world.bodies[oid].velocity.x, world.bodies[oid].velocity.y)
        for oid in world.bodies
    }

    traj: list[dict[str, Any]] = []
    for t in timestamps:
        # Step in small slices (~0.01s) so contact/bounce detection never misses an impact
        while world.time_s + 1e-9 < t:
            t_next = min(t, world.time_s + 0.01)
            world.step_to(t_next)
            for oid, body in world.bodies.items():
                vx, vy = float(body.velocity.x), float(body.velocity.y)
                pvx, pvy = prev_vel[oid]
                if (pvy < -0.45 and vy > 0.25) or (pvx * vx < 0 and abs(vx - pvx) > 0.50):
                    contacts[oid]["bounce"] = True
                    if abs(pvx) > 0.15 or abs(vx - pvx) > 0.1:
                        contacts[oid]["oblique_bounce"] = True

                def _check_arb(arb: Any, _oid: str = oid) -> None:
                    if abs(arb.contact_point_set.normal.y) > 0.25:
                        contacts[_oid]["surface_contact"] = True

                body.each_arbiter(_check_arb)
                prev_vel[oid] = (vx, vy)

        st_px = world.get_state_px()
        traj.append({
            "t_s": float(t),
            "objects": {
                oid: {"x": s["x"], "y": s["y"], "angle": s["angle"]}
                for oid, s in st_px.items()
            },
        })
    return traj, contacts


def _evaluate_rmse_px(
    traj: list[dict[str, Any]],
    obs_by_time: dict[float, list[tuple[str, float, float]]],
) -> float:
    sq_errs: list[float] = []
    for entry in traj:
        t = entry["t_s"]
        for oid, ox, oy in obs_by_time.get(t, []):
            pred = entry["objects"][oid]
            sq_errs.append((pred["x"] - ox) ** 2 + (pred["y"] - oy) ** 2)
    if not sq_errs:
        return 0.0
    return float(math.sqrt(sum(sq_errs) / len(sq_errs)))


def _run_lsq(
    scene: dict[str, Any],
    coord: CoordTransform,
    base_params: dict[str, BodyParams],
    active_vars: dict[str, list[str]],
    priors_by_id: dict[str, dict[str, float]],
    t_start_s: float,
    timestamps: list[float],
    obs_by_time: dict[float, list[tuple[str, float, float]]],
    motion_model: str,
    sigma_px: float,
    max_nfev: int = 80,
) -> tuple[dict[str, BodyParams], list[dict[str, Any]], dict[str, dict[str, bool]], float]:
    """Pack `active_vars` into a vector and run bounded scipy least_squares."""
    layout: list[tuple[str, str]] = []
    x_init: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    w_m = coord.len_px_to_m(coord.width_px)
    h_m = coord.len_px_to_m(coord.height_px)
    pos_margin = max(0.5, 0.35 * max(w_m, h_m))
    max_vel = max(15.0, 6.0 * max(w_m, h_m))

    obj_by_id = {o["id"]: o for o in scene.get("objects", [])}

    for oid, var_names in active_vars.items():
        bp = base_params[oid]
        obj = obj_by_id.get(oid, {})
        half_h = _obj_vertical_half_m(obj, coord) if obj else 0.0
        # Non-penetration bounds: a body starting on or above a support surface cannot
        # start buried inside the solid surface, nor slam downward into it on step 0.
        y_min = bp.y0 - pos_margin
        vy_min = -max_vel
        if half_h > 0:
            for sy, slope in _surface_top_and_slope_at_x(scene, coord, bp.x0):
                surf_center_y = sy + half_h
                if bp.y0 >= surf_center_y - 0.35 * half_h:
                    y_min = max(y_min, surf_center_y - 0.002)
                if motion_model in ("roll", "slide", "other") and abs(bp.y0 - surf_center_y) <= 0.35 * half_h:
                    vy_surf = slope * bp.vx0
                    vy_min = max(vy_min, vy_surf - 0.02)

        for vname in var_names:
            layout.append((oid, vname))
            val = float(getattr(bp, vname))
            if vname == "x0":
                lo, hi = bp.x0 - pos_margin, bp.x0 + pos_margin
            elif vname == "y0":
                lo, hi = y_min, max(y_min + 0.05, bp.y0 + pos_margin)
            elif vname == "vx0":
                lo, hi = -max_vel, max_vel
            elif vname == "vy0":
                lo, hi = vy_min, max_vel
            elif vname == "friction":
                lo, hi = FRICTION_BOUNDS
            elif vname == "elasticity":
                lo, hi = ELASTICITY_BOUNDS
            elif vname == "damping":
                lo, hi = DAMPING_BOUNDS
            else:
                lo, hi = -1e3, 1e3
            x_init.append(min(max(val, lo + 1e-6), hi - 1e-6))
            lb.append(lo)
            ub.append(hi)

    if not layout:
        traj, contacts = _rollout_track(scene, base_params, t_start_s, timestamps, motion_model)
        return base_params, traj, contacts, _evaluate_rmse_px(traj, obs_by_time)

    prior_stds = {
        "friction": PRIOR_STD_FRICTION,
        "elasticity": PRIOR_STD_ELASTICITY,
        "damping": PRIOR_STD_DAMPING,
    }

    def _unpack(vec: np.ndarray) -> dict[str, BodyParams]:
        out = {oid: BodyParams.from_dict(bp.to_dict()) for oid, bp in base_params.items()}
        for (oid, vname), val in zip(layout, vec):
            setattr(out[oid], vname, float(val))
        return out

    def _residuals(vec: np.ndarray) -> np.ndarray:
        cand = _unpack(vec)
        traj, _ = _rollout_track(scene, cand, t_start_s, timestamps, motion_model)
        res: list[float] = []
        for entry in traj:
            t = entry["t_s"]
            for oid, ox, oy in obs_by_time.get(t, []):
                pred = entry["objects"][oid]
                res.append((pred["x"] - ox) / sigma_px)
                res.append((pred["y"] - oy) / sigma_px)
        for (oid, vname), val in zip(layout, vec):
            if vname in prior_stds:
                p0 = priors_by_id[oid][vname]
                res.append((float(val) - p0) / prior_stds[vname])
        return np.asarray(res, dtype=float)

    opt = least_squares(
        _residuals,
        x0=np.asarray(x_init, dtype=float),
        bounds=(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)),
        method="trf",
        loss="soft_l1",
        f_scale=10.0,
        x_scale="jac",
        diff_step=1e-3,
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=max_nfev,
    )
    best_params = _unpack(opt.x)
    traj, contacts = _rollout_track(scene, best_params, t_start_s, timestamps, motion_model)
    rmse = _evaluate_rmse_px(traj, obs_by_time)
    return best_params, traj, contacts, rmse


def fit_scene(
    scene: dict[str, Any],
    *,
    motion_model: str | None = None,
    initial_guess: dict[str, dict[str, Any]] | None = None,
    param_priors: dict[str, dict[str, float]] | None = None,
    attempt: int = 0,
    on_event: OnEvent | None = None,
) -> FitResult:
    """Fit physics parameters for all tracked dynamic objects in `scene`."""
    coord = CoordTransform.from_scene(scene)
    model = motion_model or scene.get("motion_type_hint", "roll")
    objects = [o for o in scene.get("objects", []) if o.get("track")]

    if not objects:
        empty = FitResult(
            params={},
            fit_params_dict={},
            fit_error_px=0.0,
            fit_trajectory=[],
            t_start_s=float(scene.get("t_cond_end_s", 3.0)),
            motion_model=model,
            n_points=0,
        )
        if on_event is not None:
            on_event("fit", {
                "attempt": attempt,
                "motion_model": model,
                "params": {},
                "fit_error_px": 0.0,
                "n_points": 0,
            })
        return empty

    # Build sorted unique observation timestamps and per-time observation map
    obs_by_time: dict[float, list[tuple[str, float, float]]] = {}
    n_points = 0
    for obj in objects:
        oid = obj["id"]
        for pt in obj["track"]:
            t = float(pt["t_s"])
            obs_by_time.setdefault(t, []).append((oid, float(pt["x"]), float(pt["y"])))
            n_points += 1
    timestamps = sorted(obs_by_time.keys())
    t_start_s = timestamps[0]

    init_params: dict[str, BodyParams] = {}
    priors_by_id: dict[str, dict[str, float]] = {}
    raw_bounce_by_id: dict[str, bool] = {}

    for obj in objects:
        oid = obj["id"]
        override = (initial_guess or {}).get(oid)
        p_override = (param_priors or {}).get(oid)
        bp, has_b = _initial_guess_for_object(
            obj, scene, coord, t_start_s, model, override=override, prior_override=p_override
        )
        init_params[oid] = bp
        priors_by_id[oid] = {
            "friction": bp.friction,
            "elasticity": bp.elasticity,
            "damping": bp.damping,
        }
        raw_bounce_by_id[oid] = has_b

    sigma_px = max(1.5, 0.0015 * max(coord.width_px, coord.height_px))

    # Stage 1: fit kinematics (plus elasticity right away if a raw bounce is present)
    stage1_vars: dict[str, list[str]] = {}
    for obj in objects:
        oid = obj["id"]
        vlist = ["x0", "y0", "vx0", "vy0"]
        if raw_bounce_by_id[oid]:
            vlist.append("elasticity")
        stage1_vars[oid] = vlist

    p1, traj1, contacts1, rmse1 = _run_lsq(
        scene, coord, init_params, stage1_vars, priors_by_id,
        t_start_s, timestamps, obs_by_time, model, sigma_px, max_nfev=80,
    )

    # Stage 2: refine only physically exercised / identifiable material parameters
    stage2_vars: dict[str, list[str]] = {}
    identifiable: dict[str, list[str]] = {}
    any_new_material = False
    any_bounce = any(raw_bounce_by_id.values()) or any(c["bounce"] for c in contacts1.values())

    for obj in objects:
        oid = obj["id"]
        shape = obj.get("shape", "circle")
        cinfo = contacts1[oid]
        n_obj_pts = len(obj.get("track", []))
        vlist = ["x0", "y0", "vx0", "vy0"]
        mat_vars: list[str] = []

        if raw_bounce_by_id[oid] or cinfo["bounce"]:
            mat_vars.append("elasticity")
        if cinfo["surface_contact"] and n_obj_pts >= 5:
            if shape == "box" or model == "slide":
                mat_vars.append("friction")
            elif shape == "circle" and model in ("roll", "other"):
                mat_vars.append("damping")

        vlist.extend(mat_vars)
        stage2_vars[oid] = vlist
        identifiable[oid] = list(mat_vars)
        if set(vlist) != set(stage1_vars[oid]):
            any_new_material = True

    best_params, best_traj, best_rmse = p1, traj1, rmse1
    if any_new_material:
        stage2_seed = init_params if rmse1 > 10.0 else p1
        p2, traj2, contacts2, rmse2 = _run_lsq(
            scene, coord, stage2_seed, stage2_vars, priors_by_id,
            t_start_s, timestamps, obs_by_time, model, sigma_px, max_nfev=90,
        )
        any_bounce = any_bounce or any(c["bounce"] for c in contacts2.values())
        # Accept Stage 2 when a bounce was newly detected OR data RMSE improved meaningfully
        improved = (rmse1 - rmse2) > max(1.0, 0.12 * rmse1)
        newly_bounced = any(
            contacts1[o["id"]]["bounce"] and not raw_bounce_by_id[o["id"]] for o in objects
        )
        if improved or newly_bounced:
            best_params, best_traj, best_rmse = p2, traj2, rmse2
        else:
            # Revert unidentifiable material parameters to priors
            identifiable = {
                oid: [m for m in ms if m == "elasticity" and raw_bounce_by_id[oid]]
                for oid, ms in identifiable.items()
            }

    fit_params_dict = {oid: bp.to_dict() for oid, bp in best_params.items()}
    result = FitResult(
        params=best_params,
        fit_params_dict=fit_params_dict,
        fit_error_px=round(best_rmse, 3),
        fit_trajectory=best_traj,
        t_start_s=t_start_s,
        motion_model=model,
        n_points=n_points,
        had_bounce=any_bounce,
        identifiable_params=identifiable,
    )

    if on_event is not None:
        on_event("fit", {
            "attempt": attempt,
            "motion_model": model,
            "params": fit_params_dict,
            "fit_error_px": result.fit_error_px,
            "n_points": n_points,
        })
    return result
