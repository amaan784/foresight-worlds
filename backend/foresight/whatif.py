"""Interactive counterfactual ('what-if') 5 s rollouts from t_cond_end_s (zero Gemini calls)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from physics import BodyParams, CoordTransform, PhysicsWorld, default_priors, implausible

from .pipeline import endpoint_error

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

MAX_SPEED_M_S = 25.0
MIN_FRICTION, MAX_FRICTION = 0.0, 1.5
MIN_BOUNCE, MAX_BOUNCE = 0.0, 0.98


class InvalidJobIdError(ValueError):
    """Raised when a job_id fails format or path-containment validation."""


class JobNotFoundError(FileNotFoundError):
    """Raised when a job's scene or result cannot be found in memory or on disk."""


class InvalidWhatIfError(ValueError):
    """Raised when counterfactual override fields are malformed or out of domain."""


@dataclass
class _Baseline:
    job_id: str
    scene: dict[str, Any]
    coord: CoordTransform
    main_id: str
    t_cond_end_s: float
    motion_model: str
    dynamics_used: str
    params_cond: dict[str, BodyParams]
    start_px_map: dict[str, tuple[float, float]]
    radius_m_map: dict[str, float]
    obj_info: dict[str, dict[str, str]]
    resting_cond: dict[str, dict[str, float]]
    real_pts: list[dict[str, float]]
    baseline_endpoint_cm: float | None


_CACHE: dict[tuple[str, int, int], _Baseline] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 32


def validate_job_id(job_id: str, out_dir: Path) -> Path:
    """Validate `job_id` strictly and return `out_dir / job_id` contained within `out_dir`."""
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise InvalidJobIdError("invalid job_id")
    root = Path(out_dir).resolve()
    target = (root / job_id).resolve()
    if target.parent != root:
        raise InvalidJobIdError("invalid job_id")
    return target


def _finite_float(val: Any, field: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise InvalidWhatIfError(f"{field} must be a finite number")
    f = float(val)
    if not math.isfinite(f):
        raise InvalidWhatIfError(f"{field} must be a finite number")
    return f


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _estimate_from_track(
    scene: dict[str, Any], coord: CoordTransform, t_cond_end_s: float, motion_model: str, n_fit: int = 6
) -> dict[str, BodyParams]:
    """Fallback initial state at `t_cond_end_s` for jobs saved with `dynamics == 'const_velocity'`."""
    out: dict[str, BodyParams] = {}
    for ob in scene.get("objects", []):
        pts = (ob.get("track") or [])[-n_fit:]
        if not pts:
            continue
        t = np.array([float(p["t_s"]) for p in pts], dtype=float)
        xy = np.array([[float(p["x"]), float(p["y"])] for p in pts], dtype=float)
        if len(pts) >= 2 and float(np.ptp(t)) > 1e-6:
            a_mat = np.stack([np.ones_like(t), t], axis=1)
            cx = np.linalg.lstsq(a_mat, xy[:, 0], rcond=None)[0]
            cy = np.linalg.lstsq(a_mat, xy[:, 1], rcond=None)[0]
            x_px, y_px = float(cx[0] + cx[1] * t_cond_end_s), float(cy[0] + cy[1] * t_cond_end_s)
            vx_m, vy_m = coord.vec_px_to_m(float(cx[1]), float(cy[1]))
        else:
            x_px, y_px = float(xy[-1, 0]), float(xy[-1, 1])
            vx_m, vy_m = 0.0, 0.0
        x0, y0 = coord.px_to_m(x_px, y_px)
        priors = default_priors(motion_model, str(ob.get("shape", "circle")))
        out[str(ob["id"])] = BodyParams(
            x0=x0, y0=y0,
            vx0=_clamp(vx_m, -MAX_SPEED_M_S, MAX_SPEED_M_S),
            vy0=_clamp(vy_m, -MAX_SPEED_M_S, MAX_SPEED_M_S),
            friction=float(priors["friction"]), elasticity=float(priors["elasticity"]),
            damping=float(priors["damping"]), t_start_s=t_cond_end_s,
        )
    return out


def _load_baseline(job_id: str, out_dir: Path, jobs_mem: dict[str, Any] | None = None) -> _Baseline:
    job_dir = validate_job_id(job_id, out_dir)
    mem_job = (jobs_mem or {}).get(job_id)
    if mem_job is not None and getattr(mem_job, "result", None) is not None:
        result = mem_job.result
        job_dir = Path(mem_job.dir).resolve()
        res_stamp = id(result)
    else:
        res_path = job_dir / "result.json"
        if not res_path.exists():
            raise JobNotFoundError(f"unknown job {job_id}")
        res_stamp = res_path.stat().st_mtime_ns
        result = json.loads(res_path.read_text())

    scene_path = job_dir / "work" / "scene.json"
    if not scene_path.exists():
        raise JobNotFoundError(f"scene not available for job {job_id}")
    cache_key = (str(job_dir), scene_path.stat().st_mtime_ns, res_stamp)

    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    scene = json.loads(scene_path.read_text())
    objects = scene.get("objects") or []
    if not objects:
        raise InvalidWhatIfError("scene has no tracked moving objects")

    coord = CoordTransform.from_scene(scene)
    t_cond_end_s = float(scene.get("t_cond_end_s", 3.0))
    phys = result.get("physics") or {}
    motion_model = str(phys.get("motion_model") or scene.get("motion_type_hint") or "roll")
    dynamics_used = str((result.get("metrics") or {}).get("dynamics_used") or result.get("dynamics") or "physics")
    main_id = str(max(objects, key=lambda ob: len(ob.get("track") or []))["id"])

    fit_params = phys.get("fit_params") or {}
    params0: dict[str, BodyParams] = {}
    starts: list[float] = []
    for ob in objects:
        oid = str(ob["id"])
        if oid not in fit_params:
            continue
        raw = dict(fit_params[oid])
        if raw.get("t_start_s") is None and ob.get("track"):
            raw["t_start_s"] = float(ob["track"][0]["t_s"])
        bp = BodyParams.from_dict(raw)
        params0[oid] = bp
        if bp.t_start_s is not None:
            starts.append(float(bp.t_start_s))

    resting_cond: dict[str, dict[str, float]] = {}
    if params0:
        t_init = min(min(starts) if starts else t_cond_end_s, t_cond_end_s)
        w0 = PhysicsWorld(scene, params0, t_start_s=t_init, motion_model=motion_model)
        if t_cond_end_s > w0.time_s + 1e-12:
            w0.step_to(t_cond_end_s)
        stm, stpx = w0.get_state_m(), w0.get_state_px()
        params_cond: dict[str, BodyParams] = {}
        start_px_map: dict[str, tuple[float, float]] = {}
        radius_m_map: dict[str, float] = {}
        for oid, bp in params0.items():
            st = stm[oid]
            params_cond[oid] = BodyParams(
                x0=st["x"], y0=st["y"], vx0=st["vx"], vy0=st["vy"],
                friction=bp.friction, elasticity=bp.elasticity, damping=bp.damping,
                angle0=st["angle"], omega0=st["omega"], t_start_s=t_cond_end_s,
            )
            start_px_map[oid] = (float(stpx[oid]["x"]), float(stpx[oid]["y"]))
            radius_m_map[oid] = float(w0.object_meta[oid]["radius_m"])
        for rest in w0.layers.resting:
            rid = str(rest["id"])
            if rid in stm:
                rx0, ry0 = coord.px_to_m(rest["x"], rest["y"])
                rst = stm[rid]
                if math.hypot(rst["x"] - rx0, rst["y"] - ry0) > 1e-6 or math.hypot(rst["vx"], rst["vy"]) > 1e-6:
                    resting_cond[rid] = rst
    else:
        params_cond = _estimate_from_track(scene, coord, t_cond_end_s, motion_model)
        if not params_cond:
            raise InvalidWhatIfError("unable to initialize state from scene tracks")
        w0 = PhysicsWorld(scene, params_cond, t_start_s=t_cond_end_s, motion_model=motion_model)
        stpx = w0.get_state_px()
        start_px_map = {oid: (float(stpx[oid]["x"]), float(stpx[oid]["y"])) for oid in params_cond}
        radius_m_map = {oid: float(w0.object_meta[oid]["radius_m"]) for oid in params_cond}

    obj_info = {
        str(ob["id"]): {"name": str(ob.get("name") or ob["id"]), "shape": str(ob.get("shape") or "circle")}
        for ob in objects if str(ob["id"]) in params_cond
    }
    if main_id not in params_cond:
        main_id = next(iter(params_cond))

    raw_real = (result.get("paths") or {}).get("real") or []
    real_pts = [{"t_s": float(p[0]), "x": float(p[1]), "y": float(p[2])} for p in raw_real if len(p) >= 3]
    ee_base = ((result.get("metrics") or {}).get("endpoint_error_cm") or {}).get("foresight")

    base = _Baseline(
        job_id=job_id, scene=scene, coord=coord, main_id=main_id, t_cond_end_s=t_cond_end_s,
        motion_model=motion_model, dynamics_used=dynamics_used, params_cond=params_cond,
        start_px_map=start_px_map, radius_m_map=radius_m_map, obj_info=obj_info,
        resting_cond=resting_cond, real_pts=real_pts,
        baseline_endpoint_cm=float(ee_base) if isinstance(ee_base, (int, float)) else None,
    )
    with _CACHE_LOCK:
        _CACHE.pop(cache_key, None)
        _CACHE[cache_key] = base
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
    return base


def _rollout_from_cond_end(base: _Baseline, params_cond: dict[str, BodyParams]) -> list[dict[str, Any]]:
    fps_out = int(base.scene.get("fps_out", 24))
    n_frames = int(round(float(base.scene.get("pred_seconds", 5.0)) * fps_out))
    timestamps = [base.t_cond_end_s + (k / fps_out) for k in range(n_frames)]

    world = PhysicsWorld(base.scene, params_cond, t_start_s=base.t_cond_end_s, motion_model=base.motion_model)
    for rid, rst in base.resting_cond.items():
        if rid in world.bodies:
            b = world.bodies[rid]
            b.position, b.velocity = (rst["x"], rst["y"]), (rst["vx"], rst["vy"])
            b.angle, b.angular_velocity = rst["angle"], rst["omega"]

    trajectory: list[dict[str, Any]] = []
    for t in timestamps:
        if t > world.time_s + 1e-12:
            world.step_to(t)
        trajectory.append({"t_s": float(t), "objects": world.get_state_px()})
    return trajectory


def _format_obj_baseline(base: _Baseline, oid: str) -> dict[str, Any]:
    bp = base.params_cond[oid]
    sx, sy = base.start_px_map[oid]
    info = base.obj_info[oid]
    return {
        "object_id": oid, "name": info["name"], "shape": info["shape"], "start_px": [sx, sy],
        "vx0": float(bp.vx0), "vy0": float(bp.vy0), "speed_m_s": float(math.hypot(bp.vx0, bp.vy0)),
        "friction": float(bp.friction), "bounce": float(bp.elasticity),
    }


def get_whatif_baseline(job_id: str, out_dir: Path, jobs_mem: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return baseline `t = t_cond_end_s` state and metadata for interactive What-If controls."""
    base = _load_baseline(job_id, out_dir, jobs_mem)
    return {
        "job_id": base.job_id,
        **_format_obj_baseline(base, base.main_id),
        "orig_size": [base.coord.width_px, base.coord.height_px],
        "px_per_m": float(base.coord.px_per_m),
        "t_cond_end_s": base.t_cond_end_s,
        "motion_model": base.motion_model,
        "dynamics_used": base.dynamics_used,
        "endpoint_error_cm": base.baseline_endpoint_cm,
        "objects": [_format_obj_baseline(base, oid) for oid in base.params_cond],
    }


def run_whatif(
    job_id: str, overrides: dict[str, Any], out_dir: Path, jobs_mem: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Apply counterfactual overrides at `t = t_cond_end_s` and simulate the next 5.0 s locally."""
    t_wall = time.perf_counter()
    base = _load_baseline(job_id, out_dir, jobs_mem)

    oid = str(overrides.get("object_id") or base.main_id)
    if oid not in base.params_cond:
        raise InvalidWhatIfError(f"unknown object_id {oid!r}")

    params_mod = {k: replace(bp) for k, bp in base.params_cond.items()}
    bp = params_mod[oid]
    sx_px, sy_px = base.start_px_map[oid]

    raw_start = overrides.get("start_px")
    if raw_start is not None:
        if not isinstance(raw_start, (list, tuple)) or len(raw_start) != 2:
            raise InvalidWhatIfError("start_px must be [x_px, y_px]")
        w_px, h_px = float(base.coord.width_px), float(base.coord.height_px)
        sx_px = _clamp(_finite_float(raw_start[0], "start_px[0]"), -0.25 * w_px, 1.25 * w_px)
        sy_px = _clamp(_finite_float(raw_start[1], "start_px[1]"), -0.25 * h_px, 1.25 * h_px)
        bp.x0, bp.y0 = base.coord.px_to_m(sx_px, sy_px)

    if overrides.get("vx0") is not None:
        new_vx = _clamp(_finite_float(overrides["vx0"], "vx0"), -MAX_SPEED_M_S, MAX_SPEED_M_S)
        if abs(new_vx - bp.vx0) > 1e-5 and base.obj_info[oid]["shape"] == "circle" and base.motion_model in ("roll", "other"):
            bp.omega0 = -new_vx / max(1e-4, base.radius_m_map[oid])
        bp.vx0 = new_vx

    if overrides.get("vy0") is not None:
        bp.vy0 = _clamp(_finite_float(overrides["vy0"], "vy0"), -MAX_SPEED_M_S, MAX_SPEED_M_S)
    if overrides.get("friction") is not None:
        bp.friction = _clamp(_finite_float(overrides["friction"], "friction"), MIN_FRICTION, MAX_FRICTION)
    if overrides.get("bounce") is not None:
        bp.elasticity = _clamp(_finite_float(overrides["bounce"], "bounce"), MIN_BOUNCE, MAX_BOUNCE)

    traj = _rollout_from_cond_end(base, params_mod)
    ee = endpoint_error(traj, base.real_pts, base.scene, base.main_id)
    bad = implausible(base.scene, SimpleNamespace(params=params_mod), traj)  # type: ignore[arg-type]
    latency_ms = (time.perf_counter() - t_wall) * 1000.0

    pred_oid = [[st["t_s"], st["objects"][oid]["x"], st["objects"][oid]["y"]] for st in traj if oid in st["objects"]]
    pred_main = (
        pred_oid if oid == base.main_id else
        [[st["t_s"], st["objects"][base.main_id]["x"], st["objects"][base.main_id]["y"]] for st in traj if base.main_id in st["objects"]]
    )
    return {
        "job_id": base.job_id, "object_id": oid, "name": base.obj_info[oid]["name"],
        "predicted": pred_oid, "main_predicted": pred_main, "start_px": [sx_px, sy_px],
        "vx0": float(bp.vx0), "vy0": float(bp.vy0), "speed_m_s": float(math.hypot(bp.vx0, bp.vy0)),
        "friction": float(bp.friction), "bounce": float(bp.elasticity), "motion_model": base.motion_model,
        "endpoint_error_cm": ee["cm"] if ee else None, "endpoint_error_px": ee["px"] if ee else None,
        "implausible_reason": bad, "latency_ms": round(latency_ms, 2),
    }
