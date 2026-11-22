"""Synthetic and unit tests for backend/physics/fit.py."""
from __future__ import annotations

import numpy as np
import pytest

from physics.fit import fit_scene
from physics.world import BodyParams, CoordTransform, PhysicsWorld


def _make_scene(
    orig_size: tuple[int, int] = (3840, 2160),
    px_per_m: float = 3700.0,
    motion_hint: str = "roll",
    shape: str = "circle",
    radius_px: float = 124.0,
    box_wh: tuple[float, float] = (160.0, 120.0),
    surfaces: list[dict] | None = None,
) -> dict:
    obj = {
        "id": "obj0",
        "name": "grey tennis ball" if shape == "circle" else "wooden block",
        "shape": shape,
        "radius_px": radius_px if shape == "circle" else None,
        "w_px": box_wh[0] if shape == "box" else None,
        "h_px": box_wh[1] if shape == "box" else None,
        "est_real_size_cm": 6.7,
        "track": [],
        "visible_at_end": True,
        "sprite_png": "/tmp/obj0.png",
    }
    return {
        "clip_id": "synthetic_fit",
        "work_dir": "/tmp/foresight_fit_test",
        "orig_size": list(orig_size),
        "fps_out": 24,
        "t_cond_end_s": 3.0,
        "pred_seconds": 5.0,
        "px_per_m": px_per_m,
        "px_per_m_source": "size",
        "gravity_m_s2": 9.81,
        "motion_type_hint": motion_hint,
        "objects": [obj],
        "static_objects": [],
        "surfaces": surfaces if surfaces is not None else [],
        "background_png": "/tmp/bg.png",
        "last_frame_png": "/tmp/last.png",
        "keyframes": [],
    }


def _sample_synthetic_track(
    scene: dict,
    true_params: BodyParams,
    timestamps: list[float],
    motion_model: str,
    noise_px: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    w = PhysicsWorld(scene, {"obj0": true_params}, t_start_s=timestamps[0], motion_model=motion_model)
    track = []
    for t in timestamps:
        w.step_to(t)
        st = w.get_state_px()["obj0"]
        x = st["x"] + (float(rng.normal(0.0, noise_px)) if noise_px > 0 else 0.0)
        y = st["y"] + (float(rng.normal(0.0, noise_px)) if noise_px > 0 else 0.0)
        track.append({"t_s": float(t), "x": x, "y": y})
    return track


def test_fit_short_roll_recovers_pos_vel_and_keeps_priors() -> None:
    """10-point rolling track with non-uniform real timestamps recovers pos/vel < 5%."""
    scene = _make_scene(
        orig_size=(3840, 2160),
        px_per_m=3700.0,
        motion_hint="roll",
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 1875.0], "p1": [3200.0, 1875.0]}],
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(300.0, 1875.0 - 124.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=1.85, vy0=0.0, friction=0.50, elasticity=0.65, damping=0.25)
    # 10 points over 0.36 s with a dropped frame gap (tests real t_s, not frame indices)
    ts = [2.600, 2.633, 2.667, 2.700, 2.767, 2.800, 2.833, 2.867, 2.900, 2.967]
    scene["objects"][0]["track"] = _sample_synthetic_track(scene, true_p, ts, "roll")

    res = fit_scene(scene)
    p = res.params["obj0"]
    assert p.x0 == pytest.approx(true_p.x0, rel=0.05, abs=0.005)
    assert p.y0 == pytest.approx(true_p.y0, rel=0.05, abs=0.005)
    assert p.vx0 == pytest.approx(true_p.vx0, rel=0.05)
    assert abs(p.vy0 - true_p.vy0) < 0.05
    assert res.fit_error_px < 3.0
    # Unidentifiable without bounce: stay at tennis-ball-on-wood priors
    assert 0.55 <= p.elasticity <= 0.75
    assert 0.40 <= p.friction <= 0.60


def test_fit_very_short_5_point_track_real_timestamps() -> None:
    """Even a 5-point (0.2 s) rolling track recovers initial pos/vel within 5%."""
    scene = _make_scene(
        orig_size=(1920, 1080),
        px_per_m=1800.0,
        motion_hint="roll",
        radius_px=60.0,
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 900.0], "p1": [1920.0, 900.0]}],
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(300.0, 900.0 - 60.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=1.60, vy0=0.0, friction=0.50, elasticity=0.65, damping=0.25)
    # 5 irregular timestamps over 0.20 s
    ts = [2.750, 2.783, 2.850, 2.883, 2.950]
    scene["objects"][0]["track"] = _sample_synthetic_track(scene, true_p, ts, "roll")

    res = fit_scene(scene)
    p = res.params["obj0"]
    assert p.x0 == pytest.approx(true_p.x0, rel=0.05, abs=0.005)
    assert p.y0 == pytest.approx(true_p.y0, rel=0.05, abs=0.005)
    assert p.vx0 == pytest.approx(true_p.vx0, rel=0.05)
    assert res.fit_error_px < 2.0


def test_fit_bounce_recovers_elasticity_and_vel_within_5pct() -> None:
    """Track spanning a bounce recovers pos, vel, AND elasticity within 5%."""
    scene = _make_scene(
        orig_size=(1920, 1080),
        px_per_m=1000.0,
        motion_hint="bounce",
        radius_px=45.0,
        surfaces=[{"name": "floor", "kind": "floor", "p0": [0.0, 950.0], "p1": [1920.0, 950.0]}],
    )
    c = CoordTransform.from_scene(scene)
    # Floor at y_m = 0.13, radius = 0.045 -> contact at y_m = 0.175
    # Start 0.35 m above floor with downward velocity -> bounces around t + 0.18 s
    x0, y0 = c.px_to_m(400.0, 950.0 - 45.0 - 350.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=0.80, vy0=-1.10, friction=0.50, elasticity=0.84, damping=0.02)
    ts = [2.50 + i * 0.04 for i in range(13)]  # 13 points over 0.48 s
    scene["objects"][0]["track"] = _sample_synthetic_track(scene, true_p, ts, "bounce")

    res = fit_scene(scene)
    p = res.params["obj0"]
    assert res.had_bounce
    assert p.x0 == pytest.approx(true_p.x0, rel=0.05, abs=0.005)
    assert p.y0 == pytest.approx(true_p.y0, rel=0.05, abs=0.005)
    assert p.vx0 == pytest.approx(true_p.vx0, rel=0.05)
    assert p.vy0 == pytest.approx(true_p.vy0, rel=0.05)
    assert p.elasticity == pytest.approx(true_p.elasticity, rel=0.05)
    assert res.fit_error_px < 3.0


def test_fit_sliding_box_deceleration_recovers_friction_within_5pct() -> None:
    """Sliding box with clear deceleration recovers pos, vel, AND friction within 5%."""
    scene = _make_scene(
        orig_size=(1920, 1080),
        px_per_m=1000.0,
        motion_hint="slide",
        shape="box",
        box_wh=(120.0, 80.0),
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 900.0], "p1": [1920.0, 900.0]}],
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(250.0, 900.0 - 40.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=2.40, vy0=0.0, friction=0.55, elasticity=0.30, damping=0.0)
    ts = [2.50 + i * 0.04 for i in range(12)]  # 12 points over 0.44 s
    scene["objects"][0]["track"] = _sample_synthetic_track(scene, true_p, ts, "slide")

    res = fit_scene(scene)
    p = res.params["obj0"]
    assert p.x0 == pytest.approx(true_p.x0, rel=0.05, abs=0.005)
    assert p.y0 == pytest.approx(true_p.y0, rel=0.05, abs=0.005)
    assert p.vx0 == pytest.approx(true_p.vx0, rel=0.05)
    assert p.friction == pytest.approx(true_p.friction, rel=0.05)
    assert res.fit_error_px < 2.0


def test_fit_rolling_strong_deceleration_recovers_damping_and_vel() -> None:
    """Rolling ball with clear deceleration recovers pos, vel, and damping within 5%."""
    scene = _make_scene(
        orig_size=(1920, 1080),
        px_per_m=1200.0,
        motion_hint="roll",
        radius_px=50.0,
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 900.0], "p1": [1920.0, 900.0]}],
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(200.0, 900.0 - 50.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=2.20, vy0=0.0, friction=0.50, elasticity=0.65, damping=1.40)
    ts = [2.50 + i * 0.045 for i in range(12)]  # 12 points over 0.495 s
    scene["objects"][0]["track"] = _sample_synthetic_track(scene, true_p, ts, "roll")

    # The default rolling prior is now 0 (rolling resistance does the usual braking); this test is about recovering
    # damping from the track, so it keeps the prior it was written with.
    res = fit_scene(scene, param_priors={"obj0": {"damping": 0.25}})
    p = res.params["obj0"]
    assert p.x0 == pytest.approx(true_p.x0, rel=0.05, abs=0.005)
    assert p.vx0 == pytest.approx(true_p.vx0, rel=0.05)
    assert p.damping == pytest.approx(true_p.damping, rel=0.05)
    assert res.fit_error_px < 2.0


def test_fit_noisy_track_recovers_vel_without_overfitting_priors() -> None:
    """Realistic tracker noise (8 px at 4K) keeps unidentifiable params at priors."""
    scene = _make_scene(
        orig_size=(3840, 2160),
        px_per_m=3700.0,
        motion_hint="roll",
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 1875.0], "p1": [3840.0, 1875.0]}],
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(300.0, 1875.0 - 124.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=1.90, vy0=0.0, friction=0.50, elasticity=0.65, damping=0.25)
    ts = [2.60 + i * (1.0 / 30.0) for i in range(11)]
    scene["objects"][0]["track"] = _sample_synthetic_track(scene, true_p, ts, "roll", noise_px=8.0, seed=7)

    events: list[tuple[str, dict]] = []
    res = fit_scene(scene, on_event=lambda name, payload: events.append((name, payload)))
    p = res.params["obj0"]
    assert p.vx0 == pytest.approx(true_p.vx0, rel=0.05)
    assert p.x0 == pytest.approx(true_p.x0, rel=0.05, abs=0.01)
    assert p.elasticity == pytest.approx(0.65, abs=0.01)
    assert p.friction == pytest.approx(0.50, abs=0.01)
    assert 0.0 <= p.damping <= 1.5
    assert len(res.fit_trajectory) == len(ts)
    assert len(events) == 1
    assert events[0][0] == "fit"
    assert events[0][1]["n_points"] == len(ts)
    assert events[0][1]["attempt"] == 0


def test_fit_free_fall_parabola_recovers_pos_vel() -> None:
    """Free-fall trajectory with fixed gravity recovers pos and vel within 5%."""
    scene = _make_scene(
        orig_size=(1920, 1080),
        px_per_m=1000.0,
        motion_hint="fall",
        radius_px=40.0,
        surfaces=[],
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(400.0, 300.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=1.40, vy0=0.60, friction=0.50, elasticity=0.65, damping=0.02)
    ts = [2.65 + i * 0.04 for i in range(9)]
    scene["objects"][0]["track"] = _sample_synthetic_track(scene, true_p, ts, "fall")

    res = fit_scene(scene)
    p = res.params["obj0"]
    assert p.x0 == pytest.approx(true_p.x0, rel=0.05, abs=0.005)
    assert p.y0 == pytest.approx(true_p.y0, rel=0.05, abs=0.005)
    assert p.vx0 == pytest.approx(true_p.vx0, rel=0.05)
    assert p.vy0 == pytest.approx(true_p.vy0, rel=0.05)
    assert res.fit_error_px < 2.0
