"""Unit tests for backend/physics/world.py."""
from __future__ import annotations

import math

import pytest

from physics.world import BodyParams, CoordTransform, PhysicsWorld


def _base_scene(
    surfaces: list[dict] | None = None,
    static_objects: list[dict] | None = None,
    shape: str = "circle",
) -> dict:
    obj = {
        "id": "obj0",
        "name": "grey tennis ball" if shape == "circle" else "wooden block",
        "shape": shape,
        "radius_px": 50.0 if shape == "circle" else None,
        "w_px": 100.0 if shape == "box" else None,
        "h_px": 80.0 if shape == "box" else None,
        "est_real_size_cm": 6.7,
        "track": [{"t_s": 2.5, "x": 300.0, "y": 750.0}],
        "visible_at_end": True,
        "sprite_png": "/tmp/obj0.png",
    }
    return {
        "clip_id": "synthetic",
        "work_dir": "/tmp/foresight_test",
        "orig_size": [1920, 1080],
        "fps_out": 24,
        "t_cond_end_s": 3.0,
        "pred_seconds": 5.0,
        "px_per_m": 1000.0,
        "px_per_m_source": "size",
        "gravity_m_s2": 9.81,
        "motion_type_hint": "roll",
        "objects": [obj],
        "static_objects": static_objects or [],
        "surfaces": surfaces if surfaces is not None else [
            {"name": "table", "kind": "table", "p0": [0.0, 800.0], "p1": [1000.0, 800.0]},
        ],
        "background_png": "/tmp/bg.png",
        "last_frame_png": "/tmp/last.png",
        "keyframes": [],
    }


def test_coord_transform_roundtrip() -> None:
    c = CoordTransform(width_px=3840, height_px=2160, px_per_m=3700.0)
    xm, ym = c.px_to_m(288.0, 1744.0)
    xpx, ypx = c.m_to_px(xm, ym)
    assert xpx == pytest.approx(288.0, abs=1e-9)
    assert ypx == pytest.approx(1744.0, abs=1e-9)
    # y DOWN -> y UP: smaller y_px means larger y_m
    assert c.px_to_m(0.0, 100.0)[1] > c.px_to_m(0.0, 500.0)[1]


def test_ball_rolls_off_finite_table() -> None:
    """Ball rolls right on finite table [0, 1.0 m], then falls off the end under g=9.81."""
    scene = _base_scene()
    c = CoordTransform.from_scene(scene)
    # Table top at y_px=800 -> y_m = 0.28; ball radius = 0.05 -> center y_m = 0.33
    x0, y0 = c.px_to_m(300.0, 750.0)
    params = BodyParams(x0=x0, y0=y0, vx0=1.5, vy0=0.0, friction=0.5, elasticity=0.65, damping=0.0)
    w = PhysicsWorld(scene, {"obj0": params}, t_start_s=0.0, motion_model="roll")

    # At t = 0.3 s, x = 0.3 + 1.5*0.3 = 0.75 m (still on table ending at 1.0 m)
    w.step_to(0.3)
    st = w.get_state_m()["obj0"]
    assert st["x"] == pytest.approx(0.75, abs=0.02)
    assert st["y"] == pytest.approx(y0, abs=0.005)
    # Rolling right => clockwise rotation => negative angle in y-up
    assert st["angle"] < -5.0

    # At t = 0.8 s (past x = 1.0 m since t ~ 0.47 s), ball must have fallen well below table!
    w.step_to(0.8)
    st2 = w.get_state_m()["obj0"]
    assert st2["x"] > 1.1
    assert st2["y"] < y0 - 0.35
    assert st2["vy"] < -2.0


def test_rolling_damping_decays_exponentially() -> None:
    """Per-body damping slows a rolling ball on a long table without affecting free fall."""
    scene = _base_scene(
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 800.0], "p1": [5000.0, 800.0]}]
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(200.0, 750.0)
    gamma = 0.8
    v0 = 2.0
    params = BodyParams(x0=x0, y0=y0, vx0=v0, vy0=0.0, friction=0.5, elasticity=0.65, damping=gamma)
    w = PhysicsWorld(scene, {"obj0": params}, t_start_s=0.0, motion_model="roll")
    w.step_to(1.0)
    st = w.get_state_m()["obj0"]
    expected_v = v0 * math.exp(-gamma * 1.0)
    assert st["vx"] == pytest.approx(expected_v, rel=0.03)


def test_free_fall_exact_gravity_despite_damping() -> None:
    """In mid-air, vertical acceleration matches fixed g=9.81 even when rolling damping > 0."""
    scene = _base_scene(surfaces=[])
    params = BodyParams(x0=0.5, y0=5.0, vx0=0.5, vy0=0.0, damping=1.0)
    w = PhysicsWorld(scene, {"obj0": params}, t_start_s=0.0, motion_model="fall")
    w.step_to(0.5)
    st = w.get_state_m()["obj0"]
    assert st["x"] == pytest.approx(0.75, abs=0.005)
    assert st["y"] == pytest.approx(5.0 - 0.5 * 9.81 * 0.25, abs=0.01)
    assert st["vy"] == pytest.approx(-9.81 * 0.5, abs=0.02)


def test_bounce_respects_exact_elasticity() -> None:
    """Surface elasticity=1.0 ensures dynamic body elasticity is exact restitution e."""
    scene = _base_scene(
        surfaces=[{"name": "floor", "kind": "floor", "p0": [0.0, 1000.0], "p1": [1920.0, 1000.0]}]
    )
    # Floor at y_m = 0.08; ball radius = 0.05; drop from h = 0.50 m above contact (y0 = 0.63)
    e = 0.72
    params = BodyParams(x0=0.5, y0=0.63, vx0=0.0, vy0=0.0, friction=0.5, elasticity=e, damping=0.0)
    w = PhysicsWorld(scene, {"obj0": params}, t_start_s=0.0, motion_model="bounce")
    max_y_after_bounce = 0.0
    bounced = False
    for k in range(1, 200):
        w.step_to(k * 0.005)
        st = w.get_state_m()["obj0"]
        if not bounced and st["vy"] > 0.1:
            bounced = True
        if bounced:
            max_y_after_bounce = max(max_y_after_bounce, st["y"])
    assert bounced
    rebound_h = max_y_after_bounce - (0.08 + 0.05)
    expected_h = 0.50 * (e ** 2)
    assert rebound_h == pytest.approx(expected_h, rel=0.04)


def test_static_box_collision_and_overlapping_pipe_skip() -> None:
    """Ahead obstacle deflects ball; pipe overlapping initial track is safely skipped."""
    scene = _base_scene(
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 800.0], "p1": [1920.0, 800.0]}],
        static_objects=[
            # Pipe overlapping ball track at x=300, y=750, r=50 -> must be skipped
            {"name": "launch pipe", "shape": "box", "bbox": [0, 680, 285, 810]},
            # Wall ahead at x=600..650 -> must deflect the ball!
            {"name": "block wall", "shape": "box", "bbox": [600, 600, 650, 800]},
        ],
    )
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(300.0, 750.0)
    params = BodyParams(x0=x0, y0=y0, vx0=1.5, vy0=0.0, friction=0.5, elasticity=0.8, damping=0.0)
    w = PhysicsWorld(scene, {"obj0": params}, t_start_s=0.0, motion_model="roll")
    assert "launch pipe" in w.skipped_static
    assert "block wall" not in w.skipped_static

    # Ball should travel right, bounce off wall at x=0.6 m, and reverse direction (vx < 0)
    w.step_to(0.35)
    st = w.get_state_m()["obj0"]
    assert st["vx"] < -0.5
    assert st["x"] < 0.6


def test_sliding_box_decelerates_with_friction() -> None:
    """A dynamic box sliding on a table decelerates at a = -mu * g."""
    scene = _base_scene(
        surfaces=[{"name": "table", "kind": "table", "p0": [0.0, 800.0], "p1": [5000.0, 800.0]}],
        shape="box",
    )
    c = CoordTransform.from_scene(scene)
    # Box h_px=80 -> half_h = 0.04 m; table at y_px=800 -> center at y_px=760
    x0, y0 = c.px_to_m(200.0, 760.0)
    mu = 0.30
    v0 = 2.0
    params = BodyParams(x0=x0, y0=y0, vx0=v0, vy0=0.0, friction=mu, elasticity=0.2, damping=0.0)
    w = PhysicsWorld(scene, {"obj0": params}, t_start_s=0.0, motion_model="slide")
    w.step_to(0.4)
    st = w.get_state_m()["obj0"]
    expected_v = v0 - mu * 9.81 * 0.4
    assert st["vx"] == pytest.approx(expected_v, rel=0.04)
