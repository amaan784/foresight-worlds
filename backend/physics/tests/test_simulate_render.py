"""Unit tests for backend/physics/simulate.py and backend/physics/render.py."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from physics.fit import fit_scene
from physics.render import _vidio, render_trajectory
from physics.simulate import evaluate_at_times, simulate_future
from physics.world import BodyParams, CoordTransform, PhysicsWorld


def _make_render_scene(tmp_path: Path, orig_size: tuple[int, int] = (1920, 1080)) -> dict:
    w, h = orig_size
    bg = np.full((h, w, 3), (45, 50, 55), dtype=np.uint8)
    # Draw a wooden table at y = 800 from x = 0 to x = 1100
    cv2.rectangle(bg, (0, 800), (1100, h), (70, 110, 150), -1)
    bg_path = tmp_path / "background.png"
    cv2.imwrite(str(bg_path), bg)

    # Create RGBA tennis ball sprite (radius 50 px) centered in 120x120 canvas with a stripe
    sprite = np.zeros((120, 120, 4), dtype=np.uint8)
    cv2.circle(sprite, (60, 60), 50, (80, 220, 190, 255), -1)
    cv2.line(sprite, (20, 60), (100, 60), (255, 255, 255, 255), 6)
    sprite_path = tmp_path / "obj0.png"
    cv2.imwrite(str(sprite_path), sprite)

    return {
        "clip_id": "synthetic_sim_render",
        "work_dir": str(tmp_path),
        "orig_size": [w, h],
        "fps_out": 24,
        "t_cond_end_s": 3.0,
        "pred_seconds": 5.0,
        "px_per_m": 1000.0,
        "px_per_m_source": "size",
        "gravity_m_s2": 9.81,
        "motion_type_hint": "roll",
        "objects": [
            {
                "id": "obj0",
                "name": "tennis ball",
                "shape": "circle",
                "radius_px": 50.0,
                "w_px": None,
                "h_px": None,
                "est_real_size_cm": 6.7,
                "track": [],
                "visible_at_end": True,
                "sprite_png": str(sprite_path),
            }
        ],
        "static_objects": [],
        "surfaces": [
            {"name": "table", "kind": "table", "p0": [0.0, 800.0], "p1": [1100.0, 800.0]},
        ],
        "background_png": str(bg_path),
        "last_frame_png": str(bg_path),
        "keyframes": [],
    }


def test_simulate_future_exact_120_frames_and_falls_off_screen(tmp_path: Path) -> None:
    scene = _make_render_scene(tmp_path)
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(300.0, 750.0)
    true_p = BodyParams(x0=x0, y0=y0, vx0=1.2, vy0=0.0, friction=0.5, elasticity=0.65, damping=0.1)

    # Generate synthetic conditioning track over [2.5, 2.95]
    ts = [2.5 + i * 0.05 for i in range(10)]
    w = PhysicsWorld(scene, {"obj0": true_p}, t_start_s=ts[0], motion_model="roll")
    for t in ts:
        w.step_to(t)
        st = w.get_state_px()["obj0"]
        scene["objects"][0]["track"].append({"t_s": t, "x": st["x"], "y": st["y"]})

    fit_res = fit_scene(scene)
    traj = simulate_future(scene, fit_res)

    assert len(traj) == 120
    assert traj[0]["t_s"] == pytest.approx(3.0, abs=1e-9)
    assert traj[-1]["t_s"] == pytest.approx(3.0 + 119.0 / 24.0, abs=1e-9)

    # Initially visible on table; later falls off table end (x > 1100) and leaves bottom of frame
    assert traj[0]["objects"]["obj0"]["visible"] is True
    assert traj[0]["objects"]["obj0"]["y"] == pytest.approx(750.0, abs=10.0)
    assert traj[-1]["objects"]["obj0"]["visible"] is False
    assert traj[-1]["objects"]["obj0"]["y"] > 1080.0

    # evaluate_at_times works for keyframe timestamps inside conditioning window
    kf_states = evaluate_at_times(scene, fit_res.params, fit_res.t_start_s, [2.55, 2.75, 2.95], "roll")
    assert len(kf_states) == 3
    assert kf_states[0]["objects"]["obj0"]["x"] < kf_states[2]["objects"]["obj0"]["x"]


def test_render_trajectory_exact_5s_mp4_and_offscreen_skip(tmp_path: Path) -> None:
    scene = _make_render_scene(tmp_path)
    c = CoordTransform.from_scene(scene)
    x0, y0 = c.px_to_m(500.0, 750.0)
    params = {"obj0": BodyParams(x0=x0, y0=y0, vx0=1.4, vy0=0.0, friction=0.5, elasticity=0.65, damping=0.1)}
    traj = simulate_future(scene, params, t_start_s=3.0, motion_model="roll")

    out_mp4 = tmp_path / "foresight.mp4"
    events: list[tuple[str, dict]] = []
    written = render_trajectory(
        scene, traj, out_mp4, out_size=(960, 540),
        on_event=lambda name, payload: events.append((name, payload)),
    )
    assert written == out_mp4
    assert out_mp4.exists()

    vidio = _vidio()
    info = vidio.probe(out_mp4)
    assert abs(info["duration"] - 5.000) <= 0.001
    assert info["frames"] == 120
    assert info["fps"] == pytest.approx(24.0, abs=0.01)
    assert info["width"] == 960
    assert info["height"] == 540

    # First frame has the bright green/teal ball composited; final frame matches clean background
    frames, _ = vidio.read_frames(out_mp4)
    assert len(frames) == 120
    bg_small = cv2.resize(cv2.imread(scene["background_png"]), (960, 540), interpolation=cv2.INTER_AREA)
    diff_first = float(np.abs(frames[0].astype(np.float32) - bg_small.astype(np.float32)).max())
    diff_last = float(np.abs(frames[-1].astype(np.float32) - bg_small.astype(np.float32)).mean())
    assert diff_first > 80.0
    assert diff_last < 3.0

    assert len(events) == 1
    assert events[0][0] == "render"
    assert events[0][1]["frames"] == 120
    assert events[0][1]["duration_s"] == pytest.approx(5.0, abs=0.001)
