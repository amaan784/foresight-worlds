"""Offline unit and CLI tests for Task A1: `foresight audit` physicality auditor (0 Gemini calls)."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pytest

import foresight
from foresight import cli, config
from foresight.audit import audit_continuation, audit_video
from foresight.video import write_mp4
from physics.render import render_trajectory


def _make_synthetic_scene(
    tmp_path: Path,
    *,
    static_objects: list[dict] | None = None,
    w: int = 960,
    h: int = 540,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Create a textured synthetic scene (1000 px/m) with a striped teal ball on a flat table at y=400."""
    rng = np.random.default_rng(42)
    bg = np.full((h, w, 3), (44, 48, 56), dtype=np.uint8)
    noise = rng.integers(-6, 7, size=(h, w, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(bg, (0, 400), (w, h), (72, 108, 142), -1)
    # Launch pipe visual on the far left (outside tracked path)
    cv2.rectangle(bg, (0, 336), (60, 400), (24, 24, 28), -1)

    for st in static_objects or []:
        x0, y0, x1, y1 = [int(round(v)) for v in st["bbox"]]
        cv2.rectangle(bg, (x0, y0), (x1, y1), (55, 85, 165), -1)
        cv2.rectangle(bg, (x0, y0), (x1, y1), (95, 130, 210), 2)

    bg_path = tmp_path / "background.png"
    cv2.imwrite(str(bg_path), bg)

    # Radius 30 px circle sprite in 68x68 RGBA canvas with high-contrast cross pattern for CSRT
    sp = np.zeros((68, 68, 4), dtype=np.uint8)
    cv2.circle(sp, (34, 34), 30, (60, 225, 185, 255), -1)
    cv2.line(sp, (10, 34), (58, 34), (250, 250, 255, 255), 4)
    cv2.line(sp, (34, 10), (34, 58), (30, 70, 190, 255), 4)
    sp_path = tmp_path / "obj0.png"
    cv2.imwrite(str(sp_path), sp)

    # Conditioning track over [2.50, 2.9583] rolling right at 400 px/s (0.80 m/s at 500 px/m)
    fps = 24
    cond_pts = []
    for k in range(12):
        t_s = round(2.5 + k / fps, 4)
        x = round(160.0 + (t_s - 2.5) * 400.0, 1)
        cond_pts.append({"t_s": t_s, "x": x, "y": 370.0})

    scene = {
        "clip_id": "synthetic_audit_roll",
        "work_dir": str(tmp_path),
        "orig_size": [w, h],
        "fps_out": fps,
        "t_cond_end_s": 3.0,
        "pred_seconds": 5.0,
        "px_per_m": 500.0,
        "px_per_m_source": "size",
        "gravity_m_s2": 9.81,
        "motion_type_hint": "roll",
        "objects": [
            {
                "id": "obj0",
                "name": "striped ball",
                "shape": "circle",
                "radius_px": 30.0,
                "w_px": None,
                "h_px": None,
                "est_real_size_cm": 12.0,
                "est_mass_g": 80.0,
                "material": "rubber",
                "track": cond_pts,
                "visible_at_end": True,
                "sprite_png": str(sp_path),
            }
        ],
        "static_objects": static_objects or [],
        "surfaces": [{"name": "table", "kind": "table", "p0": [0.0, 400.0], "p1": [float(w), 400.0]}],
        "background_png": str(bg_path),
        "last_frame_png": str(bg_path),
        "keyframes": [],
    }
    return scene, bg, sp


def _paste_rgba(dst: np.ndarray, sp: np.ndarray, cx: float, cy: float) -> None:
    sh, sw = sp.shape[:2]
    x0, y0 = int(round(cx - sw / 2.0)), int(round(cy - sh / 2.0))
    H, W = dst.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + sw), min(H, y0 + sh)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = sp[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = dst[cy0:cy1, cx0:cx1].astype(np.float32)
    dst[cy0:cy1, cx0:cx1] = (roi * (1.0 - a) + sub[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def _write_cond_mp4(path: Path, bg: np.ndarray, sp: np.ndarray, scene: dict) -> Path:
    fps = int(scene["fps_out"])
    n = int(round(scene["t_cond_end_s"] * fps))
    frames = []
    for k in range(n):
        t_s = k / fps
        f = bg.copy()
        if t_s >= 2.5:
            x = 160.0 + (t_s - 2.5) * 400.0
            _paste_rgba(f, sp, x, 370.0)
        frames.append(f)
    return write_mp4(path, frames, fps)


def test_audit_plausible_rollout_matches_reality(tmp_path: Path) -> None:
    scene, bg, sp = _make_synthetic_scene(tmp_path)
    cond_mp4 = _write_cond_mp4(tmp_path / "cond.mp4", bg, sp, scene)

    # Decelerating rollout that comes to rest at x = 760 px on the table
    fps = 24
    traj = []
    for k in range(120):
        dt = k / fps
        # Smooth braking over 2.0 s from v0=400 px/s at x0=360 to rest at x=760
        tau = min(dt, 2.0)
        x = 360.0 + 400.0 * tau - 100.0 * tau * tau
        traj.append({"t_s": round(3.0 + dt, 4), "objects": {"obj0": {"x": x, "y": 370.0, "angle": 0.0, "visible": True}}})

    real_mp4 = render_trajectory(scene, traj, tmp_path / "real.mp4")
    pred_mp4 = render_trajectory(scene, traj, tmp_path / "pred.mp4")

    report = audit_video(cond_mp4, pred_mp4, reality_video=real_mp4, scene=scene)
    assert report["verdict"]["passed"] is True
    assert report["permanence"]["vanished"] is False
    assert report["permanence"]["duplicated"] is False
    assert report["permanence"]["status"] == "visible_to_end"
    assert report["plausibility"]["plausible"] is True
    assert report["position_error"] is not None
    assert report["position_error"]["mean_penalized_cm"] is not None
    assert report["position_error"]["mean_penalized_cm"] < 2.0
    for row in report["position_error"]["by_time"]:
        assert row["status"] == "tracked"
        assert row["penalized_err_cm"] is not None and row["penalized_err_cm"] < 2.0


def test_audit_vanishing_and_duplicate_respawn_penalizes_fixed_times(tmp_path: Path) -> None:
    scene, bg, sp = _make_synthetic_scene(tmp_path)
    cond_mp4 = _write_cond_mp4(tmp_path / "cond.mp4", bg, sp, scene)
    fps = 24

    # Reality: ball rolls smoothly and rests on the table at x = 760
    real_frames = []
    for k in range(120):
        dt = k / fps
        tau = min(dt, 2.0)
        x_real = 360.0 + 400.0 * tau - 100.0 * tau * tau
        f = bg.copy()
        _paste_rgba(f, sp, x_real, 370.0)
        real_frames.append(f)
    real_mp4 = write_mp4(tmp_path / "real.mp4", real_frames, fps)

    # Prediction: ball rolls until +0.75 s (frame 18), vanishes mid-table at x~580,
    # and a duplicate ball respawns near the launch pipe (x=150..260) from +1.25 s onward.
    pred_frames = []
    for k in range(120):
        dt = k / fps
        f = bg.copy()
        if k < 18:
            _paste_rgba(f, sp, 360.0 + 300.0 * dt, 370.0)
        if k >= 30:
            x_dup = min(250.0, 145.0 + 180.0 * (dt - 30 / fps))
            _paste_rgba(f, sp, x_dup, 370.0)
        pred_frames.append(f)
    pred_mp4 = write_mp4(tmp_path / "pred_vanish_dup.mp4", pred_frames, fps)

    report = audit_continuation(cond_mp4, pred_mp4, reality_video=real_mp4, scene=scene)
    assert report["verdict"]["passed"] is False
    assert report["permanence"]["vanished"] is True
    assert report["permanence"]["duplicated"] is True

    pos = report["position_error"]
    assert pos is not None
    vanished_rows = [r for r in pos["by_time"] if r["dt_s"] >= 1.0]
    assert len(vanished_rows) == 4
    for r in vanished_rows:
        assert r["status"] == "vanished"
        assert r["raw_err_cm"] is None
        assert r["penalized_err_cm"] is not None
        assert r["penalized_err_cm"] >= pos["off_course_floor_cm"]


def test_audit_foreground_occluder_is_hidden_not_vanished(tmp_path: Path) -> None:
    # Foreground box bottom is at y=475 > table_y(400) + 1.2 * r(30) = 436 -> nearer camera -> occluder in layers.py
    fg_box = {
        "name": "foreground crate",
        "shape": "box",
        "bbox": [520.0, 310.0, 760.0, 475.0],
        "perceived_shape": "box",
        "est_real_size_cm": 28.0,
        "est_mass_g": 2500.0,
        "material": "hard",
    }
    scene, bg, sp = _make_synthetic_scene(tmp_path, static_objects=[fg_box])
    cond_mp4 = _write_cond_mp4(tmp_path / "cond.mp4", bg, sp, scene)

    fps = 24
    traj = []
    for k in range(120):
        dt = k / fps
        x = min(640.0, 360.0 + 320.0 * dt)  # rolls behind the foreground crate and stops hidden behind it
        traj.append({"t_s": round(3.0 + dt, 4), "objects": {"obj0": {"x": x, "y": 370.0, "angle": 0.0, "visible": True}}})
    pred_mp4 = render_trajectory(scene, traj, tmp_path / "pred_occluded.mp4")

    report = audit_video(cond_mp4, pred_mp4, scene=scene)
    assert report["permanence"]["status"] == "occluded"
    assert report["permanence"]["occluded"] is True
    assert report["permanence"]["vanished"] is False
    assert report["verdict"]["passed"] is True


def test_audit_energy_blowup_fails_plausibility(tmp_path: Path) -> None:
    scene, bg, sp = _make_synthetic_scene(tmp_path)
    cond_mp4 = _write_cond_mp4(tmp_path / "cond.mp4", bg, sp, scene)

    # Conditioning speed v0 is 0.80 m/s on a level table (drop_m = 0 -> bound ~ 1.34 m/s).
    # After 0.25 s, accelerate unphysically on the flat table to 3.2 m/s (1600 px/s -> ~67 px/frame).
    fps = 24
    frames = []
    x = 360.0
    for k in range(120):
        vx = 400.0 if k < 6 else 1600.0
        if k > 0:
            x += vx / fps
        f = bg.copy()
        if x < 930.0:
            _paste_rgba(f, sp, x, 370.0)
        frames.append(f)
    pred_mp4 = write_mp4(tmp_path / "pred_blowup.mp4", frames, fps)

    report = audit_video(cond_mp4, pred_mp4, scene=scene)
    assert report["plausibility"]["plausible"] is False
    assert report["verdict"]["passed"] is False
    assert len(report["plausibility"]["violations"]) >= 1
    assert report["plausibility"]["max_speed_m_s"] > report["plausibility"]["max_bound_m_s"]


def test_cli_audit_writes_json_and_self_contained_html(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pyproject = tomllib.loads((config.ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["foresight"] == "foresight.cli:main"
    assert foresight.audit_video is audit_video
    assert foresight.audit_continuation is audit_continuation

    scene, bg, sp = _make_synthetic_scene(tmp_path)
    cond_mp4 = _write_cond_mp4(tmp_path / "cond.mp4", bg, sp, scene)
    traj = [
        {"t_s": round(3.0 + k / 24, 4), "objects": {"obj0": {"x": min(760.0, 360.0 + 250.0 * (k / 24)), "y": 370.0, "angle": 0.0, "visible": True}}}
        for k in range(120)
    ]
    pred_mp4 = render_trajectory(scene, traj, tmp_path / "pred.mp4")
    real_mp4 = render_trajectory(scene, traj, tmp_path / "real.mp4")
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps(scene))
    out_dir = tmp_path / "audit_out"

    rc = cli.main(
        [
            "audit",
            str(cond_mp4),
            str(pred_mp4),
            "--reality",
            str(real_mp4),
            "--scene",
            str(scene_json),
            "--physics-iq-score",
            "91.5",
            "--label",
            "foresight-unit",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["passed"] is True
    assert stdout_summary["physics_iq_score_pct"] == pytest.approx(91.5)

    json_path = out_dir / "audit.json"
    html_path = out_dir / "audit.html"
    assert json_path.exists() and html_path.exists()

    saved = json.loads(json_path.read_text())
    assert saved["label"] == "foresight-unit"
    assert saved["physics_iq_score_pct"] == pytest.approx(91.5)
    assert saved["verdict"]["passed"] is True

    html_txt = html_path.read_text()
    assert "<!doctype html>" in html_txt.lower()
    assert "<svg" in html_txt
    assert "91.50%" in html_txt
    assert "Fixed-Time Position Error" in html_txt
