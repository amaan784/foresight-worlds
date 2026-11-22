"""Offline tests for Task A3: interactive counterfactual ('what-if') physics rollouts."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from foresight import api, whatif
from foresight.pipeline import endpoint_error
from physics import BodyParams, CoordTransform, simulate_future


async def _asgi_json(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    body = json.dumps(payload).encode() if payload is not None else b""
    headers = [(b"content-type", b"application/json")] if payload is not None else []
    sent = False
    status_code = 500
    chunks: list[bytes] = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg):
        nonlocal status_code
        if msg["type"] == "http.response.start":
            status_code = msg["status"]
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method.upper(), "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": headers, "client": ("127.0.0.1", 5173), "server": ("127.0.0.1", 8000),
    }
    await api.app(scope, receive, send)
    return status_code, json.loads(b"".join(chunks) or b"{}")


def _write_saved_job(
    out_dir: Path,
    job_id: str,
    *,
    motion_model: str = "roll",
    y_track_px: float = 580.0,
    vx0: float = 0.45,
    vy0: float = 0.0,
    elasticity: float = 0.65,
    const_velocity_only: bool = False,
) -> tuple[list[list[float]], float | None]:
    job_dir = out_dir / job_id
    (job_dir / "work").mkdir(parents=True, exist_ok=True)
    scene = {
        "orig_size": [1280, 720],
        "fps_out": 24,
        "t_cond_end_s": 3.0,
        "pred_seconds": 5.0,
        "px_per_m": 800.0,
        "gravity_m_s2": 9.81,
        "motion_type_hint": motion_model,
        "objects": [{
            "id": "obj0", "name": "rubber ball", "shape": "circle", "material": "rubber",
            "radius_px": 20.0, "est_real_size_cm": 5.0, "est_mass_g": 60.0,
            "track": [
                {"t_s": 0.5 + 0.25 * k, "x": 200.0 + 90.0 * k, "y": y_track_px}
                for k in range(10)
            ],
        }],
        "surfaces": [{"name": "table", "kind": "table", "material": "hard", "p0": [0.0, 600.0], "p1": [1280.0, 600.0]}],
        "static_objects": [],
    }
    (job_dir / "work" / "scene.json").write_text(json.dumps(scene))
    coord = CoordTransform.from_scene(scene)
    x0, y0 = coord.px_to_m(200.0, y_track_px)
    bp = BodyParams(x0=x0, y0=y0, vx0=vx0, vy0=vy0, friction=0.45, elasticity=elasticity, damping=0.02, t_start_s=0.5)
    traj = simulate_future(scene, {"obj0": bp}, t_start_s=0.5, motion_model=motion_model)
    pred = [[st["t_s"], st["objects"]["obj0"]["x"], st["objects"]["obj0"]["y"]] for st in traj]
    real_pts = [{"t_s": p[0], "x": p[1] + 12.0, "y": p[2]} for p in pred[::6]]
    ee = endpoint_error(traj, real_pts, scene, "obj0")

    # Deliberately omit t_start_s in serialized fit_params to test pre-C5 saved job compatibility too.
    fp_dict = bp.to_dict()
    fp_dict.pop("t_start_s", None)
    result = {
        "job_id": job_id, "status": "done",
        "dynamics": "const_velocity" if const_velocity_only else "physics",
        "physics": None if const_velocity_only else {"motion_model": motion_model, "fit_params": {"obj0": fp_dict}},
        "metrics": {"dynamics_used": "const_velocity" if const_velocity_only else "physics",
                    "endpoint_error_cm": {"foresight": ee["cm"] if ee else None}},
        "paths": {"orig_size": [1280, 720], "seen": [], "predicted": pred,
                  "real": [[p["t_s"], p["x"], p["y"]] for p in real_pts]},
    }
    (job_dir / "result.json").write_text(json.dumps(result))
    return pred, (ee["cm"] if ee else None)


def test_whatif_identity_matches_baseline_on_disk_and_in_memory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(api, "OUT", out_dir)
    monkeypatch.setattr(api, "JOBS", {})
    with whatif._CACHE_LOCK:
        whatif._CACHE.clear()

    pred_saved, ee_cm = _write_saved_job(out_dir, "j_roll0001")

    code, base = asyncio.run(_asgi_json("GET", "/api/jobs/j_roll0001/whatif"))
    assert code == 200
    assert base["object_id"] == "obj0"
    assert base["endpoint_error_cm"] == pytest.approx(ee_cm)

    code, ident = asyncio.run(_asgi_json("POST", "/api/jobs/j_roll0001/whatif", {}))
    assert code == 200
    assert ident["latency_ms"] < 200.0
    assert len(ident["predicted"]) == len(pred_saved) == 120
    max_diff = max(math.hypot(a[1] - b[1], a[2] - b[2]) for a, b in zip(ident["predicted"], pred_saved))
    assert max_diff < 1e-3
    assert ident["endpoint_error_cm"] == pytest.approx(ee_cm)

    # Also verify in-memory jobs and const_velocity saved jobs work cleanly.
    _write_saved_job(out_dir, "j_cv000001", const_velocity_only=True)
    code_cv, cv_res = asyncio.run(_asgi_json("POST", "/api/jobs/j_cv000001/whatif", {"vx0": 0.8}))
    assert code_cv == 200 and len(cv_res["predicted"]) == 120

    mem_dir = out_dir / "j_mem00001"
    _write_saved_job(out_dir, "j_mem00001")
    mem_res = json.loads((mem_dir / "result.json").read_text())
    (mem_dir / "result.json").unlink()
    monkeypatch.setattr(api, "JOBS", {"j_mem00001": SimpleNamespace(id="j_mem00001", dir=mem_dir, result=mem_res)})
    code_m, mem_out = asyncio.run(_asgi_json("GET", "/api/jobs/j_mem00001/whatif"))
    assert code_m == 200 and mem_out["job_id"] == "j_mem00001"


def test_whatif_velocity_bounce_start_drag_and_clamping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(api, "OUT", out_dir)
    monkeypatch.setattr(api, "JOBS", {})
    with whatif._CACHE_LOCK:
        whatif._CACHE.clear()

    _write_saved_job(out_dir, "j_roll0002", vx0=0.90)
    _, base = asyncio.run(_asgi_json("GET", "/api/jobs/j_roll0002/whatif"))
    assert base["vx0"] > 0.35
    _, slow = asyncio.run(_asgi_json("POST", "/api/jobs/j_roll0002/whatif", {"vx0": base["vx0"]}))
    _, fast = asyncio.run(_asgi_json("POST", "/api/jobs/j_roll0002/whatif", {"vx0": 2.0 * base["vx0"]}))
    assert fast["predicted"][-1][1] > slow["predicted"][-1][1] + 150.0

    # Dragging start_px shifts frame-0 rollout origin to exact dragged pixel coordinates.
    _, dragged = asyncio.run(_asgi_json("POST", "/api/jobs/j_roll0002/whatif", {"start_px": [640.0, 520.0]}))
    assert dragged["predicted"][0][1] == pytest.approx(640.0, abs=1e-3)
    assert dragged["predicted"][0][2] == pytest.approx(520.0, abs=1e-3)

    # Increasing bounce (0.1 -> 0.9) on a falling ball increases rebound height after floor impact.
    _write_saved_job(out_dir, "j_bounce01", motion_model="bounce", y_track_px=240.0, vx0=0.2, vy0=0.0, elasticity=0.5)
    _, low_b = asyncio.run(_asgi_json("POST", "/api/jobs/j_bounce01/whatif", {"start_px": [300.0, 260.0], "vy0": 0.0, "bounce": 0.1}))
    _, high_b = asyncio.run(_asgi_json("POST", "/api/jobs/j_bounce01/whatif", {"start_px": [300.0, 260.0], "vy0": 0.0, "bounce": 0.9}))
    impact_idx = max(range(len(low_b["predicted"])), key=lambda i: low_b["predicted"][i][2])
    apex_low_y_px = min(p[2] for p in low_b["predicted"][impact_idx:])
    apex_high_y_px = min(p[2] for p in high_b["predicted"][impact_idx:])
    assert apex_high_y_px < apex_low_y_px - 80.0  # smaller y_px = higher rebound apex in image

    # Out-of-bounds overrides clamp cleanly instead of blowing up the solver.
    code_c, clamped = asyncio.run(_asgi_json("POST", "/api/jobs/j_roll0002/whatif", {"vx0": 999.0, "friction": 9.0, "bounce": 5.0}))
    assert code_c == 200
    assert clamped["vx0"] == pytest.approx(whatif.MAX_SPEED_M_S)
    assert clamped["friction"] == pytest.approx(whatif.MAX_FRICTION)
    assert clamped["bounce"] == pytest.approx(whatif.MAX_BOUNCE)


def test_whatif_rejects_unknown_and_traversal_job_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "result.json").write_text("{}")
    monkeypatch.setattr(api, "OUT", out_dir)
    monkeypatch.setattr(api, "JOBS", {})

    for bad_id in ("../secret", "..", ".hidden", "bad..id", "a/b", ""):
        with pytest.raises(whatif.InvalidJobIdError):
            whatif.validate_job_id(bad_id, out_dir)

    code_bad, _ = asyncio.run(_asgi_json("GET", "/api/jobs/.hidden/whatif"))
    assert code_bad == 400
    code_dots, _ = asyncio.run(_asgi_json("POST", "/api/jobs/bad..id/whatif", {}))
    assert code_dots == 400
    code_404, _ = asyncio.run(_asgi_json("GET", "/api/jobs/j_missing99/whatif"))
    assert code_404 == 404

    _write_saved_job(out_dir, "j_valid001")
    code_obj, _ = asyncio.run(_asgi_json("POST", "/api/jobs/j_valid001/whatif", {"object_id": "nope"}))
    assert code_obj == 400
    code_px, _ = asyncio.run(_asgi_json("POST", "/api/jobs/j_valid001/whatif", {"start_px": [100.0]}))
    assert code_px == 400
