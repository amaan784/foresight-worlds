"""Offline tests for Task C2: phone capture honors stage Omni/Genie toggles via QR URL and server session state."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pytest

from foresight import api, config, genie, omni
from foresight.video import write_mp4


async def _asgi_request(
    method: str,
    path: str,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    body: bytes = b"",
) -> tuple[int, bytes]:
    sent = False
    status_code = 500
    chunks: list[bytes] = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    raw_path, _, query = path.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "headers": headers or [],
        "client": ("127.0.0.1", 5173),
        "server": ("127.0.0.1", 8000),
    }
    await api.app(scope, receive, send)
    return status_code, b"".join(chunks)


def _multipart_body(fields: dict[str, str], file_field: tuple[str, str, str, bytes]) -> tuple[bytes, bytes]:
    boundary = "----ForesightC2Boundary987654"
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    fname_key, filename, content_type, data = file_field
    parts.append(
        (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{fname_key}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}".encode(), b"".join(parts)


def _make_dummy_clip(path: Path) -> bytes:
    frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(96)]  # 4.0 s @ 24 FPS
    write_mp4(path, frames, 24)
    return path.read_bytes()


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
    calls = {"omni": 0, "genie": 0, "predict_clip": 0}
    monkeypatch.setattr(api, "OUT", tmp_path / "outputs")
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "JOBS", {})
    monkeypatch.setattr(api, "SESSIONS", {})
    monkeypatch.setattr(config, "OMNI_ENABLED", True)
    monkeypatch.setattr(config, "GENIE_ENABLED", True)

    def _fake_omni(cond_video: Path, out_5s: Path, prompt: str = "", tag: str = "") -> dict:
        calls["omni"] += 1
        write_mp4(out_5s, [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(120)], 24)
        return {"latency_s": 0.1}

    def _fake_genie(cond_video: Path, out_5s: Path, prompt: str = "", tag: str = "") -> dict:
        calls["genie"] += 1
        write_mp4(out_5s, [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(120)], 24)
        return {"latency_s": 0.1}

    def _fake_predict_clip(cond_video: Path, out_video: Path, **kwargs) -> dict:
        calls["predict_clip"] += 1
        frames = [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(120)]
        write_mp4(out_video, frames, 24)
        scene = {
            "orig_size": [96, 64],
            "fps_out": 24,
            "px_per_m": 500.0,
            "px_per_m_source": "size",
            "gravity_m_s2": 9.81,
            "motion_type_hint": "roll",
            "objects": [
                {
                    "id": "obj0",
                    "name": "ball",
                    "shape": "circle",
                    "radius_px": 8.0,
                    "est_real_size_cm": 6.7,
                    "est_mass_g": 58.0,
                    "track": [{"t_s": 2.9, "x": 20.0, "y": 40.0}],
                }
            ],
            "static_objects": [],
            "surfaces": [{"name": "table", "kind": "table", "material": "hard", "p0": [0.0, 48.0], "p1": [96.0, 48.0]}],
        }
        traj = [{"t_s": 3.0 + k / 24.0, "objects": {"obj0": {"x": 22.0 + k * 0.5, "y": 40.0, "angle": 0.0, "visible": True}}} for k in range(120)]
        return {
            "status": "done",
            "dynamics": kwargs.get("dynamics", "physics"),
            "renderer": "physics.predict",
            "prediction": str(out_video),
            "probe": {"duration": 5.0, "fps": 24.0, "frames": 120, "width": 96, "height": 64},
            "metrics": {
                "latency_s_total": 0.2,
                "latency_s_by_stage": {"perceive": 0.1},
                "fit_error_px": 1.2,
                "decision1_confidences": {"outline_on_object": 0.95, "motion": 0.96},
                "repairs_used": 0,
                "dynamics_used": "physics",
                "fallback_reason": None,
                "endpoint_error_cm": {"foresight": 2.0},
                "endpoint_error_px": {"foresight": 10.0},
            },
            "scene": {"objects": [{"id": "obj0", "name": "ball", "shape": "circle", "est_real_size_cm": 6.7}], "motion_type": "roll", "px_per_m": 500.0},
            "trajectory": {"t_s": [s["t_s"] for s in traj], "x": [s["objects"]["obj0"]["x"] for s in traj], "y": [s["objects"]["obj0"]["y"] for s in traj]},
            "physics": {"fit_params": {"obj0": {"vx0": 0.5, "vy0": 0.0, "friction": 0.5, "elasticity": 0.65, "damping": 0.0}}, "motion_model": "roll", "decision1_answers": {"outline_on_object": "yes", "motion": "roll"}},
            "_work_dir": str(out_video.parent / "work"),
            "_scene": scene,
            "_real_pts": [{"t_s": 3.0, "x": 22.0, "y": 40.0}],
            "_traj_full": traj,
        }

    monkeypatch.setattr(omni, "continue_clip", _fake_omni)
    monkeypatch.setattr(genie, "continue_from_frame", _fake_genie)
    monkeypatch.setattr(api, "predict_clip", _fake_predict_clip)
    monkeypatch.setattr(api, "track_in_video", lambda *_a, **_kw: [{"t_s": 3.0, "x": 22.0, "y": 40.0}])
    return calls


def _wait_job(job_id: str, timeout_s: float = 10.0) -> api.Job:
    j = api.JOBS[job_id]
    deadline = time.time() + timeout_s
    while not j.closed and time.time() < deadline:
        time.sleep(0.02)
    assert j.closed, f"job {job_id} did not finish within {timeout_s}s"
    return j


def test_session_endpoints_roundtrip_and_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "SESSIONS", {})

    code, _ = asyncio.run(_asgi_request("GET", "/api/sessions/missing"))
    assert code == 404

    payload = json.dumps({"omni": False, "genie": True, "dynamics": "physics"}).encode()
    code, raw = asyncio.run(
        _asgi_request("POST", "/api/sessions/s_demo", headers=[(b"content-type", b"application/json")], body=payload)
    )
    assert code == 200
    saved = json.loads(raw)
    assert saved["session"] == "s_demo"
    assert saved["omni"] is False
    assert saved["genie"] is True
    assert saved["dynamics"] == "physics"

    code, raw = asyncio.run(_asgi_request("GET", "/api/sessions/s_demo"))
    assert code == 200
    fetched = json.loads(raw)
    assert fetched["omni"] is False
    assert fetched["genie"] is True

    bad = json.dumps({"omni": False, "genie": False, "dynamics": "invalid"}).encode()
    code, _ = asyncio.run(
        _asgi_request("POST", "/api/sessions/s_demo", headers=[(b"content-type", b"application/json")], body=bad)
    )
    assert code == 400


def test_phone_capture_with_toggles_off_makes_zero_omni_or_genie_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_pipeline(monkeypatch, tmp_path)
    clip_bytes = _make_dummy_clip(tmp_path / "phone.mp4")

    # 1. Stage sets both toggles OFF for session s_off (even though server OMNI_ENABLED & GENIE_ENABLED are True).
    code, _ = asyncio.run(
        _asgi_request(
            "POST",
            "/api/sessions/s_off",
            headers=[(b"content-type", b"application/json")],
            body=json.dumps({"omni": False, "genie": False, "dynamics": "physics"}).encode(),
        )
    )
    assert code == 200

    # Phone uploads with source="capture" — even if QR URL originally had omni=true/genie=true before user unticked!
    ctype, body = _multipart_body(
        {
            "session": "s_off",
            "source": "capture",
            "cond_seconds": "3",
            "omni": "true",
            "genie": "true",
            "dynamics": "physics",
        },
        ("video", "capture.mp4", "video/mp4", clip_bytes),
    )
    code, raw = asyncio.run(_asgi_request("POST", "/api/predict", headers=[(b"content-type", ctype)], body=body))
    assert code == 202
    job_id = json.loads(raw)["job_id"]
    j = _wait_job(job_id)

    assert j.status == "done"
    assert calls["predict_clip"] == 1
    assert calls["omni"] == 0
    assert calls["genie"] == 0
    assert j.result is not None
    assert j.result["omni"]["status"] == "skipped"
    assert j.result["genie"]["status"] == "skipped"


def test_phone_capture_qr_query_fallback_when_session_missing_on_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_pipeline(monkeypatch, tmp_path)
    clip_bytes = _make_dummy_clip(tmp_path / "phone_qr.mp4")

    # Simulate QR URL `/capture?session=s_restarted&omni=0&genie=0` when server SESSIONS has no entry:
    # Capture.tsx sends omni="false" (or "0") and genie="false" (or "0").
    qs = urlencode({"session": "s_restarted", "omni": "0", "genie": "0"})
    assert "omni=0" in qs and "genie=0" in qs
    ctype, body = _multipart_body(
        {
            "session": "s_restarted",
            "source": "capture",
            "cond_seconds": "3",
            "omni": "0",
            "genie": "0",
            "dynamics": "physics",
        },
        ("video", "capture.mp4", "video/mp4", clip_bytes),
    )
    code, raw = asyncio.run(_asgi_request("POST", "/api/predict", headers=[(b"content-type", ctype)], body=body))
    assert code == 202
    j = _wait_job(json.loads(raw)["job_id"])

    assert j.status == "done"
    assert calls["omni"] == 0
    assert calls["genie"] == 0


def test_phone_capture_runs_selected_baseline_when_stage_toggle_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_pipeline(monkeypatch, tmp_path)
    clip_bytes = _make_dummy_clip(tmp_path / "phone_omni.mp4")

    asyncio.run(
        _asgi_request(
            "POST",
            "/api/sessions/s_omni_only",
            headers=[(b"content-type", b"application/json")],
            body=json.dumps({"omni": True, "genie": False, "dynamics": "physics"}).encode(),
        )
    )
    ctype, body = _multipart_body(
        {"session": "s_omni_only", "source": "capture", "cond_seconds": "3"},
        ("video", "capture.mp4", "video/mp4", clip_bytes),
    )
    code, raw = asyncio.run(_asgi_request("POST", "/api/predict", headers=[(b"content-type", ctype)], body=body))
    assert code == 202
    j = _wait_job(json.loads(raw)["job_id"])

    assert j.status == "done"
    assert calls["omni"] == 1
    assert calls["genie"] == 0
    assert j.result is not None
    assert j.result["omni"]["status"] == "done"
    assert j.result["genie"]["status"] == "skipped"
