"""Offline test for Task C3: a 4.0 s video upload produces a 1.0 s reality clip alongside a full 5.0 s prediction."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from foresight import api
from foresight.video import probe, write_mp4
from physics.tests.test_c2_session_capture import _asgi_request, _multipart_body, _stub_pipeline, _wait_job


def test_4s_upload_produces_1s_reality_and_5s_foresight_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pipeline(monkeypatch, tmp_path)
    upload_mp4 = tmp_path / "short_4s.mp4"
    write_mp4(upload_mp4, [np.zeros((64, 96, 3), dtype=np.uint8) for _ in range(96)], 24)  # 4.000 s @ 24 FPS

    ctype, body = _multipart_body(
        {"cond_seconds": "3", "omni": "false", "genie": "false", "dynamics": "physics"},
        ("video", "short_4s.mp4", "video/mp4", upload_mp4.read_bytes()),
    )
    code, raw = asyncio.run(_asgi_request("POST", "/api/predict", headers=[(b"content-type", ctype)], body=body))
    assert code == 202
    job_id = json.loads(raw)["job_id"]
    j = _wait_job(job_id)

    assert j.status == "done"
    assert j.result is not None
    cond_probe = probe(j.dir / "conditioning.mp4")
    real_probe = probe(j.dir / "reality.mp4")
    fore_probe = probe(j.dir / "foresight.mp4")

    assert cond_probe["frames"] == 72 and abs(cond_probe["duration"] - 3.0) < 0.05
    assert real_probe["frames"] == 24 and abs(real_probe["duration"] - 1.0) < 0.05
    assert fore_probe["frames"] == 120 and abs(fore_probe["duration"] - 5.0) < 0.05
