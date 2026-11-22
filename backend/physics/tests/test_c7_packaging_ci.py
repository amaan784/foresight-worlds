"""Offline tests for Task C7: PEP 621 pyproject.toml, secret-free GitHub Actions CI, and Gemini wrapper offline logic."""
from __future__ import annotations

import asyncio
import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from foresight import api, config, gemini
from physics.tests.conftest import ORIGINAL_GEMINI_CALL
from physics.tests.test_c2_session_capture import _asgi_request

ROOT = config.ROOT


def test_pyproject_toml_declares_packages_dependencies_and_pytest_paths() -> None:
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.exists()
    data = tomllib.loads(pyproject_path.read_text())

    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    project = data["project"]
    assert project["name"] == "foresight-worlds"
    assert project["requires-python"] == ">=3.12"

    deps_joined = " ".join(project["dependencies"]).lower()
    for required in (
        "google-genai",
        "opencv-contrib-python-headless",
        "numpy",
        "scipy",
        "pymunk",
        "pillow",
        "pydantic",
        "python-dotenv",
        "huggingface_hub",
        "pandas",
        "matplotlib",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pyyaml",
        "tqdm",
        "joblib",
    ):
        assert required in deps_joined, f"missing dependency {required} in pyproject.toml"

    dev_joined = " ".join(project["optional-dependencies"]["dev"]).lower()
    assert "pytest" in dev_joined

    find_cfg = data["tool"]["setuptools"]["packages"]["find"]
    assert find_cfg["where"] == ["backend"]
    assert "foresight*" in find_cfg["include"]
    assert "physics*" in find_cfg["include"]

    pytest_cfg = data["tool"]["pytest"]["ini_options"]
    assert "backend" in pytest_cfg["pythonpath"]
    assert "phase0" in pytest_cfg["pythonpath"]
    assert "backend/physics/tests" in pytest_cfg["testpaths"]


def test_github_actions_ci_workflow_is_valid_and_secret_free() -> None:
    ci_path = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists()
    raw_text = ci_path.read_text()
    assert "secrets." not in raw_text, "CI workflow must never reference repository secrets"

    wf = yaml.safe_load(raw_text)
    triggers = wf["on"]
    assert "main" in triggers["push"]["branches"]
    assert "pull_request" in triggers

    assert wf["env"]["GEMINI_API_KEY"] == ""
    assert wf["env"]["GOOGLE_API_KEY"] == ""

    jobs = wf["jobs"]
    assert "backend" in jobs and "frontend" in jobs

    backend_runs = "\n".join(s.get("run", "") for s in jobs["backend"]["steps"])
    assert 'pip install -e ".[dev]"' in backend_runs
    assert "pytest" in backend_runs

    frontend_runs = "\n".join(s.get("run", "") for s in jobs["frontend"]["steps"])
    assert "npm --prefix frontend ci" in frontend_runs
    assert "npm --prefix frontend test" in frontend_runs
    assert "tsc -p frontend/tsconfig.app.json --noEmit" in frontend_runs
    assert "tsc -p frontend/tsconfig.node.json --noEmit" in frontend_runs
    assert "npm --prefix frontend run build" in frontend_runs


def test_unit_tests_enforce_empty_api_keys_and_block_live_gemini_calls() -> None:
    assert os.environ.get("GEMINI_API_KEY") == ""
    assert os.environ.get("GOOGLE_API_KEY") == ""
    with pytest.raises(AssertionError, match="never invoke live foresight.gemini.call"):
        gemini.call(lambda **_kw: None, kind="test", model="gemini-3.8-flash")


def test_gemini_call_wrapper_429_stop_503_fallback_and_log_scrubbing_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(gemini.time, "sleep", lambda _s: None)

    class _FakeHttpError(Exception):
        def __init__(self, code: int, msg: str):
            super().__init__(msg)
            self.code = code

    # 1. HTTP 429 raises SpendCapHit immediately without retrying.
    calls_429 = 0

    def _fail_429(*, model: str, **_kw):
        nonlocal calls_429
        calls_429 += 1
        raise _FakeHttpError(429, "Resource exhausted")

    with pytest.raises(gemini.SpendCapHit):
        ORIGINAL_GEMINI_CALL(_fail_429, kind="interactions.create", model="gemini-3.8-flash", tag="unit:429")
    assert calls_429 == 1

    # 2. Repeated 503 switches from primary model to configured fallback and scrubs inline media in logs/.
    models_seen: list[str] = []

    def _flaky_503(*, model: str, **_kw):
        models_seen.append(model)
        if len(models_seen) < 4:
            raise _FakeHttpError(503, "Service unavailable")
        return {"ok": True, "used_model": model}

    inline_b64 = "A" * 512
    resp = ORIGINAL_GEMINI_CALL(
        _flaky_503,
        kind="interactions.create",
        model="gemini-3.8-flash",
        tag="unit:503",
        max_retries=2,
        fallback=True,
        input=[{"type": "image", "data": inline_b64, "mime_type": "image/jpeg"}],
    )
    assert resp == {"ok": True, "used_model": "gemini-3.6-flash"}
    assert models_seen[:2] == ["gemini-3.8-flash", "gemini-3.8-flash"]
    assert models_seen[-1] == "gemini-3.6-flash"

    log_files = list(tmp_path.glob("genai_calls_*.jsonl"))
    assert len(log_files) == 1
    entries = [json.loads(ln) for ln in log_files[0].read_text().splitlines()]
    ok_entry = entries[-1]
    assert ok_entry["model"] == "gemini-3.6-flash"
    assert ok_entry["requested_model"] == "gemini-3.8-flash"
    assert inline_b64 not in json.dumps(ok_entry["request"])


def test_dockerfile_bundles_both_main_and_wide_eval_results() -> None:
    dockerfile_text = (ROOT / "Dockerfile").read_text()
    assert "eval/results.json" in dockerfile_text
    assert "eval/results_wide.json" in dockerfile_text
    assert "eval/chart.png" in dockerfile_text


def test_have_key_false_in_ci_and_upload_video_rejects_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert gemini.have_key() is False
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(gemini.time, "sleep", lambda _s: None)

    states = [
        SimpleNamespace(name="files/abc123", state="FileState.PROCESSING"),
        SimpleNamespace(name="files/abc123", state="FileState.FAILED"),
    ]

    class _StubFiles:
        def upload(self, **_kw):
            return states[0]

        def get(self, *, name: str):
            assert name == "files/abc123"
            return states[1]

    monkeypatch.setattr(gemini, "client", lambda: SimpleNamespace(files=_StubFiles()))
    monkeypatch.setattr(gemini, "call", ORIGINAL_GEMINI_CALL)

    dummy_mp4 = tmp_path / "clip.mp4"
    dummy_mp4.write_bytes(b"00")
    with pytest.raises(RuntimeError, match="failed processing"):
        gemini.upload_video(dummy_mp4)


def test_eval_recent_cancel_and_saved_sse_replay_endpoints_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "OUT", tmp_path)
    monkeypatch.setattr(api, "JOBS", {})

    # 1. Both /api/eval and /api/eval/wide serve committed JSON results with zero secrets.
    code_main, raw_main = asyncio.run(_asgi_request("GET", "/api/eval"))
    assert code_main == 200
    assert json.loads(raw_main)["n_clips"] == 15

    code_wide, raw_wide = asyncio.run(_asgi_request("GET", "/api/eval/wide"))
    assert code_wide == 200
    assert "foresight" in json.loads(raw_wide)["systems"]

    # 2. Saved finished run on disk appears in /api/recent and replays SSE from events.jsonl.
    saved_dir = tmp_path / "j_saved01"
    saved_dir.mkdir(parents=True)
    (saved_dir / "result.json").write_text(
        json.dumps(
            {
                "job_id": "j_saved01",
                "status": "done",
                "title": "Ball rolls off",
                "created": "2026-11-22T10:00:00+00:00",
                "dynamics": "physics",
                "metrics": {"endpoint_error_cm": {"foresight": 4.9, "omni": None, "genie": None}},
            }
        )
    )
    (saved_dir / "events.jsonl").write_text(
        json.dumps({"event": "stage", "data": {"stage": "ingest", "status": "done", "t": 0.1}})
        + "\n"
        + json.dumps({"event": "end", "data": {"status": "done", "t": 1.2}})
        + "\n"
    )

    code_rec, raw_rec = asyncio.run(_asgi_request("GET", "/api/recent?limit=4"))
    assert code_rec == 200
    rec = json.loads(raw_rec)
    assert len(rec) == 1 and rec[0]["job_id"] == "j_saved01"

    sse_resp = api.events("j_saved01")
    assert sse_resp.media_type == "text/event-stream"

    async def _collect_sse() -> str:
        parts: list[str] = []
        async for chunk in sse_resp.body_iterator:
            parts.append(chunk.decode() if isinstance(chunk, bytes) else str(chunk))
        return "".join(parts)

    sse_text = asyncio.run(_collect_sse())
    assert "event: stage" in sse_text and "event: end" in sse_text

    # 3. Cancelling a running job transitions status to cancelled and emits end.
    running_job = api.Job("sess_cancel")
    api.JOBS[running_job.id] = running_job
    code_cancel, raw_cancel = asyncio.run(_asgi_request("POST", f"/api/jobs/{running_job.id}/cancel"))
    assert code_cancel == 200
    assert json.loads(raw_cancel)["status"] == "cancelled"
    assert running_job.closed is True
