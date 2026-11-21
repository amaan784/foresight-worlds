"""Foresight HTTP API (CONTRACT.md §4).

Run from the repo root:
  source .venv/bin/activate && uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import csv
import importlib.util
import json
import socket
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, omni
from .gemini import SpendCapHit
from .pipeline import draw_overlay, endpoint_error, predict_clip, track_in_video
from .video import probe, transcode

ROOT = config.ROOT
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
UI_MAX_W = 1920  # browser-friendly size for every video we serve

app = FastAPI(title="Foresight")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUT)), name="outputs")

# ------------------------------------------------------------------------------------------------ jobs
JOBS: dict[str, "Job"] = {}


class Job:
    def __init__(self, session: str | None):
        self.id = "j_" + uuid.uuid4().hex[:8]
        self.session = session
        self.created = datetime.now(timezone.utc).isoformat()
        self.t0 = time.time()
        self.dir = OUT / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events: list[tuple[str, dict]] = []
        self.cv = threading.Condition()
        self.status = "running"
        self.closed = False
        self.result: dict | None = None

    def url(self, name: str) -> str:
        return f"/outputs/{self.id}/{name}"

    def emit(self, name: str, payload: dict) -> None:
        payload = {**payload, "t": round(time.time() - self.t0, 2)}
        with self.cv:
            self.events.append((name, payload))
            with open(self.dir / "events.jsonl", "a") as f:
                f.write(json.dumps({"event": name, "data": payload}) + "\n")
            self.cv.notify_all()

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()


# ------------------------------------------------------------------------------------------------ samples
DATA_DIRS = [ROOT / "data" / "physics-IQ-benchmark-verified", ROOT / "data" / "physics-IQ-benchmark"]


def _scene_meta() -> dict:
    import yaml

    d = yaml.safe_load(open(ROOT / "third_party" / "physics-IQ-benchmark" / "descriptions" / "data.yaml"))
    return {s["scene"]: s for s in d["scenes"]}


def _sample_files(scene: str) -> dict | None:
    meta = _scene_meta().get(scene)
    if not meta:
        return None
    t1 = f"{int(next(t for t in meta['takes'] if t['take'] == 1)['center']):04d}"
    for base in DATA_DIRS:
        c24 = base / "split-videos" / "conditioning" / "24FPS" / f"{t1}_conditioning-videos_24FPS_perspective-center_take-1_trimmed-{scene}.mp4"
        real = base / "split-videos" / "testing" / "24FPS" / f"{t1}_testing-videos_24FPS_perspective-center_take-1_trimmed-{scene}.mp4"
        c30 = base / "split-videos" / "conditioning" / "30FPS" / f"{t1}_conditioning-videos_30FPS_perspective-center_take-1_trimmed-{scene}.mp4"
        if c24.exists() and real.exists():
            return {"cond": c24, "fit": c30 if c30.exists() else None, "reality": real, "id": t1,
                    "prompt_csv_key": f"{t1}_perspective-center_take-1_trimmed-{scene}.mp4"}
    return None


def _benchmark_prompt(key: str) -> str | None:
    p = ROOT / "third_party" / "physics-IQ-benchmark" / "descriptions" / "best_practice" / "descriptions_base.csv"
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["scenario"] == key:
                return row["description"] + " Continue the scene."
    return None


def _list_samples() -> list[dict]:
    out = []
    thumbs = OUT / "samples"
    thumbs.mkdir(exist_ok=True)
    for scene in sorted(_scene_meta()):
        fs = _sample_files(scene)
        if not fs:
            continue
        th = thumbs / f"{scene}.jpg"
        if not th.exists():
            cap = cv2.VideoCapture(str(fs["cond"]))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
            ok, fr = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(th), cv2.resize(fr, (480, 270), interpolation=cv2.INTER_AREA))
        out.append({"id": scene, "title": scene.replace("-", " ").capitalize(), "thumb_url": f"/outputs/samples/{scene}.jpg"})
    return out


# ------------------------------------------------------------------------------------------------ helpers
def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _physics_available() -> bool:
    return importlib.util.find_spec("physics") is not None


def _ui_size(w: int, h: int) -> tuple[int, int] | None:
    if w <= UI_MAX_W:
        return None
    return UI_MAX_W, int(round(h * UI_MAX_W / w / 2)) * 2


# ------------------------------------------------------------------------------------------------ worker
def _run(job: Job, src: dict, run_omni: bool, dynamics: str) -> None:
    omni_thread = None
    omni_state: dict = {"status": "skipped"}
    scene_ready = threading.Event()
    holder: dict = {}
    try:
        job.emit("stage", {"stage": "ingest", "status": "start", "message": src["kind"]})
        cond, fit, reality = src["cond"], src.get("fit"), src.get("reality")
        pc = probe(cond)
        transcode(cond, job.dir / "conditioning.mp4", 24, max_w=UI_MAX_W)
        has_reality = bool(reality and Path(reality).exists())
        if has_reality:
            transcode(reality, job.dir / "reality.mp4", 24, dur_s=5.0, max_w=UI_MAX_W)
        job.emit("ingest", {"fps": 24, "cond_frames": pc["frames"], "reality_frames": probe(job.dir / "reality.mp4")["frames"] if has_reality else 0,
                            "width": pc["width"], "height": pc["height"], "conditioning_url": job.url("conditioning.mp4"),
                            "has_reality": has_reality})
        job.emit("stage", {"stage": "ingest", "status": "done", "message": f"{pc['frames']} frames"})

        if run_omni:
            def omni_lane():
                job.emit("omni", {"status": "start", "url": None, "latency_s": None, "error": None})
                try:
                    info = omni.continue_clip(Path(cond), job.dir / "omni.mp4", prompt=src.get("omni_prompt") or omni.GENERIC_PROMPT,
                                              tag=f"api:{job.id}")
                    omni_state.update(status="done", url=job.url("omni.mp4"), latency_s=info["latency_s"], info=info)
                    scene_ready.wait(timeout=600)
                    if "scene" in holder and has_reality:
                        pts = track_in_video(job.dir / "omni.mp4", holder["scene"])
                        omni_state["endpoint"] = endpoint_error(_omni_traj(pts, holder["scene"]), holder["real_pts"], holder["scene"], holder["main"]) if pts else None
                except SpendCapHit as e:
                    omni_state.update(status="error", error=str(e))
                except Exception as e:  # noqa: BLE001
                    omni_state.update(status="error", error=f"{type(e).__name__}: {e}"[:400])
                job.emit("omni", {"status": omni_state["status"], "url": omni_state.get("url"), "latency_s": omni_state.get("latency_s"),
                                  "error": omni_state.get("error"),
                                  "endpoint_error_cm": (omni_state.get("endpoint") or {}).get("cm")})
            omni_thread = threading.Thread(target=omni_lane, daemon=True)
            omni_thread.start()
        else:
            job.emit("omni", {"status": "skipped", "url": None, "latency_s": None, "error": None})

        out_size = _ui_size(pc["width"], pc["height"])
        meta = predict_clip(Path(cond), job.dir / "foresight.mp4", out_size=out_size, dynamics=dynamics,
                            fit_video=Path(fit) if fit else None, reality_video=Path(reality) if has_reality else None,
                            on_event=job.emit, work_dir=job.dir / "work", clip_id=src.get("clip_id"))
        scene = meta["_scene"]
        main = max(scene["objects"], key=lambda ob: len(ob["track"]))["id"]
        holder.update(scene=scene, real_pts=meta["_real_pts"], main=main)
        scene_ready.set()
        draw_overlay(job.dir / ("reality.mp4" if has_reality else "foresight.mp4"), scene, meta["_traj_full"],
                     job.dir / "overlay.mp4", meta["_real_pts"])
        job.emit("render", {"url": job.url("foresight.mp4"), "frames": meta["probe"]["frames"], "fps": meta["probe"]["fps"],
                            "duration_s": round(meta["probe"]["duration"], 3)})
        result = {k: v for k, v in meta.items() if not k.startswith("_")}
        result.update(job_id=job.id, status="done", urls={
            "conditioning": job.url("conditioning.mp4"), "reality": job.url("reality.mp4") if has_reality else None,
            "foresight": job.url("foresight.mp4"), "omni": None, "overlay": job.url("overlay.mp4")})
        result["metrics"]["endpoint_error_cm"]["omni"] = None
        result["metrics"]["endpoint_error_px"]["omni"] = None
        job.result = result
        job.emit("result", result)
        if omni_thread:
            omni_thread.join(timeout=600)
            if omni_state.get("status") == "done":
                result["urls"]["omni"] = omni_state["url"]
                ep = omni_state.get("endpoint") or {}
                result["metrics"]["endpoint_error_cm"]["omni"] = ep.get("cm")
                result["metrics"]["endpoint_error_px"]["omni"] = ep.get("px")
            result["omni"] = {k: omni_state.get(k) for k in ("status", "latency_s", "error")}
            job.result = result
        job.status = "done"
        (job.dir / "result.json").write_text(json.dumps(job.result, indent=2))
        job.emit("end", {"status": "done"})
    except SpendCapHit as e:
        job.status = "error"
        job.emit("error", {"stage": "gemini", "message": str(e)})
    except Exception as e:  # noqa: BLE001
        job.status = "error"
        traceback.print_exc()
        job.emit("error", {"stage": "pipeline", "message": f"{type(e).__name__}: {e}"[:500]})
    finally:
        scene_ready.set()
        job.close()


def _omni_traj(pts: list[dict], scene: dict) -> list[dict]:
    main = max(scene["objects"], key=lambda ob: len(ob["track"]))["id"]
    return [{"t_s": p["t_s"], "objects": {main: {"x": p["x"], "y": p["y"]}}} for p in pts]


# ------------------------------------------------------------------------------------------------ routes
@app.get("/api/info")
def info():
    return {"lan_ip": _lan_ip(), "physics_available": _physics_available(), "omni_enabled": True,
            "models": {"perception": config.HYDROGEN, "repair": config.CARBON, "verify": config.DECISION,
                       "baseline_video": config.OMNI, "managed_agent": config.MANAGED_AGENT}}


@app.get("/api/samples")
def samples():
    return _list_samples()


@app.post("/api/predict", status_code=202)
async def predict(video: UploadFile | None = File(None), sample_id: str | None = Form(None), session: str | None = Form(None),
                  cond_seconds: float = Form(3.0), omni: str = Form("true"), dynamics: str = Form("physics")):
    if dynamics not in ("physics", "const_velocity"):
        raise HTTPException(400, "dynamics must be physics or const_velocity")
    if dynamics == "physics" and not _physics_available():
        raise HTTPException(503, "physics module (backend/physics) is not available yet")
    job = Job(session)
    if sample_id:
        fs = _sample_files(sample_id)
        if not fs:
            raise HTTPException(404, f"unknown sample {sample_id}")
        src = {"kind": f"sample:{sample_id}", "cond": fs["cond"], "fit": fs["fit"], "reality": fs["reality"],
               "omni_prompt": _benchmark_prompt(fs["prompt_csv_key"]), "clip_id": f"{fs['id']}_{sample_id}"}
    elif video is not None:
        suffix = Path(video.filename or "capture.mp4").suffix or ".mp4"
        raw = job.dir / f"upload{suffix}"
        raw.write_bytes(await video.read())
        cond = transcode(raw, job.dir / "cond_src.mp4", 24, 0.0, cond_seconds, max_w=UI_MAX_W)
        fit = transcode(raw, job.dir / "fit_src.mp4", 30, 0.0, cond_seconds, max_w=UI_MAX_W)
        reality = None
        try:
            dur = probe(raw)["duration"]
        except Exception:  # noqa: BLE001  (some webm files lack a container duration)
            dur = 0.0
        if dur == 0.0 or dur >= cond_seconds + 1.0:
            reality = transcode(raw, job.dir / "reality_src.mp4", 24, cond_seconds, 5.0, max_w=UI_MAX_W)
            if probe(reality)["frames"] < 24:
                reality = None
        src = {"kind": "upload", "cond": cond, "fit": fit, "reality": reality, "clip_id": job.id}
    else:
        raise HTTPException(400, "send a video file or a sample_id")
    JOBS[job.id] = job
    threading.Thread(target=_run, args=(job, src, omni.lower() == "true", dynamics), daemon=True).start()
    return {"job_id": job.id, "events_url": f"/api/jobs/{job.id}/events", "result_url": f"/api/jobs/{job.id}"}


@app.get("/api/jobs")
def jobs(session: str | None = None):
    js = [j for j in JOBS.values() if session is None or j.session == session]
    return [{"job_id": j.id, "created": j.created, "status": j.status} for j in sorted(js, key=lambda j: j.created, reverse=True)]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        p = OUT / job_id / "result.json"
        if p.exists():
            return JSONResponse(json.loads(p.read_text()))
        raise HTTPException(404, "unknown job")
    if j.result is not None:
        return j.result
    return {"job_id": j.id, "status": j.status, "events_so_far": [{"event": n, "data": d} for n, d in j.events]}


@app.get("/api/jobs/{job_id}/events")
def events(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")

    def gen():
        i = 0
        while True:
            with j.cv:
                if i >= len(j.events) and not j.closed:
                    j.cv.wait(timeout=15)
                batch = j.events[i:]
                i += len(batch)
                done = j.closed and i >= len(j.events)
            if not batch and not done:
                yield ": keep-alive\n\n"
            for name, payload in batch:
                yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"
            if done:
                break

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/eval")
def eval_results():
    p = ROOT / "eval" / "results.json"
    if not p.exists():
        raise HTTPException(404, "eval/results.json not produced yet")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/eval/chart.png")
def eval_chart():
    p = ROOT / "eval" / "chart.png"
    if not p.exists():
        raise HTTPException(404, "eval/chart.png not produced yet")
    return FileResponse(p)
