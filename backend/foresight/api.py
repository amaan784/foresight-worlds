"""Foresight HTTP API (CONTRACT.md §4).

Run from the repo root:
  source .venv/bin/activate && uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import csv
import functools
import importlib.util
import json
import os
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
from pydantic import BaseModel

from . import config, genie, omni
from .gemini import SpendCapHit, _status
from .pipeline import draw_overlay, endpoint_error, predict_clip, track_in_video
from .video import probe, transcode

ROOT = config.ROOT
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
UI_MAX_W = int(os.environ.get("FORESIGHT_UI_MAX_W", "1920"))  # browser-friendly size for every video we serve

app = FastAPI(title="Foresight")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=str(OUT)), name="outputs")

# ------------------------------------------------------------------------------------------------ jobs & sessions
JOBS: dict[str, "Job"] = {}
SESSIONS: dict[str, dict] = {}
SESSIONS_LOCK = threading.Lock()


class SessionPrefs(BaseModel):
    omni: bool = False
    genie: bool = False
    dynamics: str | None = None


def _parse_form_bool(val: str | None) -> bool | None:
    if val is None or not val.strip():
        return None
    s = val.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


class JobCancelled(Exception):
    """Raised inside the worker at its next event once the user stops the job."""


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
        self.cancelled = False
        self.result: dict | None = None

    def url(self, name: str) -> str:
        return f"/outputs/{self.id}/{name}"

    def emit(self, name: str, payload: dict) -> None:
        if self.cancelled and name not in ("error", "end"):
            raise JobCancelled(self.id)
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


@functools.lru_cache(maxsize=1)
def _scene_meta() -> dict:
    import yaml

    p = ROOT / "third_party" / "physics-IQ-benchmark" / "descriptions" / "data.yaml"
    if not p.exists():
        return {}
    d = yaml.safe_load(open(p))
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


def _poster(video: Path, t_s: float = 0.25) -> None:
    """Write <video>.jpg (a frame at t_s) so the stage never shows a black panel while a video loads."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
    ok, f = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(video.with_suffix(".jpg")), f, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _ui_size(w: int, h: int) -> tuple[int, int] | None:
    if w <= UI_MAX_W:
        return None
    return UI_MAX_W, int(round(h * UI_MAX_W / w / 2)) * 2


# ------------------------------------------------------------------------------------------------ worker
def _run(job: Job, src: dict, run_omni: bool, dynamics: str, run_genie: bool = False) -> None:
    lanes: dict[str, dict] = {}  # baseline video models running next to Foresight: name -> state
    threads: dict[str, threading.Thread] = {}
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
        _poster(job.dir / "conditioning.mp4", 2.9)
        if has_reality:
            _poster(job.dir / "reality.mp4")
        job.emit("ingest", {"fps": 24, "cond_frames": pc["frames"], "reality_frames": probe(job.dir / "reality.mp4")["frames"] if has_reality else 0,
                            "width": pc["width"], "height": pc["height"], "conditioning_url": job.url("conditioning.mp4"),
                            "has_reality": has_reality})
        job.emit("stage", {"stage": "ingest", "status": "done", "message": f"{pc['frames']} frames"})

        def lane(name: str, model_id: str, make_video) -> None:
            """One baseline: make its 5 s video, then (once the scene is known) its endpoint error."""
            state = lanes[name]
            job.emit(name, {"status": "start", "url": None, "latency_s": None, "error": None, "message": None})
            try:
                info = make_video(job.dir / f"{name}.mp4")
                _poster(job.dir / f"{name}.mp4")
                state.update(status="done", url=job.url(f"{name}.mp4"), latency_s=info["latency_s"], info=info, message=None)
                scene_ready.wait(timeout=600)
                if "scene" in holder and has_reality:
                    pts = track_in_video(job.dir / f"{name}.mp4", holder["scene"])
                    state["endpoint"] = endpoint_error(_omni_traj(pts, holder["scene"]), holder["real_pts"], holder["scene"], holder["main"]) if pts else None
            except JobCancelled:
                return
            except SpendCapHit as e:
                state.update(status="error", error=str(e), message=None)
            except Exception as e:  # noqa: BLE001
                code = _status(e)
                if code in (403, 404):
                    state.update(
                        status="skipped",
                        error=None,
                        message=f"{model_id} is unavailable on this API key (HTTP {code}); baseline skipped.",
                    )
                else:
                    state.update(status="error", error=f"{type(e).__name__}: {e}"[:400], message=None)
            try:
                job.emit(name, {"status": state["status"], "url": state.get("url"), "latency_s": state.get("latency_s"),
                                "error": state.get("error"), "message": state.get("message"),
                                "endpoint_error_cm": (state.get("endpoint") or {}).get("cm")})
            except JobCancelled:
                return

        prompt = src.get("omni_prompt")
        makers = {
            "omni": (
                run_omni,
                config.OMNI_ENABLED,
                config.OMNI,
                lambda out: omni.continue_clip(Path(cond), out, prompt=prompt or omni.GENERIC_PROMPT, tag=f"api:{job.id}"),
            ),
            "genie": (
                run_genie,
                config.GENIE_ENABLED,
                config.GENIE,
                lambda out: genie.continue_from_frame(
                    Path(cond), out, prompt=(prompt or "").replace(" Continue the scene.", "") or genie.GENERIC_PROMPT, tag=f"api:{job.id}"
                ),
            ),
        }
        for name, (requested, enabled, model_id, make) in makers.items():
            if requested and enabled:
                lanes[name] = {"status": "start", "message": None}
                threads[name] = threading.Thread(target=lane, args=(name, model_id, make), daemon=True)
                threads[name].start()
            else:
                msg = (
                    f"{model_id} is disabled on this server (set FORESIGHT_{name.upper()}_ENABLED=true to enable)."
                    if not enabled
                    else "Not requested for this run."
                )
                lanes[name] = {"status": "skipped", "url": None, "latency_s": None, "error": None, "message": msg}
                job.emit(name, lanes[name])

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
        _poster(job.dir / "foresight.mp4")
        _poster(job.dir / "overlay.mp4")
        job.emit("render", {"url": job.url("foresight.mp4"), "frames": meta["probe"]["frames"], "fps": meta["probe"]["fps"],
                            "duration_s": round(meta["probe"]["duration"], 3)})
        result = {k: v for k, v in meta.items() if not k.startswith("_")}
        export = _trajectory_export(scene, meta)
        (job.dir / "trajectory.json").write_text(json.dumps(export, indent=1))
        _trajectory_csv(export, job.dir / "trajectory.csv")
        result.update(job_id=job.id, status="done", source=src["kind"], title=_title(src), created=job.created,
                      models=_active_models(), understanding=_understanding(scene, meta), paths=_paths(scene, meta, main),
                      frame_url=job.url("conditioning.jpg"), urls={
            "conditioning": job.url("conditioning.mp4"), "reality": job.url("reality.mp4") if has_reality else None,
            "foresight": job.url("foresight.mp4"), "omni": None, "genie": None, "overlay": job.url("overlay.mp4"),
            "trajectory_json": job.url("trajectory.json"), "trajectory_csv": job.url("trajectory.csv")})
        for name in ("omni", "genie"):
            result["metrics"]["endpoint_error_cm"][name] = None
            result["metrics"]["endpoint_error_px"][name] = None
            result[name] = {k: lanes.get(name, {}).get(k) for k in ("status", "latency_s", "error", "message")}
        job.result = result
        job.emit("result", result)
        for name, th in threads.items():
            th.join(timeout=600)
            state = lanes[name]
            with job.cv:
                if state.get("status") == "done":
                    result["urls"][name] = state["url"]
                    ep = state.get("endpoint") or {}
                    result["metrics"]["endpoint_error_cm"][name] = ep.get("cm")
                    result["metrics"]["endpoint_error_px"][name] = ep.get("px")
                result[name] = {k: state.get(k) for k in ("status", "latency_s", "error", "message")}
                job.result = result
        job.status = "done"
        (job.dir / "result.json").write_text(json.dumps(job.result, indent=2))
        job.emit("end", {"status": "done"})
    except JobCancelled:
        pass  # the cancel endpoint already reported it
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


def _title(src: dict) -> str:
    kind = src.get("kind", "")
    return kind.split(":", 1)[1].replace("-", " ").capitalize() if kind.startswith("sample:") else "Your video"


def _paths(scene: dict, meta: dict, main: str) -> dict:
    """What the stage draws over the last seen frame: seen track, predicted path, real path (original pixels)."""
    ob = next(o for o in scene["objects"] if o["id"] == main)
    return {"orig_size": scene["orig_size"],
            "seen": [[p["t_s"], p["x"], p["y"]] for p in ob["track"]],
            "predicted": [[s["t_s"], s["objects"][main]["x"], s["objects"][main]["y"]] for s in meta["_traj_full"] if main in s["objects"]],
            "real": [[p["t_s"], p["x"], p["y"]] for p in meta["_real_pts"]]}


def _understanding(scene: dict, meta: dict) -> dict:
    """What Foresight read and fitted, in plain units, for the stage's explanation panel."""
    from physics.layers import assign_layers

    layers = assign_layers(scene)
    phys = meta.get("physics") or {}
    params = phys.get("fit_params") or {}
    objects = []
    for o in scene["objects"]:
        p = params.get(o["id"]) or {}
        speed = (p.get("vx0", 0.0) ** 2 + p.get("vy0", 0.0) ** 2) ** 0.5 if p else None
        objects.append({"id": o["id"], "name": o["name"], "shape": o["shape"], "size_cm": o.get("est_real_size_cm"),
                        "mass_g": o.get("est_mass_g"), "role": "moving",
                        "fit": {"speed_m_s": round(speed, 2) if speed is not None else None,
                                "friction": p.get("friction"), "bounce": p.get("elasticity"),
                                "damping_per_s": p.get("damping")} if p else None})
    resting = {r["static"] for r in layers.resting}
    for i, st in enumerate(scene.get("static_objects", [])):
        objects.append({"id": f"static{i}", "name": st["name"], "shape": st.get("perceived_shape", "box"),
                        "size_cm": st.get("est_real_size_cm"), "mass_g": st.get("est_mass_g"),
                        "role": "can be knocked" if i in resting else "container" if i in layers.container_statics else "static",
                        "fit": None})
    m = meta.get("metrics", {})
    return {"objects": objects,
            "surfaces": [{"name": s["name"], "kind": s.get("kind"), "material": s.get("material", "hard")} for s in scene.get("surfaces", [])],
            "depth_notes": layers.notes, "px_per_m": scene["px_per_m"], "scale_source": scene.get("px_per_m_source"),
            "motion_model": phys.get("motion_model"), "checks": {"answers": phys.get("decision1_answers") or {},
                                                                 "confidences": m.get("decision1_confidences") or {}},
            "repairs_used": m.get("repairs_used"), "dynamics_used": m.get("dynamics_used"),
            "fallback_reason": m.get("fallback_reason")}


def _trajectory_export(scene: dict, meta: dict) -> dict:
    """The predicted 5 s for every simulated body, in metres (x right, y up, origin at the image's bottom-left)
    and in image pixels, plus the fitted parameters: what a robotics team takes into its own simulator."""
    W, H = scene["orig_size"]
    ppm = float(scene["px_per_m"])
    bodies: dict[str, list] = {}
    for st in meta["_traj_full"]:
        for oid, o in st["objects"].items():
            bodies.setdefault(oid, []).append({"t_s": round(st["t_s"], 4), "x_m": round(o["x"] / ppm, 5),
                                               "y_m": round((H - o["y"]) / ppm, 5), "x_px": round(o["x"], 1),
                                               "y_px": round(o["y"], 1), "angle_rad": round(o.get("angle", 0.0) or 0.0, 4),
                                               "visible": bool(o.get("visible", True))})
    names = {o["id"]: o["name"] for o in scene["objects"]}
    return {"units": {"position": "m (x right, y up, origin at the image's bottom-left)", "time": "s from clip start"},
            "px_per_m": ppm, "image_size_px": [W, H], "gravity_m_s2": scene.get("gravity_m_s2", 9.81),
            "fps": scene["fps_out"], "dynamics": (meta.get("metrics") or {}).get("dynamics_used") or meta.get("dynamics"),
            "fit_params": (meta.get("physics") or {}).get("fit_params"),
            "bodies": {oid: {"name": names.get(oid, oid), "path": path} for oid, path in bodies.items()}}


def _trajectory_csv(export: dict, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["body", "name", "t_s", "x_m", "y_m", "x_px", "y_px", "angle_rad", "visible"])
        for oid, b in export["bodies"].items():
            for p in b["path"]:
                w.writerow([oid, b["name"], p["t_s"], p["x_m"], p["y_m"], p["x_px"], p["y_px"], p["angle_rad"], int(p["visible"])])


def _source_from_events(d: Path) -> str:
    """Runs saved before results carried their source: the first ingest event names it (sample:<id> or upload)."""
    try:
        with open(d / "events.jsonl") as f:
            for line in f:
                e = json.loads(line)
                if e["event"] == "stage" and e["data"].get("stage") == "ingest":
                    return e["data"].get("message") or ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def _recent(limit: int) -> list[dict]:
    """Finished runs, newest first, including ones from earlier server runs (read from outputs/)."""
    rows = []
    for d in sorted(OUT.glob("j_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        rp = d / "result.json"
        if not rp.exists():
            continue
        try:
            r = json.loads(rp.read_text())
        except Exception:  # noqa: BLE001
            continue
        ee = (r.get("metrics") or {}).get("endpoint_error_cm") or {}
        rows.append({"job_id": d.name, "created": r.get("created") or datetime.fromtimestamp(rp.stat().st_mtime, timezone.utc).isoformat(),
                     "title": r.get("title") or _title({"kind": r.get("source") or _source_from_events(d)}),
                     "poster": f"/outputs/{d.name}/conditioning.jpg" if (d / "conditioning.jpg").exists() else None,
                     "dynamics": (r.get("metrics") or {}).get("dynamics_used") or r.get("dynamics"),
                     "endpoint_error_cm": {k: ee.get(k) for k in ("foresight", "omni", "genie")}})
        if len(rows) >= limit:
            break
    return rows


def _active_models() -> dict[str, str]:
    return {
        "perception": config.PERCEPTION,
        "fallback": config.FALLBACK_MODEL,
        "verify": config.DECISION if config.VERIFY_MODE == "decisions" else config.VERIFY,
        "repair": "local-multistart" if config.REPAIR_MODE == "local" else config.REPAIR,
        "baseline_video": config.OMNI,
        "world_model": config.GENIE,
        "managed_agent": config.MANAGED_AGENT,
    }


# ------------------------------------------------------------------------------------------------ routes
@app.get("/api/info")
def info():
    return {
        "lan_ip": _lan_ip(),
        "physics_available": _physics_available(),
        "omni_enabled": config.OMNI_ENABLED,
        "genie_enabled": config.GENIE_ENABLED,
        "verify_mode": config.VERIFY_MODE,
        "repair_mode": config.REPAIR_MODE,
        "models": _active_models(),
    }


@app.get("/api/samples")
def samples():
    return _list_samples()


@app.post("/api/sessions/{session}")
def set_session(session: str, prefs: SessionPrefs):
    if prefs.dynamics is not None and prefs.dynamics not in ("physics", "const_velocity"):
        raise HTTPException(400, "dynamics must be physics or const_velocity")
    entry = {
        "session": session,
        "omni": bool(prefs.omni),
        "genie": bool(prefs.genie),
        "dynamics": prefs.dynamics,
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    with SESSIONS_LOCK:
        SESSIONS.pop(session, None)
        SESSIONS[session] = entry
        while len(SESSIONS) > 512:
            SESSIONS.pop(next(iter(SESSIONS)))
    return entry


@app.get("/api/sessions/{session}")
def get_session(session: str):
    with SESSIONS_LOCK:
        entry = SESSIONS.get(session)
    if entry is None:
        raise HTTPException(404, "unknown session")
    return entry


@app.post("/api/predict", status_code=202)
async def predict(video: UploadFile | None = File(None), sample_id: str | None = Form(None), session: str | None = Form(None),
                  cond_seconds: float = Form(3.0), omni: str | None = Form(None), genie: str | None = Form(None),
                  dynamics: str = Form("physics"), source: str | None = Form(None)):
    with SESSIONS_LOCK:
        sess = dict(SESSIONS[session]) if session and session in SESSIONS else None
    if source == "capture" and sess and sess.get("dynamics") in ("physics", "const_velocity"):
        dynamics = sess["dynamics"]
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
    parsed_omni = _parse_form_bool(omni)
    parsed_genie = _parse_form_bool(genie)
    if sess is not None and (source == "capture" or parsed_omni is None):
        want_omni = bool(sess["omni"])
    elif parsed_omni is not None:
        want_omni = parsed_omni
    else:
        want_omni = config.OMNI_ENABLED
    if sess is not None and (source == "capture" or parsed_genie is None):
        want_genie = bool(sess["genie"])
    elif parsed_genie is not None:
        want_genie = parsed_genie
    else:
        want_genie = config.GENIE_ENABLED
    threading.Thread(target=_run, args=(job, src, want_omni, dynamics, want_genie), daemon=True).start()
    return {"job_id": job.id, "events_url": f"/api/jobs/{job.id}/events", "result_url": f"/api/jobs/{job.id}"}


@app.get("/api/recent")
def recent(limit: int = 8):
    return _recent(max(1, min(limit, 50)))


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


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    """Stop a running job: the worker stops at its next step; a Gemini call already in flight still finishes."""
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    if j.status == "running":
        j.cancelled = True
        j.status = "cancelled"
        j.emit("error", {"stage": "cancelled", "message": "Stopped by the user."})
        j.emit("end", {"status": "cancelled"})
        j.close()
    return {"job_id": j.id, "status": j.status}


@app.get("/api/jobs/{job_id}/events")
def events(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        saved = OUT / job_id / "events.jsonl"
        if not saved.exists():
            raise HTTPException(404, "unknown job")

        def replay():  # a finished job from an earlier server run: replay its saved events
            for line in saved.read_text().splitlines():
                e = json.loads(line)
                yield f"event: {e['event']}\ndata: {json.dumps(e['data'])}\n\n"

        return StreamingResponse(replay(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

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


@app.get("/api/eval/wide")
def eval_wide():
    p = ROOT / "eval" / "results_wide.json"
    if not p.exists():
        raise HTTPException(404, "eval/results_wide.json not produced yet")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/eval/chart.png")
def eval_chart():
    p = ROOT / "eval" / "chart.png"
    if not p.exists():
        raise HTTPException(404, "eval/chart.png not produced yet")
    return FileResponse(p)
