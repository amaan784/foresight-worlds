"""Orchestrator (CONTRACT.md §3): conditioning clip -> scene -> dynamics -> 5.000 s prediction (+ metrics)."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config, track
from .scene import build_scene
from .video import probe, read_video, write_mp4

Emit = Callable[[str, dict], None]


def _noop(name: str, payload: dict) -> None:
    pass


# ----------------------------------------------------------------------------- dynamics: const velocity
def const_velocity_trajectory(scene: dict, n_fit: int = 6) -> list[dict]:
    """Physics-free baseline: least-squares velocity over the last n_fit track points, extrapolated linearly."""
    W, H = scene["orig_size"]
    t0, fps, n = scene["t_cond_end_s"], scene["fps_out"], int(round(scene["pred_seconds"] * scene["fps_out"]))
    per_obj = {}
    for ob in scene["objects"]:
        pts = ob["track"][-n_fit:]
        if not pts:
            continue
        t = np.array([p["t_s"] for p in pts])
        xy = np.array([[p["x"], p["y"]] for p in pts])
        if len(pts) >= 2:
            A = np.stack([np.ones_like(t), t], axis=1)
            cx, cy = np.linalg.lstsq(A, xy[:, 0], rcond=None)[0], np.linalg.lstsq(A, xy[:, 1], rcond=None)[0]
        else:
            cx, cy = np.array([xy[0, 0], 0.0]), np.array([xy[0, 1], 0.0])
        per_obj[ob["id"]] = (cx, cy, ob.get("radius_px") or max(ob.get("w_px") or 1, ob.get("h_px") or 1) / 2)
    traj = []
    for k in range(n):
        tk = t0 + k / fps
        objs = {}
        for oid, (cx, cy, r) in per_obj.items():
            x, y = cx[0] + cx[1] * tk, cy[0] + cy[1] * tk
            objs[oid] = {"x": float(x), "y": float(y), "angle": float((x - cx[0]) / max(r, 1.0)),
                         "visible": bool(-r < x < W + r and -r < y < H + r)}
        traj.append({"t_s": round(tk, 4), "objects": objs})
    return traj


# ----------------------------------------------------------------------------- rendering
def _paste(dst: np.ndarray, sprite: np.ndarray, cx: float, cy: float) -> None:
    h, w = sprite.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = dst.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    s = sprite[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = dst[sy0:sy1, sx0:sx1].astype(np.float32)
    dst[sy0:sy1, sx0:sx1] = (roi * (1 - a) + s[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def render_simple(scene: dict, trajectory: list[dict], out_path: Path, out_size: tuple[int, int] | None = None) -> Path:
    """Fallback renderer (used only when physics.render is unavailable): background + rotated sprites."""
    bg = cv2.imread(scene["background_png"])
    W, H = scene["orig_size"]
    scale = 1.0
    if out_size and tuple(out_size) != (W, H):
        scale = out_size[0] / W
        bg = cv2.resize(bg, tuple(out_size), interpolation=cv2.INTER_AREA)
    sprites = {}
    for ob in scene["objects"]:
        sp = cv2.imread(ob["sprite_png"], cv2.IMREAD_UNCHANGED)
        if scale != 1.0:
            sp = cv2.resize(sp, (max(2, int(sp.shape[1] * scale)), max(2, int(sp.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        sprites[ob["id"]] = sp
    frames = []
    for st in trajectory:
        f = bg.copy()
        for oid, p in st["objects"].items():
            if not p.get("visible", True) or oid not in sprites:
                continue
            sp = sprites[oid]
            M = cv2.getRotationMatrix2D((sp.shape[1] / 2, sp.shape[0] / 2), -np.degrees(p.get("angle") or 0.0), 1.0)
            rot = cv2.warpAffine(sp, M, (sp.shape[1], sp.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            _paste(f, rot, p["x"] * scale, p["y"] * scale)
        frames.append(f)
    return write_mp4(out_path, frames, scene["fps_out"])


def _renderer():
    try:
        from physics.render import render_trajectory  # physics agent's renderer (CONTRACT.md §2)
        return render_trajectory, "physics.render"
    except Exception:  # noqa: BLE001
        return render_simple, "foresight.render_simple"


def _resize_video(src: Path, dst: Path, out_size: tuple[int, int]) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale={out_size[0]}:{out_size[1]}",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)], check=True)
    return dst


# ----------------------------------------------------------------------------- endpoint error
def track_in_video(video: Path, scene: dict, proc_w: int = 960) -> list[dict]:
    """CSRT-track the main object through a 5 s video (reality or Omni), initialised from its last-conditioning box.
    Returns ok points in ORIGINAL scene px: [{t_s (absolute), x, y}]."""
    main = max(scene["objects"], key=lambda ob: len(ob["track"]))
    if not main["track"]:
        return []
    v = read_video(video, proc_w=proc_w)
    if not v["proc"]:
        return []
    W, H = scene["orig_size"]
    ph, pw = v["proc"][0].shape[:2]
    sx, sy = pw / W, ph / H
    last = main["track"][-1]
    r = main.get("radius_px") or max(main.get("w_px") or 1, main.get("h_px") or 1) / 2
    box = (int((last["x"] - r) * sx), int((last["y"] - r) * sy), max(2, int(2 * r * sx)), max(2, int(2 * r * sy)))
    # small search: the object moved a little between the last conditioning frame and reality frame 0
    run = track.csrt_run(v["proc"], 0, box, +1)
    out = []
    for rr in run:
        bx, by, bw2, bh2 = rr["box"]
        if bx < bw2 / 2 or by < bh2 / 2 or bx + bw2 > pw - bw2 / 2 or by + bh2 > ph - bh2 / 2:
            break  # the object reached the frame edge: anything after is the tracker latching onto background
        if not rr["ok"]:
            break  # first lost frame ends the track: a later "re-find" is usually a background patch
        x, y, bw, bh = rr["box"]
        out.append({"t_s": scene["t_cond_end_s"] + rr["frame"] / v["fps"], "x": (x + bw / 2) / sx, "y": (y + bh / 2) / sy})
    return out


def endpoint_error(pred: list[dict], real_pts: list[dict], scene: dict, oid: str) -> dict | None:
    if len(real_pts) < 2:
        return None
    end = real_pts[-1]
    st = min(pred, key=lambda s: abs(s["t_s"] - end["t_s"]))
    p = st["objects"].get(oid)
    if p is None:
        return None
    px = float(np.hypot(p["x"] - end["x"], p["y"] - end["y"]))
    return {"px": round(px, 1), "cm": round(100 * px / scene["px_per_m"], 1), "t_s": round(end["t_s"], 3)}


def draw_overlay(base_video: Path, scene: dict, pred: list[dict], out: Path, real_pts: list[dict] | None = None) -> Path:
    v = read_video(base_video, proc_w=960)
    W, H = scene["orig_size"]
    ph, pw = v["proc"][0].shape[:2]
    s = pw / W
    main = max(scene["objects"], key=lambda ob: len(ob["track"]))["id"]
    path = [(int(st["objects"][main]["x"] * s), int(st["objects"][main]["y"] * s)) for st in pred if main in st["objects"]]
    frames = []
    for i, f in enumerate(v["proc"][: len(pred)]):
        im = f.copy()
        for a, b in zip(path[:-1], path[1:]):
            cv2.line(im, a, b, (0, 200, 255), 2)
        if i < len(path):
            cv2.circle(im, path[i], 10, (0, 200, 255), 3)
        if real_pts:
            for rp in real_pts:
                if abs(rp["t_s"] - pred[i]["t_s"]) < 0.5 / scene["fps_out"]:
                    cv2.circle(im, (int(rp["x"] * s), int(rp["y"] * s)), 6, (0, 255, 0), -1)
        cv2.putText(im, "orange = Foresight prediction", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        frames.append(im)
    return write_mp4(out, frames, scene["fps_out"])


# ----------------------------------------------------------------------------- main entry
def predict_clip(cond_video: Path, out_video: Path, out_fps: int = config.BENCH_FPS, out_size: tuple[int, int] | None = None,
                 dynamics: str = "physics", fit_video: Path | None = None, cache_dir: Path | None = None,
                 reality_video: Path | None = None, on_event: Emit | None = None, work_dir: Path | None = None,
                 clip_id: str | None = None) -> dict:
    emit = on_event or _noop
    t_start = time.time()
    out_video = Path(out_video).resolve()
    work_dir = Path(work_dir or out_video.parent / f".work_{out_video.stem}_{dynamics}").resolve()
    scene, info = build_scene(Path(cond_video), work_dir, Path(fit_video) if fit_video else None,
                              Path(cache_dir) if cache_dir else None, out_fps, emit, clip_id)
    lat = {k: round(v, 2) for k, v in info["timings_s"].items() if isinstance(v, float)}

    phys = {}
    if dynamics == "physics":
        from physics import predict as physics_predict  # physics agent (CONTRACT.md §2)
        t0 = time.time()
        phys = physics_predict(scene, on_event=emit)
        lat["physics"] = round(time.time() - t0, 2)
        trajectory = phys["trajectory"]
        src = Path(phys["prediction_mp4"])
        if out_size and tuple(out_size) != tuple(scene["orig_size"]):
            _resize_video(src, out_video, out_size)
        elif src.resolve() != out_video:
            out_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out_video)
        renderer = "physics.predict"
    elif dynamics == "const_velocity":
        emit("stage", {"stage": "simulate", "status": "start", "message": "constant velocity (no physics)"})
        trajectory = const_velocity_trajectory(scene)
        render, renderer = _renderer()
        t0 = time.time()
        render(scene, trajectory, out_video, out_size)
        lat["render"] = round(time.time() - t0, 2)
    else:
        raise ValueError(f"unknown dynamics {dynamics!r}")

    pr = probe(out_video)

    main_id = max(scene["objects"], key=lambda ob: len(ob["track"]))["id"]
    ee = {"foresight": None}
    real_pts = []
    if reality_video:
        real_pts = track_in_video(Path(reality_video), scene)
        ee["foresight"] = endpoint_error(trajectory, real_pts, scene, main_id)

    meta = {
        "status": "done", "dynamics": dynamics, "renderer": renderer, "prediction": str(out_video), "probe": pr,
        "metrics": {
            "latency_s_total": round(time.time() - t_start, 2), "latency_s_by_stage": lat,
            "fit_error_px": phys.get("fit_error_px"), "decision1_confidences": phys.get("decision1_confidences"),
            "repairs_used": phys.get("repairs_used"),
            "endpoint_error_cm": {"foresight": ee["foresight"]["cm"] if ee["foresight"] else None},
            "endpoint_error_px": {"foresight": ee["foresight"]["px"] if ee["foresight"] else None},
        },
        "scene": {"objects": [{k: o[k] for k in ("id", "name", "shape", "est_real_size_cm")} for o in scene["objects"]],
                  "motion_type": scene["motion_type_hint"], "px_per_m": scene["px_per_m"]},
        "trajectory": {"t_s": [s["t_s"] for s in trajectory],
                       "x": [s["objects"].get(main_id, {}).get("x") for s in trajectory],
                       "y": [s["objects"].get(main_id, {}).get("y") for s in trajectory]},
        "physics": {k: phys.get(k) for k in ("fit_params", "motion_model", "decision1_answers")} if phys else None,
        "_work_dir": str(work_dir), "_scene": scene, "_real_pts": real_pts, "_traj_full": trajectory,
    }
    (work_dir / "predict_meta.json").write_text(json.dumps({k: v for k, v in meta.items() if not k.startswith("_")}, indent=2))
    return meta
