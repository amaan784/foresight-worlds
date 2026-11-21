"""Phase 0b: Hydrogen scene parsing (agentic video, structured output) + OpenCV CSRT tracking.

Usage: python phase0/p0b_perceive_track.py ball-rolls-off [--source gcs|hf]
Writes phase0/out/0b_<scene>/{perception.json, track.json, overlay.mp4, report.json}
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "phase0"))

from foresight import config, gemini  # noqa: E402
from physiq_data import benchmark_dir  # noqa: E402
from static_pred import conditioning_path  # noqa: E402
from vidio import read_frames, write_mp4  # noqa: E402

PROC_W, PROC_H = 960, 540  # processing resolution (original kept for output)


class SceneObject(BaseModel):
    name: str = Field(description="short noun phrase, e.g. 'grey tennis ball'")
    shape: Literal["circle", "box"]
    est_real_size_cm: float = Field(description="real-world diameter (circle) or width (box) in cm")
    moving: bool = Field(description="true if this object moves at any time in the clip")
    visible_frame0: bool
    bbox_frame0: List[int] = Field(description="[ymin, xmin, ymax, xmax] normalized 0-1000 in the FIRST frame; [0,0,0,0] if not visible")
    bbox_last_frame: List[int] = Field(description="[ymin, xmin, ymax, xmax] normalized 0-1000 in the LAST frame; [0,0,0,0] if not visible")


class SupportSurface(BaseModel):
    name: str
    kind: Literal["table", "floor", "ramp", "wall", "container", "other"]
    line: List[int] = Field(description="top edge the object rests/rolls on, as [y1, x1, y2, x2] normalized 0-1000 in the last frame")


class ScenePerception(BaseModel):
    objects: List[SceneObject]
    support_surfaces: List[SupportSurface]
    motion_type: Literal["roll", "slide", "fall", "bounce", "other"]
    motion_start_s: float = Field(description="time in seconds when the main object starts moving (or appears)")


PROMPT = """You are the perception stage of a physics simulator for robotics teams.
The video is a 3-second side-view clip from a static camera. Image A is its first frame, image B its last frame.
List the rigid objects that move (or are about to move) and any large static objects they could hit.
Approximate each object as a circle or a box and estimate its real size in cm from everyday knowledge
(e.g. tennis ball ~6.7 cm). Give the support surfaces as line segments along their top edges.
Classify the main object's motion. Bounding boxes and lines use [y, x] order normalized to 0-1000."""


def jpg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def norm_to_px(b: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    """[ymin,xmin,ymax,xmax] 0-1000 -> (x, y, w, h) pixels."""
    y0, x0, y1, x1 = b
    return (int(x0 / 1000 * w), int(y0 / 1000 * h), max(2, int((x1 - x0) / 1000 * w)), max(2, int((y1 - y0) / 1000 * h)))


def perceive(video_path: Path, frames: list[np.ndarray], fps: float, tag: str) -> tuple[ScenePerception, dict]:
    f = gemini.upload_video(video_path)
    c = gemini.client()
    t0 = time.time()
    resp = gemini.call(
        c.interactions.create,
        kind="interactions.create", model=config.HYDROGEN, tag=tag, fallback=True,
        input=[
            {"type": "video", "uri": f.uri, "mime_type": f.mime_type, "processing": "agentic"},
            {"type": "text", "text": "Image A (first frame):"},
            {"type": "image", "data": jpg_b64(frames[0]), "mime_type": "image/jpeg"},
            {"type": "text", "text": "Image B (last frame):"},
            {"type": "image", "data": jpg_b64(frames[-1]), "mime_type": "image/jpeg"},
            {"type": "text", "text": PROMPT},
        ],
        generation_config={"thinking_level": "low"},
        response_format={"type": "text", "mime_type": "application/json", "schema": ScenePerception.model_json_schema()},
    )
    latency = time.time() - t0
    scene = ScenePerception.model_validate_json(resp.output_text)
    usage = resp.usage.model_dump(mode="json", exclude_none=True) if getattr(resp, "usage", None) else None
    return scene, {"latency_s": round(latency, 2), "model": getattr(resp, "model", config.HYDROGEN), "usage": usage, "file_uri": f.uri}


def csrt_track(frames: list[np.ndarray], init_idx: int, init_box: tuple, direction: int) -> list[dict]:
    """Track from frames[init_idx] in `direction` (+1 forward, -1 backward). Stops after 3 consecutive losses."""
    tr = cv2.TrackerCSRT_create()
    tr.init(frames[init_idx], init_box)
    out = [{"frame": init_idx, "ok": True, "box": list(map(int, init_box))}]
    misses = 0
    idx = init_idx + direction
    while 0 <= idx < len(frames):
        ok, box = tr.update(frames[idx])
        out.append({"frame": idx, "ok": bool(ok), "box": [int(v) for v in box]})
        misses = 0 if ok else misses + 1
        if misses >= 3:
            break
        idx += direction
    return sorted(out, key=lambda r: r["frame"])


def motion_energy(frames: list[np.ndarray]) -> list[float]:
    g = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames]
    return [0.0] + [float((cv2.absdiff(g[i], g[i - 1]) > 15).mean()) for i in range(1, len(g))]


def track_quality(track: list[dict], frames: list[np.ndarray]) -> dict:
    ok = [r for r in track if r["ok"]]
    centers = np.array([[r["box"][0] + r["box"][2] / 2, r["box"][1] + r["box"][3] / 2] for r in ok]) if ok else np.zeros((0, 2))
    areas = np.array([r["box"][2] * r["box"][3] for r in ok]) if ok else np.zeros(0)
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros(0)
    # appearance check: mean abs color diff of the tracked patch vs. the init patch
    r0 = next(r for r in track if r["ok"])
    x, y, w, h = r0["box"]
    ref = cv2.resize(frames[r0["frame"]][y:y + h, x:x + w], (32, 32)).astype(np.float32)
    app = []
    for r in ok:
        x, y, w, h = r["box"]
        patch = frames[r["frame"]][max(0, y):y + h, max(0, x):x + w]
        if patch.size:
            app.append(float(np.abs(cv2.resize(patch, (32, 32)).astype(np.float32) - ref).mean()))
    return {
        "frames_total": len(frames),
        "frames_tracked_ok": len(ok),
        "first_ok_frame": ok[0]["frame"] if ok else None,
        "last_ok_frame": ok[-1]["frame"] if ok else None,
        "path_length_px": float(jumps.sum()),
        "max_jump_px": float(jumps.max()) if len(jumps) else 0.0,
        "area_cv": float(areas.std() / areas.mean()) if len(areas) else None,
        "appearance_l1_mean": float(np.mean(app)) if app else None,
        "appearance_l1_max": float(np.max(app)) if app else None,
    }


def overlay(frames: list[np.ndarray], scene: ScenePerception, tracks: dict, fps: float, path: Path) -> Path:
    h, w = frames[0].shape[:2]
    out = []
    for i, f in enumerate(frames):
        im = f.copy()
        for s in scene.support_surfaces:
            y1, x1, y2, x2 = s.line
            cv2.line(im, (int(x1 / 1000 * w), int(y1 / 1000 * h)), (int(x2 / 1000 * w), int(y2 / 1000 * h)), (255, 200, 0), 2)
        for name, tr in tracks.items():
            r = next((r for r in tr if r["frame"] == i), None)
            if r:
                x, y, bw, bh = r["box"]
                col = (0, 255, 0) if r["ok"] else (0, 0, 255)
                cv2.rectangle(im, (x, y), (x + bw, y + bh), col, 2)
                cv2.putText(im, name, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(im, f"t={i / fps:.2f}s  f={i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.append(im)
    return write_mp4(path, out, fps)


def main(scene_name: str, source: str, view: str = "center", fps: int = config.BENCH_FPS) -> dict:
    out_dir = ROOT / "phase0" / "out" / f"0b_{scene_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    full, src_fps = read_frames(conditioning_path(benchmark_dir(source), scene_name, view, fps))
    frames = [cv2.resize(f, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA) for f in full]
    small = write_mp4(out_dir / "conditioning_960x540.mp4", frames, src_fps)

    scene, meta = perceive(small, frames, src_fps, tag=f"0b:{scene_name}")
    (out_dir / "perception.json").write_text(json.dumps({"scene": scene.model_dump(), "meta": meta}, indent=2))

    t0 = time.time()
    tracks = {}
    for o in scene.objects:
        if not o.moving:
            continue
        if o.bbox_last_frame != [0, 0, 0, 0]:
            tracks[o.name] = csrt_track(frames, len(frames) - 1, norm_to_px(o.bbox_last_frame, PROC_W, PROC_H), -1)
        elif o.visible_frame0:
            tracks[o.name] = csrt_track(frames, 0, norm_to_px(o.bbox_frame0, PROC_W, PROC_H), +1)
    track_s = time.time() - t0
    quality = {k: track_quality(v, frames) for k, v in tracks.items()}
    energy = motion_energy(frames)
    moving_frames = [i for i, e in enumerate(energy) if e > 0.0005]
    (out_dir / "track.json").write_text(json.dumps({"proc_size": [PROC_W, PROC_H], "fps": src_fps, "tracks": tracks}, indent=1))
    overlay(frames, scene, tracks, src_fps, out_dir / "overlay.mp4")
    report = {
        "scene": scene_name, "view": view,
        "perception_latency_s": meta["latency_s"], "perception_model": meta["model"], "usage": meta["usage"],
        "objects": [o.model_dump() for o in scene.objects], "motion_type": scene.motion_type,
        "motion_start_s_model": scene.motion_start_s,
        "frames_with_pixel_motion": [moving_frames[0], moving_frames[-1]] if moving_frames else None,
        "csrt_seconds": round(track_s, 3), "track_quality": quality,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="ball-rolls-off")
    ap.add_argument("--source", default="gcs", choices=["hf", "gcs"])
    a = ap.parse_args()
    if not gemini.have_key():
        sys.exit("GEMINI_API_KEY missing: add it to .env")
    print(json.dumps(main(a.scene, a.source), indent=2))
