"""Phase 0d: Gemini Robotics-ER 2 point tracking vs. OpenCV CSRT on the same clip.

Requires 0b output (phase0/out/0b_<scene>/{conditioning_960x540.mp4, perception.json, track.json, report.json}).
Writes phase0/out/0d_<scene>/{er2_points.json, compare.json, overlay.mp4}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "phase0"))

from foresight import config, gemini  # noqa: E402
from vidio import read_frames, write_mp4  # noqa: E402


class TrackPoint(BaseModel):
    t_s: float = Field(description="timestamp of the frame in seconds from the start of the video")
    point: List[int] = Field(description="[y, x] normalized 0-1000: the object's center in that frame")


class ObjectTrack(BaseModel):
    label: str
    points: List[TrackPoint]


class Tracks(BaseModel):
    tracks: List[ObjectTrack]


def prompt(names: list[str], fps: float) -> str:
    objs = ", ".join(names)
    return (
        f"Track these moving objects: {objs}. The video is sampled at {fps:g} frames per second. "
        "For every frame in which an object is visible, point to the center of that object. "
        "Points are [y, x] normalized to 0-1000; t_s is the frame's timestamp in seconds. "
        "Omit frames where the object is hidden."
    )


def er2_track(video: Path, names: list[str], fps: float, tag: str) -> tuple[Tracks, dict]:
    f = gemini.upload_video(video)
    c = gemini.client_for(config.ROBOTICS_ER)
    t0 = time.time()
    resp = gemini.call(
        c.interactions.create, kind="interactions.create", model=config.ROBOTICS_ER, tag=tag,
        input=[
            {"type": "video", "uri": f.uri, "mime_type": f.mime_type, "processing": {"type": "static", "fps": fps}},
            {"type": "text", "text": prompt(names, fps)},
        ],
        response_format={"type": "text", "mime_type": "application/json", "schema": Tracks.model_json_schema()},
    )
    latency = time.time() - t0
    usage = resp.usage.model_dump(mode="json", exclude_none=True) if getattr(resp, "usage", None) else None
    return Tracks.model_validate_json(resp.output_text), {"latency_s": round(latency, 2), "usage": usage}


def compare(er2: ObjectTrack, csrt: list[dict], fps: float, w: int, h: int, scale_to_orig: float) -> dict:
    centers = {r["frame"]: np.array([r["box"][0] + r["box"][2] / 2, r["box"][1] + r["box"][3] / 2])
               for r in csrt if r["ok"]}
    radius = np.median([min(r["box"][2], r["box"][3]) / 2 for r in csrt if r["ok"]]) if centers else float("nan")
    pts = [(p.t_s, np.array([p.point[1] / 1000 * w, p.point[0] / 1000 * h])) for p in er2.points]

    def errs(shift: int) -> list[float]:
        out = []
        for t, xy in pts:
            fr = int(round(t * fps)) + shift
            if fr in centers:
                out.append(float(np.linalg.norm(xy - centers[fr])))
        return out

    base = errs(0)
    best_shift = min(range(-3, 4), key=lambda s: np.median(errs(s)) if errs(s) else 1e9)
    best = errs(best_shift)
    ts = [t for t, _ in pts]
    span = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    return {
        "er2_points": len(pts),
        "er2_time_span_s": round(span, 3),
        "er2_points_per_s": round(len(pts) / span, 2) if span else None,
        "csrt_frames_ok": len(centers),
        "csrt_points_per_s": fps,
        "matched_frames": len(base),
        "object_radius_px_proc": float(radius),
        "err_px_proc_median": float(np.median(base)) if base else None,
        "err_px_proc_mean": float(np.mean(base)) if base else None,
        "err_px_proc_max": float(np.max(base)) if base else None,
        "err_px_orig_median": float(np.median(base) * scale_to_orig) if base else None,
        "err_over_radius_median": float(np.median(base) / radius) if base else None,
        "best_time_shift_frames": best_shift,
        "err_px_proc_median_best_shift": float(np.median(best)) if best else None,
    }


def overlay(frames, er2: Tracks, csrt_tracks: dict, fps: float, path: Path) -> Path:
    h, w = frames[0].shape[:2]
    out = []
    for i, f in enumerate(frames):
        im = f.copy()
        for tr in csrt_tracks.values():
            r = next((r for r in tr if r["frame"] == i and r["ok"]), None)
            if r:
                x, y, bw, bh = r["box"]
                cv2.rectangle(im, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        for t in er2.tracks:
            for p in t.points:
                if int(round(p.t_s * fps)) == i:
                    cv2.circle(im, (int(p.point[1] / 1000 * w), int(p.point[0] / 1000 * h)), 6, (255, 0, 255), -1)
        cv2.putText(im, f"green=CSRT  magenta=ER2  f={i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.append(im)
    return write_mp4(path, out, fps)


def main(scene: str, fps: float) -> dict:
    src = ROOT / "phase0" / "out" / f"0b_{scene}"
    out_dir = ROOT / "phase0" / "out" / f"0d_{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    perception = json.loads((src / "perception.json").read_text())["scene"]
    track = json.loads((src / "track.json").read_text())
    b_report = json.loads((src / "report.json").read_text())
    names = [o["name"] for o in perception["objects"] if o["moving"]]
    frames, vfps = read_frames(src / "conditioning_960x540.mp4")
    h, w = frames[0].shape[:2]

    er2, meta = er2_track(src / "conditioning_960x540.mp4", names, fps, tag=f"0d:{scene}")
    (out_dir / "er2_points.json").write_text(json.dumps({"tracks": er2.model_dump()["tracks"], "meta": meta}, indent=1))

    results = {}
    for t in er2.tracks:
        key = next((n for n in track["tracks"] if n.lower() == t.label.lower()), None) or next(iter(track["tracks"]), None)
        if key:
            results[t.label] = compare(t, track["tracks"][key], vfps, w, h, scale_to_orig=3840 / w)
    overlay(frames, er2, track["tracks"], vfps, out_dir / "overlay.mp4")
    comp = {
        "scene": scene, "er2_model": config.ROBOTICS_ER, "er2_sampling_fps": fps,
        "er2_latency_s": meta["latency_s"], "csrt_latency_s": b_report["csrt_seconds"],
        "per_object": results,
    }
    (out_dir / "compare.json").write_text(json.dumps(comp, indent=2))
    return comp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="ball-rolls-off")
    ap.add_argument("--fps", type=float, default=config.BENCH_FPS)
    a = ap.parse_args()
    if not gemini.have_key():
        sys.exit("GEMINI_API_KEY missing: add it to .env")
    print(json.dumps(main(a.scene, a.fps), indent=2))
