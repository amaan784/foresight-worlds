"""Build the scene JSON (CONTRACT.md §1) from a conditioning clip: perceive -> track -> scale -> assets.

CLI (writes <work_dir>/scene.json + PNGs):
  PYTHONPATH=backend python -m foresight.scene <cond_video> <work_dir> [--fit-video V30] [--cache-dir D]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config, perceive, scale, track
from .video import read_video, write_mp4

PROC_W = 960
Emit = Callable[[str, dict], None]


def _noop(name: str, payload: dict) -> None:
    pass


def _norm_box_to_orig(b: list[int], W: int, H: int) -> list[float]:
    """[ymin,xmin,ymax,xmax] 0-1000 -> [x0, y0, x1, y1] orig px."""
    y0, x0, y1, x1 = b
    return [x0 / 1000 * W, y0 / 1000 * H, x1 / 1000 * W, y1 / 1000 * H]


def _inpaint(img: np.ndarray, boxes: list[list[float]], grow: float = 0.35) -> np.ndarray:
    """Inpaint the given orig-px boxes (grown) on a cropped region for speed."""
    out = img.copy()
    H, W = img.shape[:2]
    for x0, y0, x1, y1 in boxes:
        gw, gh = (x1 - x0) * grow, (y1 - y0) * grow
        bx0, by0 = int(max(0, x0 - gw)), int(max(0, y0 - gh))
        bx1, by1 = int(min(W, x1 + gw)), int(min(H, y1 + gh))
        pad = max(16, int(0.5 * max(bx1 - bx0, by1 - by0)))
        cx0, cy0, cx1, cy1 = max(0, bx0 - pad), max(0, by0 - pad), min(W, bx1 + pad), min(H, by1 + pad)
        crop = out[cy0:cy1, cx0:cx1]
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[by0 - cy0:by1 - cy0, bx0 - cx0:bx1 - cx0] = 255
        out[cy0:cy1, cx0:cx1] = cv2.inpaint(crop, mask, max(5, pad // 6), cv2.INPAINT_TELEA)
    return out


def _overlap(a: list[float], b: list[float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _fill_background(last: np.ndarray, first: np.ndarray, boxes_last: list[list[float]],
                     boxes_first: list[list[float]], grow: float = 0.7) -> tuple[np.ndarray, list[str]]:
    """Static camera: copy each object's (grown) last-frame region from the first frame when the object was
    elsewhere at t=0 (removes object + shadow cleanly, feathered edge); otherwise fall back to cv2.inpaint."""
    out = last.copy()
    H, W = last.shape[:2]
    methods = []
    for x0, y0, x1, y1 in boxes_last:
        gw, gh = (x1 - x0) * grow, (y1 - y0) * grow
        R = [max(0, x0 - gw), max(0, y0 - gh), min(W, x1 + gw), min(H, y1 + gh)]
        if any(_overlap(R, b) for b in boxes_first):
            out = _inpaint(out, [[x0, y0, x1, y1]])
            methods.append("inpaint")
            continue
        rx0, ry0, rx1, ry1 = [int(v) for v in R]
        mask = np.zeros((H, W), np.float32)
        mask[ry0:ry1, rx0:rx1] = 1.0
        feather = max(3, int(0.15 * min(rx1 - rx0, ry1 - ry0))) | 1
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)[..., None]
        out = (out * (1 - mask) + first * mask).astype(np.uint8)
        methods.append("temporal")
    return out, methods


def _sprite(img: np.ndarray, cx: float, cy: float, shape: str, w: float, h: float) -> np.ndarray:
    """RGBA crop centred on (cx, cy); circle -> feathered disc alpha, box -> rectangle alpha."""
    H, W = img.shape[:2]
    half_w, half_h = (w / 2, h / 2) if shape == "box" else (max(w, h) / 2,) * 2
    x0, y0 = int(round(cx - half_w)), int(round(cy - half_h))
    x1, y1 = int(round(cx + half_w)), int(round(cy + half_h))
    crop = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    alpha = np.zeros(crop.shape[:2], np.uint8)
    if shape == "circle":
        r = min(crop.shape[:2]) / 2
        cv2.circle(alpha, (crop.shape[1] // 2, crop.shape[0] // 2), max(1, int(r * 0.96)), 255, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, r * 0.04))
    else:
        alpha[:] = 255
    return np.dstack([crop, alpha])


def build_scene(cond_video: Path, work_dir: Path, fit_video: Path | None = None, cache_dir: Path | None = None,
                out_fps: int = config.BENCH_FPS, on_event: Emit | None = None, clip_id: str | None = None) -> tuple[dict, dict]:
    emit = on_event or _noop
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # ---- ingest
    t0 = time.time()
    cond = read_video(cond_video, proc_w=PROC_W, keep_full={0, -1})
    W, H = cond["orig_size"]
    n_cond, cond_fps = cond["n"], cond["fps"]
    t_cond_end = n_cond / cond_fps
    first_full, last_full = cond["full"][0], cond["full"][n_cond - 1]
    proc_video = write_mp4(work_dir / "cond_proc.mp4", cond["proc"], cond_fps)
    timings["ingest"] = time.time() - t0

    # ---- perceive
    emit("stage", {"stage": "perceive", "status": "start", "message": f"{config.HYDROGEN} reading the clip"})
    t0 = time.time()
    scene_p, pmeta = perceive.perceive(proc_video, first_full, last_full, tag=f"perceive:{clip_id or cond_video.name}",
                                       cache_file=(cache_dir / "perception.json") if cache_dir else None)
    timings["perceive"] = time.time() - t0
    emit("perception", {"model": pmeta["model"], "latency_s": pmeta["latency_s"], "cached": pmeta.get("cached", False),
                        "objects": [{k: o.model_dump()[k] for k in ("name", "shape", "est_real_size_cm", "moving", "bbox_last_frame")}
                                    for o in scene_p.objects],
                        "support_surfaces": [s.model_dump() for s in scene_p.support_surfaces],
                        "motion_type": scene_p.motion_type, "motion_start_s": scene_p.motion_start_s})
    emit("stage", {"stage": "perceive", "status": "done", "message": f"{len(scene_p.objects)} objects"})

    # ---- track (on the fit video if given: real timestamps; Physics-IQ 24 FPS files are frame-dropped)
    emit("stage", {"stage": "track", "status": "start", "message": "CSRT"})
    t0 = time.time()
    fit = read_video(fit_video, proc_w=PROC_W) if fit_video else cond
    fit_fps, sx, sy = fit["fps"], W / fit["proc"][0].shape[1], H / fit["proc"][0].shape[0]
    moving = [o for o in scene_p.objects if o.moving]
    if not moving:
        raise RuntimeError("perception found no moving object")
    objects, tracks_for_events, raw_tracks = [], [], {}
    for i, o in enumerate(moving):
        tr = track.track_object(fit["proc"], o.bbox_frame0 if o.visible_frame0 else None, o.bbox_last_frame)
        seg = track.moving_segment(tr)
        pts = [{"t_s": round(r["frame"] / fit_fps, 4),
                "x": round((r["box"][0] + r["box"][2] / 2) * sx, 1),
                "y": round((r["box"][1] + r["box"][3] / 2) * sy, 1)} for r in seg]
        raw_tracks[f"obj{i}"] = tr
        ref_box = o.bbox_last_frame if o.bbox_last_frame != [0, 0, 0, 0] else o.bbox_frame0
        bx0, by0, bx1, by1 = _norm_box_to_orig(ref_box, W, H)
        bw, bh = bx1 - bx0, by1 - by0
        objects.append({
            "id": f"obj{i}", "name": o.name, "shape": o.shape,
            "radius_px": round(min(bw, bh) / 2, 1) if o.shape == "circle" else None,
            "w_px": round(bw, 1) if o.shape == "box" else None, "h_px": round(bh, 1) if o.shape == "box" else None,
            "est_real_size_cm": o.est_real_size_cm, "track": pts,
            "visible_at_end": o.bbox_last_frame != [0, 0, 0, 0],
            "_box_last": [bx0, by0, bx1, by1],
        })
        tracks_for_events.append({"object": o.name, "frames_ok": sum(r["ok"] for r in tr), "frames_total": fit["n"],
                                  "points": [[p["t_s"], p["x"], p["y"]] for p in pts]})
    timings["track"] = time.time() - t0
    for ev in tracks_for_events:
        emit("tracking", ev)
    emit("stage", {"stage": "track", "status": "done", "message": f"{tracks_for_events[0]['frames_ok']} frames"})

    # ---- scale (main object = the one with the longest track)
    main = max(objects, key=lambda ob: len(ob["track"]))
    size_px = 2 * main["radius_px"] if main["shape"] == "circle" else main["w_px"]
    sc = scale.choose_scale(size_px, main["est_real_size_cm"], main["track"], scene_p.motion_type)
    emit("scale", sc)

    # ---- assets: background (moving objects inpainted), sprites, keyframes
    t0 = time.time()
    first_boxes = [_norm_box_to_orig(o.bbox_frame0, W, H) for o in moving if o.visible_frame0 and o.bbox_frame0 != [0, 0, 0, 0]]
    bg, bg_methods = _fill_background(last_full, first_full, [ob["_box_last"] for ob in objects if ob["visible_at_end"]], first_boxes)
    timings["background_method"] = bg_methods
    cv2.imwrite(str(work_dir / "background.png"), bg, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cv2.imwrite(str(work_dir / "last_frame.png"), last_full, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    for ob in objects:
        x0, y0, x1, y1 = ob.pop("_box_last")
        src = last_full if ob["visible_at_end"] else first_full
        spr = _sprite(src, (x0 + x1) / 2, (y0 + y1) / 2, ob["shape"], x1 - x0, y1 - y0)
        ob["sprite_png"] = str(work_dir / f"{ob['id']}.png")
        cv2.imwrite(ob["sprite_png"], spr)
    kf_frames = [int(round(p["t_s"] * fit_fps)) for p in main["track"]]
    picks = sorted(set(kf_frames[int(k)] for k in np.linspace(0, len(kf_frames) - 1, min(4, len(kf_frames))))) if kf_frames else []
    kf_full = read_video(fit_video or cond_video, keep_full=set(picks))["full"] if picks else {}
    keyframes = []
    for f_idx in picks:
        p = work_dir / f"kf_{f_idx:03d}.png"
        cv2.imwrite(str(p), kf_full[f_idx], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        keyframes.append({"t_s": round(f_idx / fit_fps, 4), "png": str(p)})
    timings["assets"] = time.time() - t0

    statics = []
    for o in scene_p.objects:
        if o.moving:
            continue
        b = o.bbox_last_frame if o.bbox_last_frame != [0, 0, 0, 0] else o.bbox_frame0
        statics.append({"name": o.name, "shape": "box", "bbox": [round(v, 1) for v in _norm_box_to_orig(b, W, H)]})
    surfaces = [{"name": s.name, "kind": s.kind,
                 "p0": [round(s.line[1] / 1000 * W, 1), round(s.line[0] / 1000 * H, 1)],
                 "p1": [round(s.line[3] / 1000 * W, 1), round(s.line[2] / 1000 * H, 1)]} for s in scene_p.support_surfaces]

    scene = {
        "clip_id": clip_id or Path(cond_video).stem, "work_dir": str(work_dir), "orig_size": [W, H],
        "fps_out": out_fps, "t_cond_end_s": round(t_cond_end, 4), "pred_seconds": config.PRED_SECONDS,
        "px_per_m": sc["px_per_m"], "px_per_m_source": sc["source"], "gravity_m_s2": scale.G,
        "motion_type_hint": scene_p.motion_type, "objects": objects, "static_objects": statics,
        "surfaces": surfaces, "background_png": str(work_dir / "background.png"),
        "last_frame_png": str(work_dir / "last_frame.png"), "keyframes": keyframes,
    }
    (work_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    info = {"perception": scene_p.model_dump(), "perception_meta": pmeta, "scale": sc, "timings_s": timings,
            "raw_tracks": raw_tracks, "fit_fps": fit_fps, "proc_size": list(fit["proc"][0].shape[1::-1])}
    return scene, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cond_video", type=Path)
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--fit-video", type=Path)
    ap.add_argument("--cache-dir", type=Path)
    a = ap.parse_args()
    s, info = build_scene(a.cond_video, a.work_dir, a.fit_video, a.cache_dir,
                          on_event=lambda n, p: print(f"[{n}]", json.dumps(p)[:300]))
    print(json.dumps({k: s[k] for k in ("px_per_m", "px_per_m_source", "t_cond_end_s", "motion_type_hint")}, indent=1))
    print(json.dumps(s["objects"], indent=1)[:1500])
    print("timings", info["timings_s"])
