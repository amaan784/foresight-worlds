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


def _bottom_line(ob: dict) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Line y = a + b x through the tracked bottoms of an object; returns (a, b, rms residual, xs, bottoms)."""
    r = ob.get("radius_px") or (ob.get("h_px") or 0) / 2
    xs = np.array([p["x"] for p in ob["track"]])
    ys = np.array([p["y"] for p in ob["track"]]) + r
    b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b), float(np.sqrt(np.mean((a + b * xs - ys) ** 2))), xs, ys


def snap_support(surfaces: list[dict], objects: list[dict], main: dict, motion: str,
                 W: float | None = None) -> tuple[list[dict], dict | None]:
    """For rolling/sliding objects the contact line is where the tracked objects' bottoms actually are.

    Hydrogen sometimes draws a table's back edge (higher in the image, in perspective) instead of the line the
    object rolls on, which puts the object under its own support. Objects rolling at the same depth as the main
    object (bottoms within 1.2 radii of each other's lines) share one line fitted through all their bottoms; the
    nearest table/floor/ramp is moved onto it, keeping its left/right ends (so the object can still fall off).
    Objects rolling at another depth (two balls passing, one nearer the camera) get their own line.
    """
    if motion not in ("roll", "slide") or len(main["track"]) < 3:
        return surfaces, None

    def radius(ob):
        return ob.get("radius_px") or (ob.get("h_px") or 0) / 2

    rolling = [main] + [ob for ob in objects if ob is not main and len(ob["track"]) >= 3
                        and _bottom_line(ob)[2] <= 0.3 * radius(ob)]
    groups: list[list[dict]] = []
    for ob in rolling:
        a, b, _, xs, ys = _bottom_line(ob)
        for g in groups:
            ga, gb, _, gx, gy = _bottom_line(g[0])
            off = min(float(np.median(np.abs(ga + gb * xs - ys))), float(np.median(np.abs(a + b * gx - gy))))
            if off <= 1.2 * radius(ob):
                g.append(ob)
                break
        else:
            groups.append([ob])

    def y_at(s, x):
        (x0, y0), (x1, y1) = s["p0"], s["p1"]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0

    info: dict = {"depth_groups": [[ob["id"] for ob in g] for g in groups]}
    for gi, g in enumerate(groups):
        xs = np.concatenate([_bottom_line(ob)[3] for ob in g])
        ys = np.concatenate([_bottom_line(ob)[4] for ob in g])
        r = max(radius(ob) for ob in g)
        b, a = np.polyfit(xs, ys, 1)
        xm = float(xs.mean())
        Wl = float(W) if W else max(float(xs.max()) * 1.5, 1.0)
        if gi > 0:  # another depth: its own line across the frame
            name = f"support under {g[0]['name']} (from track)"
            surfaces = surfaces + [{"name": name, "kind": "table", "p0": [0.0, round(float(a), 1)],
                                    "p1": [Wl, round(float(a + b * Wl), 1)]}]
            info.setdefault("own_supports", []).append(name)
            continue
        cands = [s for s in surfaces if s["kind"] in ("table", "floor", "ramp")] or surfaces
        if not cands:
            surfaces = surfaces + [{"name": "support (from track)", "kind": "table", "p0": [0.0, round(float(a), 1)],
                                    "p1": [Wl, round(float(a + b * Wl), 1)]}]
            info |= {"created": True, "slope": round(float(b), 4)}
            continue
        s = min(cands, key=lambda c: abs(y_at(c, xm) - (a + b * xm)))
        before = y_at(s, xm)
        x0 = min(s["p0"][0], s["p1"][0], float(xs.min()) - r)
        x1 = max(s["p0"][0], s["p1"][0], float(xs.max()) + r)
        if W and abs((a + b * xm) - before) > 2 * r:
            # Hydrogen drew a different edge (e.g. the table's back edge in perspective); its ends are not the
            # ends of the line the object rolls on, so let the support span the frame.
            x0, x1 = 0.0, float(W)
        s["p0"], s["p1"] = [round(float(x0), 1), round(float(a + b * x0), 1)], [round(float(x1), 1), round(float(a + b * x1), 1)]
        info |= {"surface": s["name"], "shift_px_at_track": round(float(a + b * xm - before), 1), "slope": round(float(b), 4)}
    return surfaces, info


def drop_source_statics(statics: list[dict], objects: list[dict]) -> tuple[list[dict], list[str]]:
    """Remove static objects the moving object overlaps while tracked (the pipe it rolls out of, the hand or
    grabber that drops it). In the simulation they would start interpenetrating the body and launch it."""
    keep, dropped = [], []
    for st in statics:
        x0, y0, x1, y1 = st["bbox"]
        hit = False
        for ob in objects:
            r = ob.get("radius_px") or max(ob.get("w_px") or 0, ob.get("h_px") or 0) / 2
            for p in ob["track"]:
                if x0 - r < p["x"] < x1 + r and y0 - r < p["y"] < y1 + r:
                    hit = True
                    break
        (dropped if hit else keep).append(st if not hit else st["name"])
        if hit:  # points where the object is still partly inside its source are occluded: unreliable centres
            for ob in objects:
                r = ob.get("radius_px") or max(ob.get("w_px") or 0, ob.get("h_px") or 0) / 2
                clear = [p for p in ob["track"] if not (x0 - r < p["x"] < x1 + r and y0 - r < p["y"] < y1 + r)]
                if len(clear) >= 3:
                    ob["track"] = clear
    return keep, dropped


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
    objects, tracks_for_events, raw_tracks, cleaning = [], [], {}, {}
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
        pts, cleaning[f"obj{i}"] = track.clean_track(pts, min(bw, bh) if o.shape == "circle" else max(bw, bh))
        objects.append({
            "id": f"obj{i}", "name": o.name, "shape": o.shape,
            "radius_px": round(min(bw, bh) / 2, 1) if o.shape == "circle" else None,
            "w_px": round(bw, 1) if o.shape == "box" else None, "h_px": round(bh, 1) if o.shape == "box" else None,
            "est_real_size_cm": o.est_real_size_cm, "est_mass_g": o.est_mass_g, "material": o.material, "track": pts,
            "visible_at_end": o.bbox_last_frame != [0, 0, 0, 0],
            "_box_last": [bx0, by0, bx1, by1],
        })
        tracks_for_events.append({"object": o.name, "frames_ok": sum(r["ok"] for r in tr), "frames_total": fit["n"],
                                  "points": [[p["t_s"], p["x"], p["y"]] for p in pts]})
    timings["track"] = time.time() - t0
    for ev in tracks_for_events:
        emit("tracking", ev)
    emit("stage", {"stage": "track", "status": "done", "message": f"{tracks_for_events[0]['frames_ok']} frames"})

    proc_size = list(fit["proc"][0].shape[1::-1])
    del fit
    cond["proc"].clear()
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
    kf_full = read_video(fit_video or cond_video, keep_full=set(picks), keep_proc=False)["full"] if picks else {}
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
        x0, y0, x1, y1 = _norm_box_to_orig(b, W, H)
        # a sprite too: the simulation may knock a static object that stands in a moving object's path
        sprite = work_dir / f"static{len(statics)}.png"
        cv2.imwrite(str(sprite), _sprite(last_full, (x0 + x1) / 2, (y0 + y1) / 2, o.shape, x1 - x0, y1 - y0))
        statics.append({"name": o.name, "shape": "box", "bbox": [round(v, 1) for v in (x0, y0, x1, y1)],
                        "perceived_shape": o.shape, "est_real_size_cm": o.est_real_size_cm, "est_mass_g": o.est_mass_g,
                        "material": o.material,
                        "sprite_png": str(sprite)})
    surfaces = [{"name": s.name, "kind": s.kind, "material": s.material,
                 "p0": [round(s.line[1] / 1000 * W, 1), round(s.line[0] / 1000 * H, 1)],
                 "p1": [round(s.line[3] / 1000 * W, 1), round(s.line[2] / 1000 * H, 1)]} for s in scene_p.support_surfaces]
    surfaces, snap = snap_support(surfaces, objects, main, scene_p.motion_type, W)
    statics, dropped = drop_source_statics(statics, objects)
    timings["surface_snap"] = snap
    timings["statics_dropped"] = dropped
    timings["track_cleaning"] = cleaning

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
            "raw_tracks": raw_tracks, "fit_fps": fit_fps, "proc_size": proc_size}
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
