"""Physicality auditor for 5 s video continuations (`foresight audit`, HANDOVER.md Option A).

Evaluates any video or world model's 5 s continuation from a 3 s conditioning clip across three
dimensions that pixel IoU (Physics-IQ) and stopped-track endpoint error miss:
  1. Object permanence: unexplained mid-frame vanishing vs. legitimate border exit or foreground
     occlusion (`physics.layers.assign_layers`), duplicate respawning, and size stability.
  2. Position error at fixed times (0.5, 1.0, 2.0, 3.0, 5.0 s + last real visible): vanished and
     early-exiting objects are penalized rather than dropped from the metric.
  3. Physical plausibility: tracked speed vs. the free-fall mechanical energy bound from
     `physics.implausible`, initialized from the conditioning exit velocity, plus teleport jumps.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config, track
from .scene import PROC_W, build_scene
from .video import read_video

EVAL_OFFSETS_S: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
SIZE_CHANGE_LIMIT = 1.65


def _load_scene(cond_video: Path, scene: dict[str, Any] | Path | str | None, *, fit_video: Path | None,
                cache_dir: Path | None, out_dir: Path | None) -> dict[str, Any]:
    if isinstance(scene, dict):
        return scene
    if scene is not None:
        return json.loads(Path(scene).read_text())
    work = (out_dir / "_scene_work") if out_dir is not None else Path(tempfile.mkdtemp(prefix="foresight_audit_"))
    built, _ = build_scene(cond_video, work, fit_video=fit_video, cache_dir=cache_dir)
    return built


def _load_clean_bg_proc(scene: dict[str, Any], cond_proc: list[np.ndarray], pw: int, ph: int) -> np.ndarray | None:
    bg_path = scene.get("background_png")
    if bg_path and Path(bg_path).exists():
        img = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
    if len(cond_proc) >= 3:
        sample = np.stack(cond_proc[:: max(1, len(cond_proc) // 9)], axis=0)
        return np.median(sample, axis=0).astype(np.uint8)
    return cond_proc[0].copy() if cond_proc else None


def _ref_patch(ob: dict[str, Any], cond_proc: list[np.ndarray], init_box: tuple[int, int, int, int]) -> np.ndarray | None:
    sp_path = ob.get("sprite_png")
    if sp_path and Path(sp_path).exists():
        raw = cv2.imread(str(sp_path), cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.size:
            return cv2.resize(raw[:, :, :3], (32, 32)).astype(np.float32)
    return track._patch(cond_proc[-1], init_box) if cond_proc else None


def _cond_velocity_and_ref(ob: dict[str, Any], scene: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (vx_px_s, vy_px_s, speed_m_s, y_ref_px) over the tail of the conditioning track."""
    pts = (ob.get("track") or [])[-6:]
    ppm = float(scene["px_per_m"])
    if len(pts) < 2:
        return 0.0, 0.0, 0.0, (float(pts[0]["y"]) if pts else 0.0)
    t = np.array([float(p["t_s"]) for p in pts], dtype=np.float64)
    xs = np.array([float(p["x"]) for p in pts], dtype=np.float64)
    ys = np.array([float(p["y"]) for p in pts], dtype=np.float64)
    dt = t - t[0]
    if float(dt[-1]) < 1e-6:
        return 0.0, 0.0, 0.0, float(ys.mean())
    A = np.stack([np.ones_like(dt), dt], axis=1)
    vx = float(np.linalg.lstsq(A, xs, rcond=None)[0][1])
    vy = float(np.linalg.lstsq(A, ys, rcond=None)[0][1])
    return vx, vy, float(math.hypot(vx, vy) / ppm), float(ys.mean())


def _track_object_continuation(scene: dict[str, Any], ob: dict[str, Any], frames_proc: list[np.ndarray], fps: float,
                               layers: Any, *, cond_proc: list[np.ndarray] | None = None,
                               bg_proc: np.ndarray | None = None, check_duplication: bool = False) -> dict[str, Any]:
    W, H = float(scene["orig_size"][0]), float(scene["orig_size"][1])
    ph, pw = frames_proc[0].shape[:2]
    sx, sy = pw / W, ph / H
    t_end = float(scene.get("t_cond_end_s", 3.0))
    oid = ob["id"]
    pts_cond = ob.get("track") or []
    if not pts_cond or not ob.get("visible_at_end", True):
        return {"name": ob.get("name", oid), "status": "not_visible_at_start", "vanished": False, "vanished_at_s": None,
                "duplicated": False, "duplicate_at_s": None, "duplicate_pos_px": None, "max_size_change_ratio": 1.0,
                "size_unstable": False, "tracked_frames": 0, "last_seen_px": None, "last_seen_t_s": None, "points": []}

    r_orig = float(ob.get("radius_px") or max(ob.get("w_px") or 20.0, ob.get("h_px") or 20.0) / 2.0)
    w_orig = float(ob.get("w_px") or 2.0 * r_orig) if ob.get("shape") == "box" else 2.0 * r_orig
    h_orig = float(ob.get("h_px") or 2.0 * r_orig) if ob.get("shape") == "box" else 2.0 * r_orig
    bw0, bh0 = max(4, int(round(w_orig * sx))), max(4, int(round(h_orig * sy)))
    r_proc = 0.5 * max(bw0, bh0)

    first, last = pts_cond[0], pts_cond[-1]
    init_box = tuple(track.clip_box(
        (int(round((last["x"] - w_orig / 2.0) * sx)), int(round((last["y"] - h_orig / 2.0) * sy)), bw0, bh0), pw, ph))
    hiders = [[x0 - 2.0 * r_orig, y0 - 2.0 * r_orig, x1 + 2.0 * r_orig, y1 + 2.0 * r_orig]
              for x0, y0, x1, y1 in layers.occluders.get(oid, [])]
    ball_ref = _ref_patch(ob, cond_proc or [], init_box)

    # Guard against immediate frame-0 disappearance (where pred[0] at init_box is already empty background).
    vanished_frame0 = False
    if bg_proc is not None and ball_ref is not None:
        p0, p_bg0 = track._patch(frames_proc[0], init_box), track._patch(bg_proc, init_box)
        if p0 is not None and p_bg0 is not None:
            fg0 = float(np.abs(p0 - p_bg0).mean())
            ball_vs_bg = float(np.abs(ball_ref - p_bg0).mean())
            if ball_vs_bg > 18.0 and fg0 < 4.5 and float(np.abs(p0 - ball_ref).mean()) > 20.0:
                vx_c, vy_c, _, _ = _cond_velocity_and_ref(ob, scene)
                dt_gap = max(0.0, t_end - float(last["t_s"]))
                ext_box = tuple(track.clip_box((int(round((last["x"] + vx_c * dt_gap - w_orig / 2.0) * sx)),
                                                int(round((last["y"] + vy_c * dt_gap - h_orig / 2.0) * sy)), bw0, bh0), pw, ph))
                pe, pe_bg = track._patch(frames_proc[0], ext_box), track._patch(bg_proc, ext_box)
                if ((float(np.abs(pe - pe_bg).mean()) if (pe is not None and pe_bg is not None) else 0.0) < 4.5):
                    vanished_frame0 = True
                else:
                    init_box = ext_box

    pts: list[dict[str, Any]] = []
    size_ratios: list[float] = [1.0]
    status = "visible_to_end"

    if vanished_frame0:
        cx0, cy0 = (init_box[0] + init_box[2] / 2.0) / sx, (init_box[1] + init_box[3] / 2.0) / sy
        status = "occluded" if any(h[0] < cx0 < h[2] and h[1] < cy0 < h[3] for h in hiders) else "vanished"
    else:
        raw_run = track.csrt_run(frames_proc, 0, init_box, +1)
        ref_init = track._patch(frames_proc[0], init_box)
        for rr in raw_run:
            fi = int(rr["frame"])
            x_b, y_b, bw, bh = rr["box"]
            cx_p, cy_p = x_b + bw / 2.0, y_b + bh / 2.0
            cx, cy = cx_p / sx, cy_p / sy

            ok = bool(rr["ok"])
            if ok and bg_proc is not None and ref_init is not None:
                p_cur = track._patch(frames_proc[fi], (x_b, y_b, bw, bh))
                p_bg = track._patch(bg_proc, (x_b, y_b, bw, bh))
                if p_cur is not None and p_bg is not None:
                    fg_l1 = float(np.abs(p_cur - p_bg).mean())
                    if float(np.abs(p_bg - ref_init).mean()) > 14.0 and fg_l1 < 4.5 and fg_l1 < 0.25 * max(rr["app"], 1.0):
                        ok = False

            if not ok:
                if len(pts) >= 2:
                    k_back = min(len(pts) - 1, 4)
                    df = max(1, pts[-1]["frame"] - pts[-1 - k_back]["frame"])
                    vx_pf = (pts[-1]["cx_p"] - pts[-1 - k_back]["cx_p"]) / df
                    vy_pf = (pts[-1]["cy_p"] - pts[-1 - k_back]["cy_p"]) / df
                    for step_m in np.linspace(1.0, max(3.0, 0.14 * fps), 6):
                        px_m, py_m = pts[-1]["cx_p"] + step_m * vx_pf, pts[-1]["cy_p"] + step_m * vy_pf
                        ox_m, oy_m = px_m / sx, py_m / sy
                        if any(h[0] < ox_m < h[2] and h[1] < oy_m < h[3] for h in hiders):
                            status = "occluded"
                            break
                        if px_m < r_proc or px_m > pw - r_proc or py_m < r_proc or py_m > ph - r_proc:
                            status = "exited_frame"
                            break
                    else:
                        status = "vanished"
                elif any(h[0] < cx < h[2] and h[1] < cy < h[3] for h in hiders):
                    status = "occluded"
                else:
                    status = "vanished"
                break

            if pts and any(h[0] < cx < h[2] and h[1] < cy < h[3] for h in hiders):
                status = "occluded"
                break
            if pts:
                prev_cx_p, prev_cy_p = pts[-1]["cx_p"], pts[-1]["cy_p"]
                out_l = x_b <= 2 or (x_b < bw / 2.0 and cx_p < prev_cx_p - 0.5)
                out_r = x_b + bw >= pw - 2 or (x_b + bw > pw - bw / 2.0 and cx_p > prev_cx_p + 0.5)
                out_t = y_b <= 2 or (y_b < bh / 2.0 and cy_p < prev_cy_p - 0.5)
                out_b = y_b + bh >= ph - 2 or (y_b + bh > ph - bh / 2.0 and cy_p > prev_cy_p + 0.5)
                if out_l or out_r or out_t or out_b:
                    status = "exited_frame"
                    break

            pts.append({"frame": fi, "t_s": round(t_end + fi / fps, 4), "x": float(cx), "y": float(cy),
                        "cx_p": float(cx_p), "cy_p": float(cy_p), "bw": int(bw), "bh": int(bh)})
            interior = (x_b > 1.5 * r_proc and y_b > 1.5 * r_proc and x_b + bw < pw - 1.5 * r_proc
                        and y_b + bh < ph - 1.5 * r_proc and not any(h[0] < cx < h[2] and h[1] < cy < h[3] for h in hiders))
            if interior:
                scale_est = math.sqrt((bw * bh) / float(max(1, bw0 * bh0)))
                if bg_proc is not None:
                    rx0, ry0 = max(0, int(round(cx_p - 2.2 * r_proc))), max(0, int(round(cy_p - 2.2 * r_proc)))
                    rx1, ry1 = min(pw, int(round(cx_p + 2.2 * r_proc))), min(ph, int(round(cy_p + 2.2 * r_proc)))
                    fg_px = int((np.abs(frames_proc[fi][ry0:ry1, rx0:rx1].astype(np.float32)
                                        - bg_proc[ry0:ry1, rx0:rx1].astype(np.float32)).mean(axis=2) > 22.0).sum())
                    base_area = math.pi * (0.5 * bw0) * (0.5 * bh0) if ob.get("shape") == "circle" else float(bw0 * bh0)
                    if fg_px >= 20 and base_area >= 16.0:
                        fg_scale = math.sqrt(fg_px / base_area)
                        if fg_scale > 1.85 and scale_est > 1.2:
                            scale_est = max(scale_est, 0.85 * fg_scale)
                size_ratios.append(max(scale_est, 1.0 / max(scale_est, 1e-3)))

    vanished = status == "vanished"
    vanished_at_s = round((pts[-1]["frame"] + 1) / fps, 3) if (vanished and pts) else (0.0 if vanished else None)

    duplicated, duplicate_at_s, duplicate_pos_px = False, None, None
    if check_duplication and bg_proc is not None and len(scene.get("objects", [])) == 1:
        ox_p, oy_p = float(first["x"]) * sx, float(first["y"]) * sy
        ref_patch = ball_ref if ball_ref is not None else track._patch(frames_proc[0], init_box)
        primary_by_frame = {p["frame"]: (p["cx_p"], p["cy_p"]) for p in pts}
        step_f, obj_area = max(2, int(round(0.25 * fps))), float(bw0 * bh0)
        dup_hits: list[tuple[int, float, float]] = []
        for fi in range(step_f, len(frames_proc), 2):
            if fi in primary_by_frame and math.hypot(primary_by_frame[fi][0] - ox_p, primary_by_frame[fi][1] - oy_p) < 4.0 * r_proc:
                continue
            if fi not in primary_by_frame and pts and math.hypot(pts[-1]["cx_p"] - ox_p, pts[-1]["cy_p"] - oy_p) < 4.0 * r_proc:
                continue

            fr, prev_fr = frames_proc[fi], frames_proc[fi - step_f]
            diff_bg = fr.astype(np.float32) - bg_proc.astype(np.float32)
            diff_bg -= np.median(diff_bg, axis=(0, 1), keepdims=True)
            mag_bg = np.abs(diff_bg).mean(axis=2)
            mag_mot = np.abs(fr.astype(np.float32) - prev_fr.astype(np.float32)).mean(axis=2)
            mask = ((mag_bg > 18.0) & (mag_mot > 12.0)).astype(np.uint8) * 255

            for f_chk in (fi, fi - step_f):
                if f_chk in primary_by_frame:
                    mx, my = primary_by_frame[f_chk]
                    cv2.circle(mask, (int(round(mx)), int(round(my))), int(round(2.4 * r_proc)), 0, -1)
            if pts and fi >= pts[-1]["frame"]:
                cv2.circle(mask, (int(round(pts[-1]["cx_p"])), int(round(pts[-1]["cy_p"]))), int(round(2.8 * r_proc)), 0, -1)
            for hx0, hy0, hx1, hy1 in hiders:
                cv2.rectangle(mask, (max(0, int(hx0 * sx)), max(0, int(hy0 * sy))),
                              (min(pw, int(hx1 * sx)), min(ph, int(hy1 * sy))), 0, -1)
            for rst in layers.resting:
                rx, ry = float(rst["x"]) * sx, float(rst["y"]) * sy
                rr_p = float(rst.get("radius_px") or max(rst.get("w_px") or 20.0, rst.get("h_px") or 20.0) / 2.0) * sx
                cv2.circle(mask, (int(round(rx)), int(round(ry))), int(round(2.5 * rr_p)), 0, -1)

            mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
                                    cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
            for li in range(1, n_lbl):
                if not (0.25 * obj_area <= float(stats[li, cv2.CC_STAT_AREA]) <= 3.0 * obj_area):
                    continue
                cx_b, cy_b = float(centroids[li][0]), float(centroids[li][1])
                near_origin = math.hypot(cx_b - ox_p, cy_b - oy_p) <= 3.5 * r_proc
                split_live = (fi in primary_by_frame and len(layers.resting) == 0
                              and math.hypot(cx_b - primary_by_frame[fi][0], cy_b - primary_by_frame[fi][1]) > 4.5 * r_proc)
                if not (near_origin or split_live):
                    continue
                cand_box = track.clip_box((int(round(cx_b - bw0 / 2.0)), int(round(cy_b - bh0 / 2.0)), bw0, bh0), pw, ph)
                p_cand = track._patch(fr, cand_box)
                app = float(np.abs(p_cand - ref_patch).mean()) if (p_cand is not None and ref_patch is not None) else 999.0
                if app < track.APPEARANCE_GATE * (1.15 if near_origin else 0.90):
                    dup_hits.append((fi, round(cx_b / sx, 1), round(cy_b / sy, 1)))
                    break
        if len(dup_hits) >= 3:
            duplicated = True
            duplicate_at_s = round(dup_hits[0][0] / fps, 3)
            duplicate_pos_px = [dup_hits[0][1], dup_hits[0][2]]

    max_ratio = round(float(max(size_ratios)), 3)
    return {
        "name": ob.get("name", oid), "status": status, "vanished": vanished, "vanished_at_s": vanished_at_s,
        "duplicated": duplicated, "duplicate_at_s": duplicate_at_s, "duplicate_pos_px": duplicate_pos_px,
        "max_size_change_ratio": max_ratio, "size_unstable": bool(max_ratio > SIZE_CHANGE_LIMIT),
        "tracked_frames": len(pts),
        "last_seen_px": [round(pts[-1]["x"], 1), round(pts[-1]["y"], 1)] if pts else None,
        "last_seen_t_s": pts[-1]["t_s"] if pts else None,
        "points": [{"t_s": p["t_s"], "x": round(p["x"], 1), "y": round(p["y"], 1)} for p in pts],
    }


def _interp_track(pts: list[dict[str, Any]], t_s: float, tol_s: float, *,
                  end_tol_s: float | None = None) -> tuple[float, float] | None:
    if not pts:
        return None
    hi_tol = tol_s if end_tol_s is None else end_tol_s
    if t_s < pts[0]["t_s"] - tol_s or t_s > pts[-1]["t_s"] + hi_tol:
        return None
    if t_s <= pts[0]["t_s"] or len(pts) == 1:
        return float(pts[0]["x"]), float(pts[0]["y"])
    if t_s >= pts[-1]["t_s"]:
        return float(pts[-1]["x"]), float(pts[-1]["y"])
    for a, b in zip(pts[:-1], pts[1:]):
        if a["t_s"] <= t_s <= b["t_s"]:
            w = (t_s - float(a["t_s"])) / max(1e-6, float(b["t_s"] - a["t_s"]))
            return float(a["x"] + w * (b["x"] - a["x"])), float(a["y"] + w * (b["y"] - a["y"]))
    return float(pts[-1]["x"]), float(pts[-1]["y"])


def _tail_velocity_px_s(pts: list[dict[str, Any]], fallback_vx: float, fallback_vy: float) -> tuple[float, float]:
    tail = pts[-6:]
    if len(tail) < 2:
        return fallback_vx, fallback_vy
    t = np.array([float(p["t_s"]) for p in tail], dtype=np.float64)
    dt = t - t[0]
    if float(dt[-1]) < 1e-6:
        return fallback_vx, fallback_vy
    A = np.stack([np.ones_like(dt), dt], axis=1)
    vx = float(np.linalg.lstsq(A, np.array([p["x"] for p in tail], dtype=np.float64), rcond=None)[0][1])
    vy = float(np.linalg.lstsq(A, np.array([p["y"] for p in tail], dtype=np.float64), rcond=None)[0][1])
    return vx, vy


def _evaluate_position_errors(scene: dict[str, Any], main_ob: dict[str, Any], pred_obj: dict[str, Any],
                              real_obj: dict[str, Any], fps: float) -> dict[str, Any]:
    W = float(scene["orig_size"][0])
    ppm = float(scene["px_per_m"])
    t_end = float(scene.get("t_cond_end_s", 3.0))
    tol_s = 0.55 / max(fps, 1.0)
    pred_end_tol = (1.15 / max(fps, 1.0)) if pred_obj["status"] == "visible_to_end" else tol_s
    real_end_tol = (1.15 / max(fps, 1.0)) if real_obj["status"] == "visible_to_end" else tol_s
    off_course_floor_cm = round(0.5 * W / ppm * 100.0, 1)
    max_extrap_cm = 1.5 * W / ppm * 100.0

    pred_pts, real_pts, cond_pts = pred_obj["points"], real_obj["points"], main_ob.get("track") or []
    vx_cond, vy_cond, _, _ = _cond_velocity_and_ref(main_ob, scene)
    vx_exit, vy_exit = _tail_velocity_px_s(pred_pts, vx_cond, vy_cond)
    last_pred_xy = ((float(pred_pts[-1]["x"]), float(pred_pts[-1]["y"]), float(pred_pts[-1]["t_s"])) if pred_pts else
                    ((float(cond_pts[-1]["x"]), float(cond_pts[-1]["y"]), t_end) if cond_pts else
                     (0.5 * W, 0.5 * float(scene["orig_size"][1]), t_end)))

    def _row(dt_s: float, t_s: float, real_xy: tuple[float, float] | None, pred_xy: tuple[float, float] | None,
             row_status: str, raw_cm: float | None, pen_cm: float | None, lingered_cm: float | None = None) -> dict[str, Any]:
        return {
            "dt_s": round(dt_s, 3), "t_s": t_s, "real_visible": real_xy is not None,
            "real_px": [round(real_xy[0], 1), round(real_xy[1], 1)] if real_xy else None,
            "pred_px": [round(pred_xy[0], 1), round(pred_xy[1], 1)] if pred_xy else None,
            "status": row_status, "raw_err_cm": raw_cm, "penalized_err_cm": pen_cm, "lingered_err_cm": lingered_cm,
        }

    def _eval_at(dt_s: float) -> dict[str, Any]:
        t_s = round(t_end + dt_s, 4)
        real_xy = _interp_track(real_pts, t_s, tol_s, end_tol_s=real_end_tol)
        pred_xy = _interp_track(pred_pts, t_s, tol_s, end_tol_s=pred_end_tol)
        if real_xy is None:
            if pred_xy is not None and real_pts and real_obj["status"] == "exited_frame":
                dist_cm = math.hypot(pred_xy[0] - float(real_pts[-1]["x"]), pred_xy[1] - float(real_pts[-1]["y"])) / ppm * 100.0
                if dist_cm > 8.0:
                    return _row(dt_s, t_s, None, pred_xy, "lingered_on_screen", None, None, round(dist_cm, 1))
            return _row(dt_s, t_s, None, pred_xy, "real_not_visible", None, None)
        if pred_xy is not None:
            err_cm = round(math.hypot(pred_xy[0] - real_xy[0], pred_xy[1] - real_xy[1]) / ppm * 100.0, 1)
            return _row(dt_s, t_s, real_xy, pred_xy, "tracked", err_cm, err_cm)
        last_cm = math.hypot(last_pred_xy[0] - real_xy[0], last_pred_xy[1] - real_xy[1]) / ppm * 100.0
        if pred_obj["status"] == "vanished":
            return _row(dt_s, t_s, real_xy, None, "vanished", None, round(max(last_cm, off_course_floor_cm), 1))
        if pred_obj["status"] == "exited_frame":
            dt_after = max(0.0, t_s - last_pred_xy[2])
            ex_x, ex_y = last_pred_xy[0] + vx_exit * dt_after, last_pred_xy[1] + vy_exit * dt_after
            extrap_cm = min(max_extrap_cm, math.hypot(ex_x - real_xy[0], ex_y - real_xy[1]) / ppm * 100.0)
            return _row(dt_s, t_s, real_xy, None, "exited_early", None, round(max(last_cm, extrap_cm), 1))
        return _row(dt_s, t_s, real_xy, None, "occluded", None, None)

    by_time = [_eval_at(dt) for dt in EVAL_OFFSETS_S]
    at_last_real, raw_endpoint_cm = None, None
    if real_pts:
        at_last_real = _eval_at(max(0.0, float(real_pts[-1]["t_s"]) - t_end))
        if pred_pts:
            end_r = real_pts[-1]
            nearest_p = min(pred_pts, key=lambda p: abs(float(p["t_s"]) - float(end_r["t_s"])))
            raw_endpoint_cm = round(
                math.hypot(float(nearest_p["x"]) - float(end_r["x"]), float(nearest_p["y"]) - float(end_r["y"])) / ppm * 100.0, 1)

    raw_vals = [r["raw_err_cm"] for r in by_time if r["raw_err_cm"] is not None]
    pen_vals = [r["penalized_err_cm"] for r in by_time if r["penalized_err_cm"] is not None]
    if not pen_vals and at_last_real and at_last_real["penalized_err_cm"] is not None:
        pen_vals = [at_last_real["penalized_err_cm"]]
    if not raw_vals and at_last_real and at_last_real["raw_err_cm"] is not None:
        raw_vals = [at_last_real["raw_err_cm"]]

    return {
        "eval_offsets_s": list(EVAL_OFFSETS_S), "by_time": by_time, "at_last_real_visible": at_last_real,
        "raw_endpoint_error_cm": raw_endpoint_cm,
        "penalized_endpoint_error_cm": at_last_real["penalized_err_cm"] if at_last_real else None,
        "mean_raw_cm": round(float(np.mean(raw_vals)), 1) if raw_vals else None,
        "mean_penalized_cm": round(float(np.mean(pen_vals)), 1) if pen_vals else None,
        "max_penalized_cm": round(float(max(pen_vals)), 1) if pen_vals else None,
        "off_course_floor_cm": off_course_floor_cm,
    }


def _evaluate_plausibility(scene: dict[str, Any], per_object_tracks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ppm = float(scene["px_per_m"])
    g = float(scene.get("gravity_m_s2", 9.81))
    t_end = float(scene.get("t_cond_end_s", 3.0))
    objs_by_id = {ob["id"]: ob for ob in scene.get("objects", [])}
    cond_speeds = {oid: _cond_velocity_and_ref(ob, scene) for oid, ob in objs_by_id.items()}
    max_scene_v0 = max((v[2] for v in cond_speeds.values()), default=0.0)

    violations: list[dict[str, Any]] = []
    main_id = max(scene["objects"], key=lambda ob: len(ob.get("track") or []))["id"] if scene.get("objects") else ""
    main_series: list[dict[str, float]] = []
    max_speed = max_bound = max_excess = 0.0
    main_v0 = cond_speeds.get(main_id, (0.0, 0.0, 0.0, 0.0))[2]

    for oid, tr_info in per_object_tracks.items():
        ob, pts = objs_by_id.get(oid), tr_info.get("points") or []
        if ob is None or len(pts) < 3:
            continue
        _, _, v0_self, y_ref_cond = cond_speeds[oid]
        v0 = v0_self if len(objs_by_id) == 1 else max(v0_self, max_scene_v0)
        size_px = float(2.0 * ob["radius_px"]) if ob.get("radius_px") else float(max(ob.get("w_px") or 20.0, ob.get("h_px") or 20.0))
        y_pred0 = float(pts[0]["y"])

        for k in range(1, len(pts)):
            dt1 = max(1e-4, float(pts[k]["t_s"] - pts[k - 1]["t_s"]))
            step_px = math.hypot(float(pts[k]["x"] - pts[k - 1]["x"]), float(pts[k]["y"] - pts[k - 1]["y"]))
            drop_m = max(0.0, (float(pts[k]["y"]) - y_ref_cond) / ppm, (float(pts[k]["y"]) - y_pred0) / ppm)
            bound = 1.3 * math.sqrt(v0 * v0 + 2.0 * g * drop_m) + 0.3
            if k >= 2:
                dt2 = max(1e-4, float(pts[k]["t_s"] - pts[k - 2]["t_s"]))
                speed = math.hypot(float(pts[k]["x"] - pts[k - 2]["x"]), float(pts[k]["y"] - pts[k - 2]["y"])) / dt2 / ppm
            else:
                speed = step_px / dt1 / ppm

            if oid == main_id:
                main_series.append({"t_s": round(float(pts[k]["t_s"]), 3), "dt_s": round(float(pts[k]["t_s"]) - t_end, 3),
                                    "speed_m_s": round(speed, 3), "bound_m_s": round(bound, 3)})
            max_speed, max_bound, max_excess = max(max_speed, speed), max(max_bound, bound), max(max_excess, speed / max(bound, 1e-6))

            if step_px > max(3.5 * size_px, 2.2 * bound * ppm * dt1 + 1.5 * size_px):
                violations.append({
                    "object_id": oid, "kind": "teleport_jump", "t_s": round(float(pts[k]["t_s"]), 3),
                    "dt_s": round(float(pts[k]["t_s"]) - t_end, 3), "step_px": round(step_px, 1),
                    "speed_m_s": round(step_px / dt1 / ppm, 2), "bound_m_s": round(bound, 2),
                    "reason": (f"{ob.get('name', oid)} jumped {step_px:.0f} px in {dt1 * 1000:.0f} ms at "
                               f"t=+{float(pts[k]['t_s']) - t_end:.2f} s ({step_px / dt1 / ppm:.1f} m/s vs {bound:.1f} m/s bound)"),
                })
            elif k >= 2 and speed > bound:
                violations.append({
                    "object_id": oid, "kind": "energy_bound_exceeded", "t_s": round(float(pts[k]["t_s"]), 3),
                    "dt_s": round(float(pts[k]["t_s"]) - t_end, 3), "speed_m_s": round(speed, 2), "bound_m_s": round(bound, 2),
                    "reason": (f"{ob.get('name', oid)} reached {speed:.2f} m/s at t=+{float(pts[k]['t_s']) - t_end:.2f} s, "
                               f"above the free-fall energy bound ({bound:.2f} m/s from v0={v0:.2f} m/s)"),
                })

    return {
        "plausible": len(violations) == 0, "v0_cond_m_s": round(main_v0, 3),
        "max_speed_m_s": round(max_speed, 3), "max_bound_m_s": round(max_bound, 3),
        "max_excess_ratio": round(max_excess, 3), "violations": violations,
        "reason": violations[0]["reason"] if violations else None, "speed_series": main_series,
    }


def _maybe_score_single_clip_physics_iq(cond_video: Path, pred_video: Path, bench_dir: Path, fps: int) -> float | None:
    """Compute single-clip Physics-IQ Verified (%) via phase0/score_subset.py when benchmark GT exists."""
    import score_subset

    cand_name = pred_video.name
    parts = cand_name.split("_")
    if not (len(parts) == 3 and parts[1].startswith("perspective-") and parts[2].startswith("trimmed-")):
        cparts = cond_video.name.split("_")
        persp = next((p for p in cparts if p.startswith("perspective-")), None)
        scen = next((p for p in cparts if p.startswith("trimmed-")), None)
        if not (cparts and cparts[0].isdigit() and persp and scen):
            return None
        cand_name = f"{cparts[0]}_{persp}_{scen}"

    with tempfile.TemporaryDirectory(prefix="foresight_piq_") as td:
        td_path = Path(td)
        stage_dir = td_path / "pred"
        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pred_video, stage_dir / cand_name)
        return float(score_subset.score_folder(stage_dir, bench_dir, fps, td_path / "out")["physics_iq_verified_score_pct"])


from .audit_html import render_audit_html


def audit_video(
    cond_video: Path | str,
    pred_video: Path | str,
    reality_video: Path | str | None = None,
    scene: dict[str, Any] | Path | str | None = None,
    out_dir: Path | str | None = None,
    physics_iq_score_pct: float | None = None,
    *,
    fit_video: Path | str | None = None,
    cache_dir: Path | str | None = None,
    bench_dir: Path | str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Audit a 5 s continuation video against physicality rules and optional ground truth."""
    from physics.layers import assign_layers

    cond_path, pred_path = Path(cond_video), Path(pred_video)
    real_path = Path(reality_video) if reality_video is not None else None
    out_path = Path(out_dir) if out_dir is not None else None
    sc = _load_scene(cond_path, scene, fit_video=Path(fit_video) if fit_video else None,
                     cache_dir=Path(cache_dir) if cache_dir else None, out_dir=out_path)

    pred_v = read_video(pred_path, proc_w=PROC_W)
    if not pred_v["proc"]:
        raise ValueError(f"Could not read frames from prediction video: {pred_path}")
    fps = float(pred_v["fps"] or sc.get("fps_out", config.BENCH_FPS))
    ph, pw = pred_v["proc"][0].shape[:2]

    cond_proc = read_video(cond_path, proc_w=PROC_W)["proc"] if cond_path.exists() else []
    bg_proc = _load_clean_bg_proc(sc, cond_proc, pw, ph)
    layers = assign_layers(sc)

    objects = [ob for ob in sc.get("objects", []) if ob.get("track")]
    if not objects:
        raise ValueError("Scene contains no tracked moving objects to audit.")
    main_ob = max(objects, key=lambda ob: len(ob["track"]))
    main_id = main_ob["id"]

    per_obj = {
        ob["id"]: _track_object_continuation(
            sc, ob, pred_v["proc"], fps, layers, cond_proc=cond_proc, bg_proc=bg_proc, check_duplication=(ob["id"] == main_id)
        )
        for ob in objects
    }
    main_perm = per_obj[main_id]
    any_vanished = any(v["vanished"] for v in per_obj.values())
    any_duplicated = any(v["duplicated"] for v in per_obj.values())
    any_size_unstable = any(v["size_unstable"] for v in per_obj.values())
    max_size_ratio = max((v["max_size_change_ratio"] for v in per_obj.values()), default=1.0)
    vanished_times = [v["vanished_at_s"] for v in per_obj.values() if v["vanished"] and v["vanished_at_s"] is not None]
    dup_times = [v["duplicate_at_s"] for v in per_obj.values() if v["duplicated"] and v["duplicate_at_s"] is not None]

    permanence = {
        "passed": not (any_vanished or any_duplicated or any_size_unstable), "status": main_perm["status"],
        "vanished": any_vanished, "vanished_at_s": min(vanished_times) if vanished_times else None,
        "duplicated": any_duplicated, "duplicate_at_s": min(dup_times) if dup_times else None,
        "size_unstable": any_size_unstable, "max_size_change_ratio": round(max_size_ratio, 3),
        "occluded": main_perm["status"] == "occluded", "exited_frame": main_perm["status"] == "exited_frame",
        "tracked_frames": main_perm["tracked_frames"], "total_frames": int(pred_v["n"]),
        "per_object": {oid: {k: v for k, v in info.items() if k != "points"} for oid, info in per_obj.items()},
    }
    plausibility = _evaluate_plausibility(sc, per_obj)

    pos_error, real_pts_out = None, None
    if real_path is not None:
        real_v = read_video(real_path, proc_w=PROC_W)
        if real_v["proc"]:
            real_info = _track_object_continuation(
                sc, main_ob, real_v["proc"], float(real_v["fps"] or fps), layers,
                cond_proc=cond_proc, bg_proc=bg_proc, check_duplication=False,
            )
            real_pts_out = real_info["points"]
            pos_error = _evaluate_position_errors(sc, main_ob, main_perm, real_info, fps)

    piq_pct = round(float(physics_iq_score_pct), 2) if physics_iq_score_pct is not None else None
    if piq_pct is None and bench_dir is not None and Path(bench_dir).exists():
        piq_pct = _maybe_score_single_clip_physics_iq(cond_path, pred_path, Path(bench_dir), int(round(fps)))

    reasons: list[str] = []
    for oid, info in per_obj.items():
        if info["vanished"]:
            reasons.append(f"{info['name']} ({oid}) vanished mid-frame at +{info['vanished_at_s']:.2f} s away from borders and occluders.")
        if info["duplicated"]:
            reasons.append(f"{info['name']} ({oid}) duplicated / respawned at +{info['duplicate_at_s']:.2f} s.")
        if info["size_unstable"]:
            reasons.append(f"{info['name']} ({oid}) changed apparent size by {info['max_size_change_ratio']:.2f}x (> {SIZE_CHANGE_LIMIT:.2f}x).")
    if not plausibility["plausible"] and plausibility["reason"]:
        reasons.append(plausibility["reason"])

    passed = bool(permanence["passed"] and plausibility["plausible"])
    report: dict[str, Any] = {
        "clip_id": sc.get("clip_id") or pred_path.stem, "label": label,
        "cond_video": str(cond_path), "pred_video": str(pred_path),
        "reality_video": str(real_path) if real_path is not None else None,
        "orig_size": list(sc["orig_size"]), "fps": round(fps, 3),
        "px_per_m": float(sc["px_per_m"]), "px_per_m_source": sc.get("px_per_m_source", "size"),
        "t_cond_end_s": float(sc.get("t_cond_end_s", 3.0)),
        "pred_duration_s": round(pred_v["n"] / max(fps, 1e-6), 3),
        "physics_iq_score_pct": piq_pct,
        "verdict": {
            "passed": passed, "permanence_ok": permanence["passed"], "plausible": plausibility["plausible"],
            "summary": "Physicality checks passed." if passed else "; ".join(reasons), "reasons": reasons,
        },
        "permanence": permanence, "position_error": pos_error, "plausibility": plausibility,
        "tracks": {
            "conditioning": main_ob.get("track") or [], "prediction": main_perm["points"], "reality": real_pts_out,
            "occluders": layers.occluders.get(main_id, []), "surfaces": sc.get("surfaces") or [],
        },
    }
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "audit.json").write_text(json.dumps(report, indent=2))
        (out_path / "audit.html").write_text(render_audit_html(report))
    return report


audit_continuation = audit_video

__all__ = ["EVAL_OFFSETS_S", "SIZE_CHANGE_LIMIT", "audit_continuation", "audit_video", "render_audit_html"]
