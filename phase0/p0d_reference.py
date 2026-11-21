"""Independent reference for 0d: background-subtraction ball centroid (no model, no tracker).

For clips where the object is hidden at the start (ball inside the pipe), the median of the
early frames is an empty background; |frame - background| gives the ball silhouette.
Scores both CSRT (0b) and ER 2 (0d) against it and adds the result to 0d compare.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))
from vidio import read_frames  # noqa: E402


def bg_centroids(frames, bg_until: int, thr: int = 30, min_area: int = 300) -> dict[int, np.ndarray]:
    bg = np.median(np.stack(frames[:bg_until]), axis=0).astype(np.uint8)
    out = {}
    for i in range(bg_until, len(frames)):
        d = cv2.absdiff(frames[i], bg).max(axis=2)
        m = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n > 1:
            k = 1 + int(np.argmax(st[1:, 4]))
            if st[k, 4] >= min_area:
                out[i] = cen[k]
    return out


def main(scene: str = "ball-rolls-off", bg_until: int = 50) -> dict:
    b = ROOT / "phase0" / "out" / f"0b_{scene}"
    d = ROOT / "phase0" / "out" / f"0d_{scene}"
    frames, fps = read_frames(b / "conditioning_960x540.mp4")
    h, w = frames[0].shape[:2]
    ref = bg_centroids(frames, bg_until)
    csrt = {r["frame"]: np.array([r["box"][0] + r["box"][2] / 2, r["box"][1] + r["box"][3] / 2])
            for r in next(iter(json.loads((b / "track.json").read_text())["tracks"].values())) if r["ok"]}
    er2 = {int(round(p["t_s"] * fps)): np.array([p["point"][1] / 1000 * w, p["point"][0] / 1000 * h])
           for p in json.loads((d / "er2_points.json").read_text())["tracks"][0]["points"]}

    def err(track):
        e = [float(np.linalg.norm(track[f] - ref[f])) for f in sorted(track) if f in ref]
        return {"matched": len(e), "median_px": round(float(np.median(e)), 1) if e else None,
                "max_px": round(float(np.max(e)), 1) if e else None}

    res = {"reference": "background-subtraction centroid", "ref_frames": len(ref),
           "csrt_vs_ref": err(csrt), "er2_vs_ref": err(er2)}
    comp = json.loads((d / "compare.json").read_text())
    comp["independent_reference"] = res
    (d / "compare.json").write_text(json.dumps(comp, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:2]), indent=2))
