"""Trivial baseline: hold the last conditioning frame for exactly 5.000 s at benchmark FPS."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase0"))
from physiq_data import scene_ids  # noqa: E402
from vidio import probe, read_frames, write_mp4  # noqa: E402


def conditioning_path(bench: Path, scene: str, view: str, fps: int) -> Path:
    id1, _ = scene_ids(scene, view)
    return bench / "split-videos" / "conditioning" / f"{fps}FPS" / (
        f"{id1}_conditioning-videos_{fps}FPS_perspective-{view}_take-1_trimmed-{scene}.mp4")


def pred_name(scene: str, view: str) -> str:
    id1, _ = scene_ids(scene, view)
    return f"{id1}_perspective-{view}_trimmed-{scene}.mp4"


def static_prediction(bench: Path, scene: str, view: str, fps: int, out_dir: Path) -> Path:
    frames, _ = read_frames(conditioning_path(bench, scene, view, fps))
    return write_mp4(out_dir / pred_name(scene, view), [frames[-1]] * (5 * fps), fps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--view", default="center")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--bench", type=Path, default=ROOT / "data" / "physics-IQ-benchmark-verified")
    ap.add_argument("--out", type=Path, default=ROOT / "phase0" / "out" / "preds" / "static-last-frame")
    a = ap.parse_args()
    for s in a.scenes:
        p = static_prediction(a.bench, s, a.view, a.fps, a.out)
        print(p.name, probe(p))
