"""Fetch a small subset of Physics-IQ (one perspective) without downloading everything.

Preferred source: HF dataset Anates-Labs-Research/Physics-IQ-Verified (gated, needs HF_TOKEN / hf auth login).
Fallback source:  public GCS bucket gs://physics-iq-benchmark (original Physics-IQ GT), via HTTPS.

Files land in data/physics-IQ-benchmark-verified/ (or data/physics-IQ-benchmark/ for the fallback)
using the exact benchmark folder layout, so the official scoring code can read them.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PHYSIQ = ROOT / "third_party" / "physics-IQ-benchmark"
HF_REPO = "Anates-Labs-Research/Physics-IQ-Verified"
GCS_BASE = "https://storage.googleapis.com/physics-iq-benchmark"

VIEWS = ("left", "center", "right")

# Rigid-body scenes: a single object (or independent objects) rolling / sliding / falling / bouncing.
# Excluded: fluids, magnets, optics/mirrors, cloth, domino chains, fracture, pendulums, grabber placement.
RIGID_SINGLE = [
    "ball-hits-nothing", "ball-rolls-off", "ball-rolls-on-glass", "ball-ramp",
    "roll-behind-box", "roll-front-box", "roll-in-box", "ball-hits-duck",
    "weight-protects-duck", "ball-in-basket", "duck-falls-in-box", "ball-and-block-fall",
]
RIGID_MULTI = ["ball-train", "balls-collide", "two-balls-pass"]


def scenes() -> dict:
    d = yaml.safe_load(open(PHYSIQ / "descriptions" / "data.yaml"))
    return {s["scene"]: s for s in d["scenes"]}


def scene_ids(scene: str, view: str) -> tuple[str, str]:
    s = scenes()[scene]
    t1 = next(t for t in s["takes"] if t["take"] == 1)[view]
    t2 = next(t for t in s["takes"] if t["take"] == 2)[view]
    return f"{int(t1):04d}", f"{int(t2):04d}"


def needed_files(scene: str, view: str, fps: int) -> list[str]:
    """Relative paths (benchmark layout) needed to predict + score one scenario/view."""
    id1, id2 = scene_ids(scene, view)
    p = f"perspective-{view}"
    sc = f"trimmed-{scene}.mp4"
    return [
        f"split-videos/conditioning/{fps}FPS/{id1}_conditioning-videos_{fps}FPS_{p}_take-1_{sc}",
        f"split-videos/testing/{fps}FPS/{id1}_testing-videos_{fps}FPS_{p}_take-1_{sc}",
        f"split-videos/testing/{fps}FPS/{id2}_testing-videos_{fps}FPS_{p}_take-2_{sc}",
        f"video-masks/real/{fps}FPS/{id1}_video-masks_{fps}FPS_{p}_take-1_{sc}",
        f"video-masks/real/{fps}FPS/{id2}_video-masks_{fps}FPS_{p}_take-2_{sc}",
        f"switch-frames/{id1}_switch-frames_anyFPS_{p}_trimmed-{scene}.jpg",
    ]


def fetch_hf(paths: list[str], dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    for rel in paths:
        if (dest / rel).exists():
            continue
        hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=dest)
        print("hf  ", rel)


def fetch_gcs(paths: list[str], dest: Path) -> None:
    for rel in paths:
        out = dest / rel
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{GCS_BASE}/{rel}", out)
        print("gcs ", rel)


def benchmark_dir(source: str) -> Path:
    return DATA / ("physics-IQ-benchmark-verified" if source == "hf" else "physics-IQ-benchmark")


def fetch(scene_list: list[str], view: str, fps: int, source: str) -> Path:
    dest = benchmark_dir(source)
    dest.mkdir(parents=True, exist_ok=True)
    paths = [p for s in scene_list for p in needed_files(s, view, fps)]
    (fetch_hf if source == "hf" else fetch_gcs)(paths, dest)
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["ball-rolls-off", "ball-hits-nothing", "ball-in-basket"])
    ap.add_argument("--view", default="center", choices=VIEWS)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--source", default="hf", choices=["hf", "gcs"])
    ap.add_argument("--list", action="store_true", help="print the rigid-body scene list and exit")
    a = ap.parse_args()
    if a.list:
        all_scenes = scenes()
        print(f"{len(all_scenes)} scenes total; rigid single-object: {len(RIGID_SINGLE)}; rigid multi-ball: {len(RIGID_MULTI)}")
        for s in RIGID_SINGLE + RIGID_MULTI:
            assert s in all_scenes, s
            print(" ", s, "|", all_scenes[s]["subject-action"][:90])
        sys.exit(0)
    print("->", fetch(a.scenes, a.view, a.fps, a.source))
