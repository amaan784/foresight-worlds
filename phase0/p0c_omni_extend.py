"""Phase 0c: can gemini-omni-1.2-flash CONTINUE a supplied 3 s clip? Run it once and save the output.

Docs: gemini-api-guides/omni.md#video-extension (interactions.create with an uploaded video + text;
output in interaction.output_video.data). SDK 3.6.1 also exposes generation_config.video_config.task="extend".
Writes phase0/out/0c_<scene>/{omni_raw.mp4, omni_5s_24fps.mp4, report.json}
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "phase0"))

from foresight import config, gemini  # noqa: E402
from physiq_data import PHYSIQ, benchmark_dir  # noqa: E402
from static_pred import conditioning_path, pred_name  # noqa: E402
from vidio import probe, read_frames, write_mp4  # noqa: E402

IN_W, IN_H = 1280, 720


def benchmark_prompt(scene: str, view: str) -> str:
    """Official Physics-IQ Verified base ('bpp') prompt for this scenario."""
    with open(PHYSIQ / "descriptions" / "best_practice" / "descriptions_base.csv") as f:
        for row in csv.DictReader(f):
            if row["scenario"].endswith(f"perspective-{view}_take-1_trimmed-{scene}.mp4"):
                return row["description"]
    raise KeyError(scene)


def extract_5s(raw: Path, input_s: float, fps: int, out: Path) -> tuple[Path, dict]:
    """Take the 5 s after the input clip (if Omni returns input+extension) else the first 5 s; resample to fps."""
    frames, rfps = read_frames(raw)
    dur = len(frames) / rfps
    start = input_s if dur >= input_s + 4.0 else 0.0
    picked = []
    for k in range(int(5 * fps)):
        src = int(round((start + k / fps) * rfps))
        picked.append(frames[min(src, len(frames) - 1)])
    padded = max(0, int(round((start + 5.0) * rfps)) - len(frames))
    return write_mp4(out, picked, fps), {"raw_duration_s": round(dur, 3), "raw_fps": rfps,
                                         "start_offset_s": start, "frames_padded_by_holding_last": padded}


def main(scene: str, source: str, view: str = "center", fps: int = config.BENCH_FPS, with_prompt: bool = True) -> dict:
    out_dir = ROOT / "phase0" / "out" / f"0c_{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    full, cfps = read_frames(conditioning_path(benchmark_dir(source), scene, view, fps))
    small = write_mp4(out_dir / f"conditioning_{IN_W}x{IN_H}.mp4",
                      [cv2.resize(f, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for f in full], cfps)
    f = gemini.upload_video(small)
    text = (benchmark_prompt(scene, view) + " Continue the scene.") if with_prompt else "Continue the scene."
    c = gemini.client()
    t0 = time.time()
    resp = gemini.call(
        c.interactions.create, kind="interactions.create", model=config.OMNI, tag=f"0c:{scene}",
        input=[{"type": "video", "uri": f.uri, "mime_type": f.mime_type}, {"type": "text", "text": text}],
        generation_config={"video_config": {"task": "extend"}},
    )
    latency = time.time() - t0
    vid = resp.output_video
    if vid is None or not getattr(vid, "data", None):
        report = {"scene": scene, "ok": False, "latency_s": round(latency, 1), "output_text": resp.output_text}
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report
    raw = out_dir / "omni_raw.mp4"
    raw.write_bytes(base64.b64decode(vid.data) if isinstance(vid.data, str) else vid.data)
    five, ex = extract_5s(raw, len(full) / cfps, fps, out_dir / "omni_5s_24fps.mp4")
    bench_copy = ROOT / "phase0" / "out" / "preds_omni" / "omni-1.2-flash" / pred_name(scene, view)
    bench_copy.parent.mkdir(parents=True, exist_ok=True)
    bench_copy.write_bytes(five.read_bytes())
    usage = resp.usage.model_dump(mode="json", exclude_none=True) if getattr(resp, "usage", None) else None
    report = {
        "scene": scene, "ok": True, "model": config.OMNI, "prompt": text, "latency_s": round(latency, 1),
        "raw": probe(raw), "extracted_5s": probe(five), **ex, "usage": usage,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="ball-rolls-off")
    ap.add_argument("--source", default="gcs", choices=["hf", "gcs"])
    ap.add_argument("--no-prompt", action="store_true")
    a = ap.parse_args()
    if not gemini.have_key():
        sys.exit("GEMINI_API_KEY missing: add it to .env")
    print(json.dumps(main(a.scene, a.source, with_prompt=not a.no_prompt), indent=2))
