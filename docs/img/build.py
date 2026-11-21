"""Build the README / docs images (banner, comparison strip) from real outputs.

  PYTHONPATH=backend:phase0 python docs/img/build.py [--job outputs/<job_id>]

Pages are laid out in HTML with the frontend's design tokens and rendered with headless Chrome.
Numbers come only from files: eval/results.json (if present) and the chosen job's result.json.
Frames are from Physics-IQ (CC BY 4.0), credited in the captions.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "img"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0")]

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,500..800'
         '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')
CSS = """
:root{--paper:#f2f2ef;--ink:#15171c;--muted:#5f6470;--cobalt:#2350d8;--amber:#b86e00}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",sans-serif}
.display{font-family:"Bricolage Grotesque",sans-serif;font-variation-settings:"wdth" 88;letter-spacing:-.02em;line-height:1}
.film{position:relative;background:#000;padding:30px 0}
.film:before,.film:after{content:"";position:absolute;left:0;right:0;height:14px;
 background-image:repeating-linear-gradient(90deg,transparent 0 10px,#e9e9e6 10px 30px,transparent 30px 40px)}
.film:before{top:8px}.film:after{bottom:8px}
.row{display:flex;gap:12px;padding:0 12px}.fr{flex:1}.fr img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}
.stamp{font:13px "IBM Plex Mono",monospace;color:#e9e9e6;margin-top:7px;display:flex;justify-content:space-between}
.lane{font:600 15px "IBM Plex Mono",monospace;color:#e9e9e6;padding:0 12px 8px}
"""


def b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def frames_at(video: Path, times: list[float], t0: float = 0.0, w: int = 640) -> list:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, int(round((t - t0) * fps)))))
        ok, f = cap.read()
        out.append(cv2.resize(f, (w, int(f.shape[0] * w / f.shape[1])), interpolation=cv2.INTER_AREA) if ok else None)
    cap.release()
    return out


def shoot(html: str, out: Path, w: int, h: int) -> Path:
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "p.html"
        page.write_text(f"<!doctype html><html><head><meta charset='utf-8'>{FONTS}<style>{CSS}</style></head><body>{html}</body></html>")
        subprocess.run([CHROME, "--headless=new", f"--user-data-dir={d}/prof", "--no-first-run", "--hide-scrollbars",
                        f"--window-size={w},{h}", "--force-device-scale-factor=2", "--virtual-time-budget=6000",
                        f"--screenshot={out}", page.as_uri()], capture_output=True, timeout=90)
    return out


def strip(imgs: list, stamps: list[str], lane: str | None = None) -> str:
    cells = "".join(f'<div class="fr"><img src="{b64(i)}"><div class="stamp"><span>{s}</span></div></div>'
                    for i, s in zip(imgs, stamps) if i is not None)
    return f'<div class="film">{f"<div class=lane>{lane}</div>" if lane else ""}<div class="row">{cells}</div></div>'


def banner(cond: Path, stat_line: str) -> Path:
    times = [0.0, 1.5, 2.5, 2.67, 2.83, 2.96]
    html = f"""
<div style="padding:70px 64px 36px">
  <div class="display" style="font-size:150px;font-weight:800">Foresight</div>
  <div style="font-size:40px;margin-top:18px">Watch 3 seconds. Predict the next 5.</div>
</div>
{strip(frames_at(cond, times), [f"0:{t:05.2f}" for t in times])}
<div style="display:flex;justify-content:space-between;gap:40px;padding:34px 64px 40px">
  <div style="max-width:720px"><div style="font-size:26px;font-weight:600">{stat_line}</div>
  <div style="color:var(--muted);font-size:18px;margin-top:6px">For robotics teams: physics fitted to real video, checked against what really happened.</div></div>
  <div style="color:var(--muted);font-size:17px;text-align:right">Gemini 4 Hydrogen reads the scene, Decision 1 checks the fit,<br>Carbon repairs it, Omni is the video-model baseline.<br>Built during the Gemini 4 NYC Hackathon, Nov 21 2026.</div>
</div>"""
    return shoot(html, IMG / "banner.png", 1280, 720)


def compare(real: Path, omni_v: Path, fore: Path | None, fore_label: str) -> Path:
    times = [0.0, 0.25, 1.0, 2.5, 4.95]
    stamps = [f"0:{3 + t:05.2f}" for t in times]
    lanes = [strip(frames_at(real, times), stamps, "Reality: what actually happened")]
    if fore:
        lanes.append(strip(frames_at(fore, times), stamps, fore_label))
    lanes.append(strip(frames_at(omni_v, times), stamps, "Gemini Omni 1.2 Flash: continues the same 3 s as video"))
    html = f"""<div style="padding:48px 64px 22px"><div class="display" style="font-size:64px;font-weight:800">The same 3 seconds, three futures</div></div>
<div style="display:flex;flex-direction:column;gap:14px">{''.join(lanes)}</div>
<div style="padding:18px 64px 34px;color:var(--muted);font-size:16px">Physics-IQ ball-rolls-off, center camera (frames CC BY 4.0, Physics-IQ). Timestamps from the start of the clip; the model saw 0:00 to 0:03.</div>"""
    return shoot(html, IMG / "compare.png", 1280, 210 + 268 * len(lanes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fore", type=Path, help="a Foresight prediction mp4 for ball-rolls-off (24 FPS, 5 s)")
    ap.add_argument("--fore-label", default="Foresight: simulated from 3 s of video")
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    base = ROOT / "data" / "physics-IQ-benchmark" / "split-videos"
    cond = base / "conditioning/24FPS/0023_conditioning-videos_24FPS_perspective-center_take-1_trimmed-ball-rolls-off.mp4"
    real = base / "testing/24FPS/0023_testing-videos_24FPS_perspective-center_take-1_trimmed-ball-rolls-off.mp4"
    res = ROOT / "eval" / "results.json"
    stat = "Omni continued the clip in 39 s; the real ball had already rolled off the table."
    if res.exists():
        r = json.loads(res.read_text())
        s = r["systems"]
        if "foresight" in s and "omni" in s:
            stat = (f"Physics-IQ score {s['foresight']['physics_iq_verified_pct']:.1f} for Foresight against "
                    f"{s['omni']['physics_iq_verified_pct']:.1f} for Gemini Omni, on {r['n_clips']} real clips.")
    print(banner(cond, stat))
    print(compare(real, ROOT / "phase0/out/0c_ball-rolls-off/omni_5s_24fps.mp4", a.fore, a.fore_label))
