"""Render a ~30 s demo video from a real stage run and the eval results (no number typed by hand).

  PYTHONPATH=backend:phase0 python docs/img/demo_video.py outputs/<job_id>  ->  docs/demo.mp4

Cards are laid out in HTML with the stage's design tokens and rendered with headless Chrome; the real videos
from the run are pasted into the card slots frame by frame.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "docs" / "img"), str(ROOT / "phase0")]
from build import CHROME, FONTS  # noqa: E402

W, H, FPS = 1920, 1080, 24
CSS = """
:root{--paper:#f2f2ef;--ink:#15171c;--muted:#5f6470;--cobalt:#2350d8;--amber:#b86e00;--good:#1f7a4d}
*{box-sizing:border-box}html,body{margin:0;width:1920px;height:1080px;background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans",sans-serif;overflow:hidden}
.d{font-family:"Bricolage Grotesque",sans-serif;font-variation-settings:"wdth" 88;letter-spacing:-.02em;line-height:1}
.slot{position:absolute;border-radius:20px;background:#0d0e11;box-shadow:0 0 0 2px #d5d6d1}
.ring{position:absolute;border-radius:20px;pointer-events:none}
.pill{position:absolute;display:inline-flex;align-items:center;gap:10px;padding:7px 16px 7px 13px;border-radius:999px;
background:rgba(13,14,17,.72);color:#f2f2ef;font:600 22px "IBM Plex Sans",sans-serif}
.pill i{width:12px;height:12px;border-radius:50%;display:inline-block}
.step{display:grid;grid-template-columns:70px 1fr;gap:18px;padding:16px 0;border-bottom:2px solid #d5d6d1}
.step .n{font-family:"Bricolage Grotesque",sans-serif;font-weight:800;font-size:48px;color:var(--cobalt);line-height:1}
.step .t{font-size:30px;font-weight:600}.step .s{font-size:24px;color:var(--muted);margin-top:4px}
"""


def card(html: str, path: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "c.html"
        page.write_text(f"<!doctype html><html><head><meta charset='utf-8'>{FONTS}<style>{CSS}</style></head><body>{html}</body></html>")
        subprocess.run([CHROME, "--headless=new", f"--user-data-dir={d}/p", "--no-first-run", "--hide-scrollbars",
                        f"--window-size={W},{H}", "--force-device-scale-factor=1", "--virtual-time-budget=5000",
                        f"--screenshot={path}", page.as_uri()], capture_output=True, timeout=90)
    return cv2.resize(cv2.imread(str(path)), (W, H))


def frames_of(video: Path) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(str(video)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def paste(bg: np.ndarray, f: np.ndarray, x: int, y: int, w: int, h: int, r: int = 20) -> None:
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    roi = bg[y:y + h, x:x + w]
    roi[mask > 0] = f[mask > 0]


def main(job: Path) -> Path:
    res = json.loads((job / "result.json").read_text())
    ev = [json.loads(line) for line in open(job / "events.jsonl")]
    get = lambda name: next((e["data"] for e in ev if e["event"] == name), {})  # noqa: E731
    per, trk, sc, fit, ver = get("perception"), get("tracking"), get("scale"), get("fit"), get("verify")
    m = res["metrics"]
    evr = json.loads((ROOT / "eval" / "results.json").read_text())
    S = evr["systems"]
    tmp = Path(tempfile.mkdtemp())
    seq: list[np.ndarray] = []
    hold = lambda img, s: seq.extend([img] * int(s * FPS))  # noqa: E731

    # 1. title
    hold(card("""<div style="padding:150px 140px"><div class="d" style="font-size:230px;font-weight:800">Foresight</div>
      <div style="font-size:64px;margin-top:40px">Watch 3 seconds. Predict the next 5.</div>
      <div style="font-size:34px;color:var(--muted);margin-top:30px;max-width:1400px">For robotics teams and world-model builders: physics
      fitted to real video with Gemini, instead of a video model making up motion.</div></div>""", tmp / "t.png"), 4.0)

    # 2. the 3 seconds the model sees (slot 1280x720)
    sx, sy, sw, sh = 320, 230, 1280, 720
    bg = card(f"""<div style="position:absolute;left:{sx}px;top:80px" class="d"><span style="font-size:72px;font-weight:800">What the model sees: 3 seconds</span></div>
      <div class="slot" style="left:{sx}px;top:{sy}px;width:{sw}px;height:{sh}px"></div>
      <div style="position:absolute;left:{sx}px;top:{sy + sh + 30}px;font-size:28px;color:var(--muted)">Physics-IQ ball-rolls-off, center camera (CC BY 4.0). The next 5 seconds stay hidden.</div>""", tmp / "c.png")
    for f in frames_of(job / "conditioning.mp4"):
        im = bg.copy()
        paste(im, f, sx, sy, sw, sh)
        seq.append(im)

    # 3. the steps, with this run's numbers
    obj = next((o for o in per.get("objects", []) if o.get("moving")), {})
    steps = [
        ("1", f"{per.get('model', 'gemini-4-hydrogen')} reads the scene", f"{obj.get('name')}, a {obj.get('shape')} about {obj.get('est_real_size_cm')} cm across; motion: {per.get('motion_type')}; {per.get('latency_s')} s"),
        ("2", "Track it frame by frame", f"found in {trk.get('frames_ok')} of {trk.get('frames_total')} frames (OpenCV CSRT)"),
        ("3", "Measure the scene", f"{round(sc.get('px_per_m', 0)):,} pixels per metre, from the object's size"),
        ("4", "Fit physics to what it saw", f"{fit.get('motion_model')}, {fit.get('fit_error_px'):.1f} px from the tracked path"),
        ("5", "Gemini Decision 1 checks the fit", "outline on object " + f"{ver.get('confidences', {}).get('outline_on_object', 0):.2f}, motion '{ver.get('answers', {}).get('motion')}' {ver.get('confidences', {}).get('motion', 0):.2f}: passed"),
        ("6", "Render the next 5 seconds", f"predicted in {m['latency_s_total']} s; Gemini Omni continues the same clip in parallel"),
    ]
    rows = "".join(f'<div class="step"><div class="n">{n}</div><div><div class="t">{t}</div><div class="s">{s}</div></div></div>' for n, t, s in steps)
    hold(card(f"""<div style="padding:80px 160px"><div class="d" style="font-size:72px;font-weight:800;margin-bottom:30px">Live, step by step</div>{rows}</div>""", tmp / "s.png"), 6.5)

    # 4. reveal: Reality | Foresight | Omni at half speed, endpoint errors underneath
    pw, ph, gap, top = 576, 324, 36, 250
    x0 = (W - 3 * pw - 2 * gap) // 2
    lanes = [("Reality", "#f2f2ef", "none", "reality.mp4", "Reference", "the hidden 5 seconds, filmed", "var(--ink)"),
             ("Foresight (physics)", "#2350d8", "#2350d8", "foresight.mp4", f"{m['endpoint_error_cm']['foresight']} cm", "from where the real ball ended up", "var(--good)"),
             ("Omni (video model)", "#b86e00", "#b86e00", "omni.mp4", f"{m['endpoint_error_cm']['omni']} cm", "from where the real ball ended up", "var(--ink)")]
    html = f'<div style="position:absolute;left:{x0}px;top:80px" class="d"><span style="font-size:72px;font-weight:800">The next 5 seconds, three ways</span></div>'
    for i, (title, dot, ring, _, big, sub, col) in enumerate(lanes):
        x = x0 + i * (pw + gap)
        html += (f'<div class="slot" style="left:{x}px;top:{top}px;width:{pw}px;height:{ph}px"></div>'
                 f'<div style="position:absolute;left:{x}px;top:{top + ph + 40}px;width:{pw}px">'
                 f'<div style="font-size:26px;color:var(--muted)">{sub}</div>'
                 f'<div class="d" style="font-size:88px;font-weight:800;color:{col}">{big}</div></div>')
    html += f'<div style="position:absolute;left:{x0}px;top:{top + ph + 230}px;font-size:28px;color:var(--muted)">Played at half speed. Endpoint error: how far each prediction ends up from the real ball at the last moment it is visible.</div>'
    bg = card(html, tmp / "r.png")
    vids = [frames_of(job / lane[3]) for lane in lanes]
    n = min(len(v) for v in vids)
    for k in range(n):
        im = bg.copy()
        for i, lane in enumerate(lanes):
            x = x0 + i * (pw + gap)
            paste(im, vids[i][k], x, top, pw, ph)
            if lane[2] != "none":
                cv2.rectangle(im, (x + 2, top + 2), (x + pw - 3, top + ph - 3), tuple(int(lane[2][j:j + 2], 16) for j in (5, 3, 1)), 5)
            cv2.rectangle(im, (x + 14, top + 14), (x + 30 + 13 * len(lane[0]), top + 52), (17, 14, 13), -1)
            cv2.circle(im, (x + 30, top + 33), 7, tuple(int(lane[1][j:j + 2], 16) for j in (5, 3, 1)), -1)
            cv2.putText(im, lane[0], (x + 46, top + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (239, 242, 242), 2, cv2.LINE_AA)
            cv2.putText(im, f"0:{3 + k / FPS:05.2f}", (x + pw - 120, top + ph - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (239, 242, 242), 2, cv2.LINE_AA)
        seq.extend([im, im])  # half speed

    # 5. results chart
    ranked = sorted(S.items(), key=lambda kv: -kv[1]["physics_iq_verified_pct"])
    names = {"foresight": "Foresight", "foresight_carbon": "Foresight (Carbon reads)", "genie": "Genie 4", "omni": "Gemini Omni 1.2",
             "const_velocity": "Constant velocity", "static": "Last frame held"}
    line = ", ".join(f"{names.get(k, k)} {v['physics_iq_verified_pct']:.1f}" for k, v in ranked)
    chart = cv2.imread(str(ROOT / "eval" / "chart.png"))
    bg = card(f"""<div style="padding:70px 140px"><div class="d" style="font-size:66px;font-weight:800">Physics-IQ, {evr['n_clips']} real clips, official metric code</div>
      <div style="font-size:26px;color:var(--muted);margin-top:18px;max-width:1640px">{line}. Foresight beats both video models; a no-physics
      baseline with the same Gemini scene reading still scores higher, because the physics fit breaks on collisions and falls.</div></div>""", tmp / "e.png")
    ch = cv2.resize(chart, (1400, int(chart.shape[0] * 1400 / chart.shape[1])))
    im = bg.copy()
    y = 300
    im[y:y + ch.shape[0], 260:260 + ch.shape[1]] = ch[: H - y]
    hold(im, 7.0)

    # 6. end
    hold(card("""<div style="padding:220px 140px"><div class="d" style="font-size:150px;font-weight:800">Foresight</div>
      <div style="font-size:44px;margin-top:40px">github.com/amaan784/foresight-worlds</div>
      <div style="font-size:30px;color:var(--muted);margin-top:24px">Built during the Gemini 4 NYC Hackathon, November 21, 2026.</div></div>""", tmp / "f.png"), 3.5)

    out = ROOT / "docs" / "demo.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for f in seq:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    print(out, f"{len(seq) / FPS:.1f} s")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
