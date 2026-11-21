"""1-minute demo with a voiceover: each scene is timed to its narration (macOS `say`, voice Samantha).

  PYTHONPATH=backend:phase0 python docs/img/demo_voiceover.py outputs/<job_id>  ->  docs/demo_voiceover.mp4
Every number spoken comes from the run's result.json or eval/results.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "docs" / "img")]
from demo_video import main as render  # noqa: E402

VOICE, RATE, LEAD, TAIL = "Samantha", 186, 0.3, 0.45


def words(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 0.25 else f"{x:.1f}"


def script(job: Path) -> dict[str, str]:
    r = json.loads((job / "result.json").read_text())
    ev = [json.loads(line) for line in open(job / "events.jsonl")]
    ver = next(e["data"] for e in ev if e["event"] == "verify")
    S = json.loads((ROOT / "eval" / "results.json").read_text())["systems"]
    m, n = r["metrics"], json.loads((ROOT / "eval" / "results.json").read_text())["n_clips"]
    conf = round(ver["confidences"]["outline_on_object"] * 100)
    return {
        "title": "This is Foresight. Robots need to know where things will be in a few seconds. "
                 "Video models can show that, but they make up the physics.",
        "cond": "It watches three seconds of real video. The next five stay hidden.",
        "steps": "Gemini 4 Hydrogen reads the scene. We track the ball, measure the scene, and fit a physics world to it. "
                 f"Gemini Decision 1 checks the fit on the real frames, {conf} percent confident here, "
                 "and Gemini 4 Carbon repairs it when unsure.",
        "reveal": f"Then the physics runs forward. Foresight ends {words(m['endpoint_error_cm']['foresight'])} centimetres "
                  f"from the real ball. Gemini Omni, a video model, ends {words(m['endpoint_error_cm']['omni'])} centimetres away.",
        "chart": f"On {n} real Physics-IQ clips, scored with the official code, Foresight scores "
                 f"{words(S['foresight']['physics_iq_verified_pct'])}, ahead of Genie 4 at {words(S['genie']['physics_iq_verified_pct'])} "
                 f"and Gemini Omni at {words(S['omni']['physics_iq_verified_pct'])}. A straight-line baseline still scores higher, "
                 "because our physics breaks on collisions and falls. That is next.",
        "end": "Foresight. Watch three seconds, predict the next five.",
    }


def main(job: Path) -> Path:
    tmp = Path(tempfile.mkdtemp())
    lines = script(job)
    clips, dur = {}, {}
    for k, text in lines.items():
        aiff = tmp / f"{k}.aiff"
        subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text], check=True)
        d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(aiff)],
                                 capture_output=True, text=True).stdout)
        clips[k], dur[k] = aiff, d + LEAD + TAIL
    silent = tmp / "silent.mp4"
    _, starts = render(job, dur=dur, out=silent)
    # place each narration at its scene start (+ lead) and mix
    keys = list(lines)
    inputs, filters = [], []
    for i, k in enumerate(keys):
        inputs += ["-i", str(clips[k])]
        ms = int((starts[i] + LEAD) * 1000)
        filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(keys))) + f"amix=inputs={len(keys)}:normalize=0[aout]"
    out = ROOT / "docs" / "demo_voiceover.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), *inputs, "-filter_complex", ";".join(filters + [mix]),
                    "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], check=True)
    (ROOT / "docs" / "demo_voiceover_script.txt").write_text("\n\n".join(lines.values()) + "\n")
    print(out, {k: round(v, 1) for k, v in dur.items()}, "total", round(sum(dur.values()) + 0, 1))
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]))
