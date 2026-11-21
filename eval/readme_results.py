"""Write the README's results section from eval/results.json (no number is typed by hand).

  python eval/readme_results.py
Replaces everything between the two RESULTS markers in README.md.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- results: start (written by eval/readme_results.py from eval/results.json; edits here are lost) -->"
END = "<!-- results: end -->"
NAMES = {"foresight": "Foresight (physics)", "omni": "Gemini Omni 1.2 Flash (video model)",
         "const_velocity": "Constant velocity (no physics)", "static": "Last frame held"}
ORDER = ["foresight", "omni", "const_velocity", "static"]


def f1(v) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def section(r: dict) -> str:
    s = r["systems"]
    have = [k for k in ORDER if k in s]
    gt = "Physics-IQ Verified" if r["gt_source"] == "hf" else "the original Physics-IQ ground truth (Verified files were not available)"
    lines = [START, "## Results", "",
             f"Every number below comes from `eval/run_eval.py`: [eval/results.json](eval/results.json), "
             f"[the per-clip table](eval/results_table.md), [the method](docs/evaluation.md). "
             f"{r['n_clips']} rigid-body clips, center camera, {r['fps']} FPS, scored against {gt} with the official "
             f"Physics-IQ code. Clean run: {'yes' if r['clean_run'] else 'no (development run with cached Gemini outputs)'}; "
             f"commit `{r['git_commit']}`, {r['generated_at']}.", ""]
    parts = [f"{NAMES[k].split(' (')[0]} {s[k]['physics_iq_verified_pct']:.1f}" for k in have]
    lines += ["**In short:** Physics-IQ score (Verified formula, %, higher is better): " + ", ".join(parts) + ".", ""]
    if "foresight" not in s:
        lines += ["Foresight itself has not been scored in this run yet; the rows below are the baselines.", ""]
    lines += ['<p align="center"><img src="eval/chart.png" alt="Physics-IQ score per system" width="88%"></p>', "",
              "| System | Physics-IQ score (Verified formula) | Original formula | Median endpoint error (cm) | Median time per clip (s) | Failed clips |",
              "|---|---|---|---|---|---|"]
    for k in have:
        v = s[k]
        lines.append(f"| {NAMES[k]} | **{v['physics_iq_verified_pct']:.1f}** | {v['physics_iq_original_pct']:.1f} | "
                     f"{f1(v.get('endpoint_error_cm_median'))} | {f1(v.get('latency_s_median'))} | {v['n_failed']} of {r['n_clips']} |")
    lines += ["", "Endpoint error is measured only where the system tracks an object (Foresight and constant velocity). "
              "Failures, causes and fixes: [eval/FAILURES.md](eval/FAILURES.md).", END]
    return "\n".join(lines)


def main() -> None:
    r = json.loads((ROOT / "eval" / "results.json").read_text())
    readme = ROOT / "README.md"
    text = readme.read_text()
    a, b = text.index(START), text.index(END) + len(END)
    readme.write_text(text[:a] + section(r) + text[b:])
    print("README results section updated")


if __name__ == "__main__":
    main()
