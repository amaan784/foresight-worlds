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
    ranked = sorted(have, key=lambda k: -s[k]["physics_iq_verified_pct"])
    parts = [f"{NAMES[k]} {s[k]['physics_iq_verified_pct']:.1f}" for k in ranked]
    lines += ["**In short:** Physics-IQ score (Verified formula, %, higher is better): " + ", ".join(parts) + ".", ""]
    runs = r.get("runs", {})
    if "foresight" in runs and "const_velocity" in runs:
        both = [sc for sc in r["scenes"] if runs["foresight"][sc].get("endpoint_error_cm") is not None
                and runs["const_velocity"][sc].get("endpoint_error_cm") is not None]
        closer = [sc for sc in both if runs["foresight"][sc]["endpoint_error_cm"] < runs["const_velocity"][sc]["endpoint_error_cm"]]
        blown = [sc for sc in r["scenes"] if (runs["foresight"][sc].get("endpoint_error_cm") or 0) > 1000]
        rr = runs["foresight"].get("ball-rolls-off", {}), runs["const_velocity"].get("ball-rolls-off", {})
        lines += [
            f"Foresight beats Gemini Omni by a wide margin. It does **not** yet beat the constant-velocity baseline, which "
            f"uses the same Gemini scene reading, tracking and renderer and only drops the physics. Where the physics fit "
            f"works it is more accurate: Foresight's object ends closer to the real one on {len(closer)} of the {len(both)} "
            f"clips with a measurable endpoint (on ball-rolls-off, {rr[0].get('endpoint_error_cm')} cm against "
            f"{rr[1].get('endpoint_error_cm')} cm). But the fit blew up on {len(blown)} clips with collisions or falls "
            f"({', '.join(blown)}), and those pull the pooled score down. Making contacts and falls robust is the next step.", ""]
    if "foresight" not in s:
        lines += ["Foresight itself has not been scored in this run yet; the rows below are the baselines.", ""]
    lines += ['<p align="center"><img src="eval/chart.png" alt="Physics-IQ score per system" width="88%"></p>', "",
              "| System | Physics-IQ score (Verified formula) | Original formula | Median endpoint error (cm) | Failed clips |",
              "|---|---|---|---|---|"]
    best = max(s[k]["physics_iq_verified_pct"] for k in have)
    for k in ranked:
        v = s[k]
        sc = f"{v['physics_iq_verified_pct']:.1f}"
        lines.append(f"| {NAMES[k]} | {'**' + sc + '**' if v['physics_iq_verified_pct'] == best else sc} | {v['physics_iq_original_pct']:.1f} | "
                     f"{f1(v.get('endpoint_error_cm_median'))} | {v['n_failed']} of {r['n_clips']} |")
    lines += ["", "Endpoint error is measured only where the system tracks an object (Foresight and constant velocity). "
              "Failures, causes and fixes: [eval/FAILURES.md](eval/FAILURES.md).", END]
    return "\n".join(lines)


F_START = "<!-- failures: start (written by eval/readme_results.py; edits here are lost) -->"
F_END = "<!-- failures: end -->"


def failures_section(r: dict) -> str:
    rows = []
    for system, clips in r.get("runs", {}).items():
        for scene, c in clips.items():
            if not c.get("ok", True):
                rows.append(f"| {system} | {scene} | crashed: {c.get('error', '')[:140]} | scored with the held last frame |")
    for scene, c in r.get("runs", {}).get("foresight", {}).items():
        ee, rep = c.get("endpoint_error_cm"), c.get("repairs_used") or 0
        if ee is not None and ee > 1000:
            rows.append(f"| foresight | {scene} | the fit blew up: endpoint {ee / 100:.1f} m off, {rep} repairs | not fixed today: contacts and falls in the fitter |")
        elif rep >= 2:
            rows.append(f"| foresight | {scene} | Decision 1 doubted the fit twice; {rep} Carbon repairs, {c.get('latency_s')} s | not fixed today |")
    body = ("No system failed on any clip." if not rows else
            "| System | Clip | What happened | Handling |\n|---|---|---|---|\n" + "\n".join(rows))
    return f"{F_START}\n{body}\n{F_END}"


def main() -> None:
    r = json.loads((ROOT / "eval" / "results.json").read_text())
    fpath = ROOT / "eval" / "FAILURES.md"
    ft = fpath.read_text()
    fa, fb = ft.index(F_START), ft.index(F_END) + len(F_END)
    fpath.write_text(ft[:fa] + failures_section(r) + ft[fb:])
    readme = ROOT / "README.md"
    text = readme.read_text()
    a, b = text.index(START), text.index(END) + len(END)
    readme.write_text(text[:a] + section(r) + text[b:])
    print("README results section updated")


if __name__ == "__main__":
    main()
