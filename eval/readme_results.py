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
         "genie": "Genie 4 (world model)", "foresight_carbon": "Foresight with Gemini 4 Carbon reading the scene",
         "foresight_neon": "Foresight with Gemini 4 Neon reading the scene",
         "const_velocity": "Constant velocity (no physics)", "static": "Last frame held"}
ORDER = ["foresight", "foresight_carbon", "foresight_neon", "omni", "genie", "const_velocity", "static"]


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
        fell = [sc for sc in r["scenes"] if runs["foresight"][sc].get("dynamics_used") == "const_velocity_fallback"]
        fh, cv = s["foresight"], s["const_velocity"]
        videos = [k for k in ("omni", "genie") if k in s]
        beats_videos = all(fh["physics_iq_verified_pct"] > s[k]["physics_iq_verified_pct"] for k in videos)
        text = (f"On the Physics-IQ score Foresight ({fh['physics_iq_verified_pct']:.1f}) "
                + (f"is above both video models (" if beats_videos else "is compared with the video models (")
                + ", ".join(f"{NAMES[k].split(' (')[0]} {s[k]['physics_iq_verified_pct']:.1f}" for k in videos) + ")"
                + (f" and above the constant-velocity baseline ({cv['physics_iq_verified_pct']:.1f}). "
                   if fh["physics_iq_verified_pct"] > cv["physics_iq_verified_pct"] else
                   f", and still **below** the constant-velocity baseline ({cv['physics_iq_verified_pct']:.1f}), which uses the "
                   f"same Gemini scene reading, tracking and renderer and only drops the physics. "))
        text += (f"On where the object actually ends up, Foresight is closer to reality than constant velocity on "
                 f"{len(closer)} of the {len(both)} clips with a measurable endpoint (median "
                 f"{f1(fh.get('endpoint_error_cm_median'))} cm against {f1(cv.get('endpoint_error_cm_median'))} cm). ")
        if fh["physics_iq_verified_pct"] <= cv["physics_iq_verified_pct"]:
            text += ("The two measures disagree because the Physics-IQ score compares where pixels change: a constant-velocity "
                     "object that leaves the frame early leaves a still picture, which matches a real scene that has come to "
                     "rest, even when the object went the wrong way. ")
        text += (f"The physics fit blew up (endpoint over 10 m off) on {len(blown)} clip{'s' if len(blown) != 1 else ''}"
                 + (f" ({', '.join(blown)})" if blown else "")
                 + f"; the energy check fell back to constant velocity on {len(fell)}"
                 + (f" ({', '.join(fell)})." if fell else "."))
        lines += [text, ""]
        for key, model in (("foresight_carbon", "Gemini 4 Carbon"), ("foresight_neon", "Gemini 4 Neon")):
            if key in s:
                fc = s[key]
                lines += [f"Reading the scene with {model} instead of Hydrogen: {fc['physics_iq_verified_pct']:.1f} against "
                          f"{fh['physics_iq_verified_pct']:.1f}, median endpoint error {f1(fc.get('endpoint_error_cm_median'))} cm "
                          f"against {f1(fh.get('endpoint_error_cm_median'))} cm.", ""]
        if "foresight_carbon" in s or "foresight_neon" in s:
            lines += ["The stage reads scenes with Hydrogen because it is the fastest; Neon takes over automatically if "
                      "Hydrogen keeps failing.", ""]
    if "foresight" not in s:
        lines += ["Foresight itself has not been scored in this run yet; the rows below are the baselines.", ""]
    lines += ['<p align="center"><img src="eval/chart.png" alt="Physics-IQ score per system" width="88%"></p>', "",
              "| System | Physics-IQ score (Verified formula) | Original formula | Median endpoint error (cm) | Failed clips | Run |",
              "|---|---|---|---|---|---|"]
    best = max(s[k]["physics_iq_verified_pct"] for k in have)
    for k in ranked:
        v = s[k]
        sc = f"{v['physics_iq_verified_pct']:.1f}"
        lines.append(f"| {NAMES[k]} | {'**' + sc + '**' if v['physics_iq_verified_pct'] == best else sc} | {v['physics_iq_original_pct']:.1f} | "
                     f"{f1(v.get('endpoint_error_cm_median'))} | {v['n_failed']} of {r['n_clips']} | {v.get('run_at', '')[:10]} |")
    lines += ["", "Endpoint error is measured only where the system tracks an object (Foresight and constant velocity). "
              "The video models' rows come from their own clean run (the Run column): their output does not depend on "
              "Foresight's code. Failures, causes and fixes: [eval/FAILURES.md](eval/FAILURES.md).", ""]
    wide = ROOT / "eval" / "results_wide.json"
    if wide.exists():
        w = json.loads(wide.read_text())
        ws = {k: v for k, v in w.get("systems", {}).items() if v.get("pooled_verified_pct") is not None}
        if ws:
            rk = sorted(ws, key=lambda k: -ws[k]["pooled_verified_pct"])
            lines += ["### Wider eval: 20 scenes, all three cameras", "",
                      f"`eval/run_wide.py` runs the same systems on {len(w['scenes'])} scenes x {len(w['views'])} camera views "
                      f"({ws[rk[0]]['n_scene_views']} scene-views; five scenes added by a rule fixed before the run). "
                      f"[eval/results_wide.json](eval/results_wide.json), [per-scene table](eval/results_wide_table.md). "
                      f"Clean run: {'yes' if w.get('clean_run') else 'no'}; commit `{w.get('git_commit')}`, {w.get('generated_at')}.", "",
                      "| System | Physics-IQ score (Verified formula, all views) | Left | Center | Right | Median endpoint error (cm) | Failed |",
                      "|---|---|---|---|---|---|---|"]
            for k in rk:
                v = ws[k]
                pv = v.get("per_view_pct", {})
                lines.append(f"| {NAMES[k]} | {v['pooled_verified_pct']:.1f} | {f1(pv.get('left'))} | {f1(pv.get('center'))} | "
                             f"{f1(pv.get('right'))} | {f1(v.get('endpoint_error_cm_median'))} | {v.get('n_failed', 0)} of {v['n_scene_views']} |")
            runs_w = w.get("runs", {})
            if "foresight" in runs_w and "const_velocity" in runs_w:
                F, C = runs_w["foresight"], runs_w["const_velocity"]
                both = [k for k in F if F[k].get("endpoint_error_cm") is not None and (C.get(k) or {}).get("endpoint_error_cm") is not None]
                closer = sum(F[k]["endpoint_error_cm"] < C[k]["endpoint_error_cm"] for k in both)
                fw, cw = ws.get("foresight", {}), ws.get("const_velocity", {})
                lines += ["", f"Across the {fw.get('n_scene_views')} scene-views Foresight scores "
                          f"{'above' if fw.get('pooled_verified_pct', 0) > cw.get('pooled_verified_pct', 0) else 'below'} constant velocity "
                          f"on the Physics-IQ score ({f1(fw.get('pooled_verified_pct'))} against {f1(cw.get('pooled_verified_pct'))}) and ends "
                          f"closer to the real object on {closer} of the {len(both)} scene-views with a measurable endpoint. Most of these "
                          f"scene-views (the left and right cameras and the five added scenes) were not used while developing."]
            lines.append("")
    lines.append(END)
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
        if c.get("dynamics_used") == "const_velocity_fallback":
            rows.append(f"| foresight | {scene} | the energy check rejected the simulation: {c.get('fallback_reason')} | constant velocity used and reported |")
        elif ee is not None and ee > 1000:
            rows.append(f"| foresight | {scene} | the fit blew up: endpoint {ee / 100:.1f} m off, {rep} repairs | open |")
        elif rep >= 2:
            rows.append(f"| foresight | {scene} | Decision 1 doubted the fit twice; {rep} Carbon repairs, {c.get('latency_s')} s | open |")
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
