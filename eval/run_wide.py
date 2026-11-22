"""Wider eval: 20 rigid-body Physics-IQ scenes x 3 camera views (60 clip-views), official metric code.

The 15-scene center-view eval (eval/run_eval.py) stays as it is; this run adds the left and right cameras and five
more scenes chosen by a rule fixed before any run (phase0/physiq_data.py, RIGID_EXTRA). Each view is scored with the
official functions on its own folder; the pooled score is the mean over all scene-view pairs, which is exactly the
official verified formula (final_score_view) over three views.

  PYTHONPATH=backend:phase0 python eval/run_wide.py --systems foresight const_velocity static [--workers 4]
  PYTHONPATH=backend:phase0 python eval/run_wide.py --score-only
Every Gemini output is fresh (clean run): a scene-view is read once by Hydrogen and shared by foresight and
const_velocity, so the only difference between those two is the dynamics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0"), str(ROOT / "eval")]

import run_eval as E  # noqa: E402

pq = E.pq
VIEWS = ["left", "center", "right"]
SCENES = pq.RIGID_SINGLE + pq.RIGID_MULTI + pq.RIGID_EXTRA
OUT = E.EVAL / "wide"
PREDS = E.EVAL / "preds_wide"
SYSTEMS = ["foresight", "const_velocity", "static", "omni", "genie"]


def pred_name(scene: str, view: str) -> str:
    id1, _ = pq.scene_ids(scene, view)
    return f"{id1}_perspective-{view}_trimmed-{scene}.mp4"


def cond_path(bench: Path, scene: str, view: str, fps: int) -> Path:
    return bench / pq.needed_files(scene, view, fps)[0]


def prompt(scene: str, view: str) -> str:
    import csv

    id1, _ = pq.scene_ids(scene, view)
    key = f"{id1}_perspective-{view}_take-1_trimmed-{scene}.mp4"
    with open(pq.PHYSIQ / "descriptions" / "best_practice" / "descriptions_base.csv") as f:
        for row in csv.DictReader(f):
            if row["scenario"] == key:
                return row["description"]
    raise KeyError(key)


def fetch(scenes: list[str]) -> Path:
    """Public original GT from the official bucket (the Verified set on Hugging Face is gated)."""
    bench = None
    for view in VIEWS:
        bench = pq.fetch(scenes, view, E.FPS, "gcs")
        pq.fetch_gcs([pq.needed_files(s, view, 30)[0] for s in scenes], bench)
    return bench


def one(system: str, scene: str, view: str, cache_root: str) -> dict:
    """One system on one scene-view; returns its run record (the video goes to preds_wide/<system>/<view>/)."""
    bench = pq.benchmark_dir("gcs")
    out_dir = PREDS / system / view
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / pred_name(scene, view)
    t0 = time.time()
    rec: dict = {"scene": scene, "view": view}
    try:
        if system == "static":
            E.static_prediction(bench, scene, view, E.FPS, out_dir)
        elif system in ("foresight", "const_velocity"):
            from foresight.pipeline import predict_clip

            real = bench / pq.needed_files(scene, view, E.FPS)[1]
            m = predict_clip(cond_path(bench, scene, view, E.FPS), out, out_fps=E.FPS, out_size=(960, 540),
                             dynamics="physics" if system == "foresight" else "const_velocity",
                             fit_video=cond_path(bench, scene, view, 30),
                             cache_dir=Path(cache_root) / "scene" / f"{scene}_{view}", reality_video=real,
                             work_dir=Path(cache_root) / "work" / system / f"{scene}_{view}", clip_id=f"{scene}_{view}")
            mm = m["metrics"]
            rec |= {"endpoint_error_cm": mm["endpoint_error_cm"]["foresight"], "fit_error_px": mm.get("fit_error_px"),
                    "repairs_used": mm.get("repairs_used"), "dynamics_used": mm.get("dynamics_used"),
                    "fallback_reason": mm.get("fallback_reason"), "motion_model": (m.get("physics") or {}).get("motion_model")}
        elif system == "omni":
            from foresight import omni

            work = Path(cache_root) / "omni_work" / f"{scene}_{view}"
            work.mkdir(parents=True, exist_ok=True)
            info = omni.continue_clip(cond_path(bench, scene, view, E.FPS), out, prompt=prompt(scene, view) + " Continue the scene.",
                                      fps=E.FPS, tag=f"wide:{scene}_{view}")
            rec |= {"omni_latency_s": info["latency_s"], "frames_padded": info["frames_padded"]}
        elif system == "genie":
            from foresight import genie

            info = genie.continue_from_frame(cond_path(bench, scene, view, E.FPS), out, prompt=prompt(scene, view),
                                             fps=E.FPS, tag=f"wide:{scene}_{view}")
            rec |= {"frames_padded": info["frames_padded"]}
        rec |= {"ok": True, "latency_s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        if "SpendCapHit" in type(e).__name__:
            raise
        traceback.print_exc()
        E.static_prediction(bench, scene, view, E.FPS, out_dir)  # keeps the pooled score over every scene-view
        rec |= {"ok": False, "error": f"{type(e).__name__}: {e}"[:400], "fallback": "static"}
    print(f"[{system}] {scene} {view} {json.dumps({k: rec.get(k) for k in ('ok', 'latency_s', 'endpoint_error_cm', 'dynamics_used')})}", flush=True)
    return rec


def score(system: str, scenes: list[str]) -> dict:
    bench = pq.benchmark_dir("gcs")
    per, views = {}, {}
    for view in VIEWS:
        d = PREDS / system / view
        if not d.exists() or not any(d.glob("*.mp4")):
            continue
        res_dir = OUT / "scores" / system / view
        masks = res_dir / "generated_masks" / view
        if masks.exists():  # masks are cached by file name; the predictions may have changed
            import shutil

            shutil.rmtree(masks)
        r = E.score_folder(d, bench, E.FPS, res_dir)
        views[view] = r["physics_iq_verified_score_pct"]
        for row in r["per_scenario"]:
            per[f"{row['scenario'].removeprefix('trimmed-').removesuffix('.mp4')}|{view}"] = round(row["final_arithmetic"] * 100, 2)
    vals = list(per.values())
    return {"pooled_verified_pct": round(sum(vals) / len(vals), 2) if vals else None, "per_view_pct": views,
            "per_scene_view_pct": per, "n_scene_views": len(vals)}


def write_outputs(results: dict) -> None:
    (E.EVAL / "results_wide.json").write_text(json.dumps(results, indent=2))
    systems = [s for s in SYSTEMS if s in results["systems"]]
    lines = [f"# Wider eval: {len(results['scenes'])} rigid-body Physics-IQ scenes x {len(VIEWS)} views", "",
             f"Generated by `eval/run_wide.py` at {results['generated_at']} (commit `{results['git_commit']}`), clean run: "
             f"**{results['clean_run']}**, ground truth: **physics-IQ-benchmark (original)**. Physics-IQ verified-formula "
             "score (%) per scene, averaged over the views run; higher is better. Best per row in bold.", "",
             "| Scene | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for sc in results["scenes"]:
        row = []
        for s in systems:
            v = [x for k, x in results["systems"][s]["per_scene_view_pct"].items() if k.split("|")[0] == sc]
            row.append(sum(v) / len(v) if v else None)
        best = max((v for v in row if v is not None), default=None)
        lines.append(f"| {sc} | " + " | ".join(("**%.1f**" % v if v == best else "%.1f" % v) if v is not None else "n/a" for v in row) + " |")
    pooled = [results["systems"][s]["pooled_verified_pct"] for s in systems]
    lines.append("| **pooled** | " + " | ".join(("**%.1f**" % v if v == max(pooled) else "%.1f" % v) for v in pooled) + " |")
    for s in systems:
        sv = results["systems"][s]
        extra = []
        if sv.get("endpoint_error_cm_median") is not None:
            extra.append(f"median endpoint error {sv['endpoint_error_cm_median']:.1f} cm")
        if sv.get("n_fallback"):
            extra.append(f"{sv['n_fallback']} scene-views fell back to constant velocity")
        if sv.get("n_failed"):
            extra.append(f"{sv['n_failed']} failed (scored as the static prediction)")
        lines.append(f"\n- {s}: per view " + ", ".join(f"{v} {x:.1f}" for v, x in sv["per_view_pct"].items()) +
                     (f"; {'; '.join(extra)}" if extra else ""))
    (E.EVAL / "results_wide_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["foresight", "const_velocity", "static"], choices=SYSTEMS)
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    path = E.EVAL / "results_wide.json"
    results = json.loads(path.read_text()) if path.exists() else {"systems": {}, "runs": {}}
    results |= {"scenes": a.scenes, "views": VIEWS, "gt_source": "gcs", "fps": E.FPS}
    if not a.score_only:
        fetch(a.scenes)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cache_root = str(E.CACHE / f"wide_{stamp}")
        for system in a.systems:  # foresight first: it caches the scene readings const_velocity reuses
            jobs = [(system, sc, v, cache_root) for sc in a.scenes for v in VIEWS]
            with ProcessPoolExecutor(max_workers=a.workers if system != "static" else 8) as ex:
                recs = list(ex.map(one, *zip(*jobs)))
            results["runs"][system] = {f"{r['scene']}|{r['view']}": r for r in recs}
            results["systems"][system] = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "clean_run": True}
            path.write_text(json.dumps(results, indent=2))
    for system in [s for s in SYSTEMS if s in results["runs"]]:
        runs = results["runs"][system]
        ees = sorted(r["endpoint_error_cm"] for r in runs.values() if r.get("endpoint_error_cm") is not None)
        results["systems"].setdefault(system, {}).update(score(system, a.scenes), **{
            "endpoint_error_cm_median": ees[len(ees) // 2] if ees else None,
            "n_failed": sum(not r["ok"] for r in runs.values()),
            "n_fallback": sum(r.get("dynamics_used") == "const_velocity_fallback" for r in runs.values())})
    results["clean_run"] = all(v.get("clean_run") for v in results["systems"].values())
    results["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    write_outputs(results)
    print(json.dumps({s: v.get("pooled_verified_pct") for s, v in results["systems"].items()}, indent=1))


if __name__ == "__main__":
    main()
