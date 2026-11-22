"""Rerun only the dynamics on cached scenes: fit -> simulate -> render -> official score. No Gemini calls.

Scene readings and tracks come from an earlier eval run (eval/cache/run_*/work/foresight/<clip>/scene.json), so a
change to the fitter or simulator can be scored on all 15 clips in a couple of minutes for free (--perception DIR
is the exception: clips with no reading in DIR get a live one). Decision 1 /
Carbon are skipped here (verify=False): this measures the first fit only. Reported numbers still come from a clean
run of eval/run_eval.py.

  PYTHONPATH=backend:phase0 python eval/offline_fit.py --tag dev [--rebuild] [--scenes roll-in-box ...] [--workers 4]
  --rebuild re-runs tracking and scene assembly too (still on the cached Gemini scene reading).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0"), str(ROOT / "eval")]

import run_eval as E  # noqa: E402

OFFLINE = E.EVAL / "cache" / "offline"


def latest_run() -> Path:
    runs = sorted(p for p in E.CACHE.glob("run_*") if (p / "work" / "foresight").is_dir())
    if not runs:
        raise SystemExit("no eval/cache/run_*/work/foresight: run eval/run_eval.py --clean once first")
    return runs[-1]


def one(scene_name: str, run: str, tag: str, dynamics: str, rebuild: bool, perception: str | None) -> dict:
    from foresight.pipeline import const_velocity_trajectory, endpoint_error, track_in_video
    from foresight.scene import build_scene
    from physics import predict
    from physics.render import render_trajectory

    work = OFFLINE / tag / "work" / scene_name
    work.mkdir(parents=True, exist_ok=True)
    bench = E.pq.benchmark_dir("gcs")
    if rebuild:  # tracking + scene assembly again, on the cached Gemini scene reading
        cache = Path(perception) / scene_name if perception else Path(run) / "scene" / scene_name
        scene, _ = build_scene(E.cond_path(bench, scene_name, E.FPS), work, E.cond_path(bench, scene_name, 30),
                               cache, E.FPS, None, scene_name)
    else:
        scene = json.loads((Path(run) / "work" / "foresight" / scene_name / "scene.json").read_text())
    scene["work_dir"] = str(work)
    out = OFFLINE / tag / "preds" / E.pred_name(scene_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec: dict = {"scene": scene_name}
    try:
        if dynamics == "physics":
            res = predict(scene, verify=False, out_size=(960, 540))
            Path(res["prediction_mp4"]).replace(out)
            traj = res["trajectory"]
            rec |= {"fit_error_px": res["fit_error_px"], "motion_model": res["motion_model"],
                    "dynamics_used": res.get("dynamics_used", "physics"), "fallback_reason": res.get("fallback_reason")}
        else:
            traj = const_velocity_trajectory(scene)
            render_trajectory(scene, traj, out, out_size=(960, 540))
        rec["latency_s"] = round(time.time() - t0, 2)
        real = bench / E.pq.needed_files(scene_name, E.VIEW, E.FPS)[1]
        main = max(scene["objects"], key=lambda ob: len(ob["track"]))["id"]
        ee = endpoint_error(traj, track_in_video(real, scene), scene, main)
        rec["endpoint_error_cm"] = ee["cm"] if ee else None
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        rec |= {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="output folder name under eval/cache/offline/")
    ap.add_argument("--scenes", nargs="+", default=E.SCENES)
    ap.add_argument("--run", default=None, help="eval/cache/run_* folder with the cached scenes (default: latest)")
    ap.add_argument("--dynamics", default="physics", choices=["physics", "const_velocity"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="re-track and rebuild scenes from the cached perception")
    ap.add_argument("--perception", default=None, help="with --rebuild: folder of <clip>/perception.json to use; a clip "
                    "missing there gets a LIVE Gemini scene reading, cached there (the only Gemini call this script makes)")
    a = ap.parse_args()
    run = Path(a.run) if a.run else latest_run()
    preds = OFFLINE / a.tag / "preds"
    if preds.exists():
        for f in preds.glob("*.mp4"):
            f.unlink()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(one, a.scenes, [str(run)] * len(a.scenes), [a.tag] * len(a.scenes), [a.dynamics] * len(a.scenes),
                           [a.rebuild] * len(a.scenes), [a.perception] * len(a.scenes)))
    bench = E.pq.benchmark_dir("gcs")
    for rec in recs:  # a failed clip gets the static prediction, as in run_eval.py
        if not rec["ok"]:
            E.static_prediction(bench, rec["scene"], E.VIEW, E.FPS, preds)
    summary = {"run": str(run), "tag": a.tag, "dynamics": a.dynamics, "clips": {r["scene"]: r for r in recs}}
    if not a.no_score:
        res_dir = OFFLINE / a.tag / "results"
        if (res_dir / "generated_masks").exists():  # masks are cached by file name; predictions changed
            import shutil

            shutil.rmtree(res_dir / "generated_masks")
        scored = E.score_folder(preds, bench, E.FPS, res_dir)
        per = {r["scenario"].removeprefix("trimmed-").removesuffix(".mp4"): round(r["final_arithmetic"] * 100, 1)
               for r in scored["per_scenario"]}
        summary["score_pct"] = scored["physics_iq_verified_score_pct"]
        for sc, v in per.items():
            summary["clips"][sc]["score_pct"] = v
        ref = json.loads((E.EVAL / "results.json").read_text())["systems"]
        print(f"{'clip':22s} {'score':>6s} {'eval F':>7s} {'eval CV':>7s} {'end cm':>9s} {'fit px':>7s}  notes")
        for sc in a.scenes:
            c = summary["clips"][sc]
            note = c.get("fallback_reason") or c.get("error") or ""
            print(f"{sc:22s} {c.get('score_pct', float('nan')):6.1f} {ref['foresight']['per_scenario_verified_pct'].get(sc, float('nan')):7.1f} "
                  f"{ref['const_velocity']['per_scenario_verified_pct'].get(sc, float('nan')):7.1f} "
                  f"{c.get('endpoint_error_cm') if c.get('endpoint_error_cm') is not None else 'n/a':>9} "
                  f"{c.get('fit_error_px') if c.get('fit_error_px') is not None else '':>7}  {note}")
        print(f"pooled score {summary['score_pct']}  (eval: Foresight {ref['foresight']['physics_iq_verified_pct']}, "
              f"constant velocity {ref['const_velocity']['physics_iq_verified_pct']})")
    (OFFLINE / a.tag / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
