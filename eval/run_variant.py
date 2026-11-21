"""Run one pipeline variant on the 15 clips in parallel processes, then score it like run_eval.py.

  # one clip (run 15 of these in parallel):
  FORESIGHT_PERCEPTION_MODEL=gemini-4-carbon PYTHONPATH=backend:phase0 python eval/run_variant.py clip foresight_carbon <scene>
  # score the folder and fold it into results.json:
  PYTHONPATH=backend:phase0 python eval/run_variant.py score foresight_carbon
Fresh Gemini calls (no cache), same predict_clip() as the Foresight row; only the scene-reading model differs.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0"), str(ROOT / "eval")]

import run_eval as E  # noqa: E402


def clip(name: str, scene: str) -> None:
    from foresight import config
    from foresight.pipeline import predict_clip

    bench = E.pq.benchmark_dir("gcs")
    out_dir = E.PREDS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / E.pred_name(scene)
    meta_path = E.CACHE / f"variant_{name}" / f"{scene}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        real = bench / E.pq.needed_files(scene, E.VIEW, E.FPS)[1]
        m = predict_clip(E.cond_path(bench, scene, E.FPS), out, out_fps=E.FPS, out_size=(960, 540), dynamics="physics",
                         fit_video=E.cond_path(bench, scene, 30), cache_dir=None, reality_video=real,
                         work_dir=E.CACHE / f"variant_{name}" / "work" / scene, clip_id=scene)
        mm = m["metrics"]
        rec = {"ok": True, "latency_s": round(time.time() - t0, 2), "endpoint_error_cm": mm["endpoint_error_cm"]["foresight"],
               "fit_error_px": mm.get("fit_error_px"), "decision1_confidences": mm.get("decision1_confidences"),
               "repairs_used": mm.get("repairs_used"), "perception_model": config.PERCEPTION,
               "motion_model": (m.get("physics") or {}).get("motion_model")}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        E.static_prediction(bench, scene, E.VIEW, E.FPS, out_dir)
        rec = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400], "fallback": "static", "perception_model": config.PERCEPTION}
    meta_path.write_text(json.dumps(rec, indent=2))
    print(scene, json.dumps(rec)[:200], flush=True)


def score(name: str) -> None:
    bench = E.pq.benchmark_dir("gcs")
    scored = E.score_folder(E.PREDS / name, bench, E.FPS, E.RESULTS)
    per = {r["scenario"].removeprefix("trimmed-").removesuffix(".mp4"): round(r["final_arithmetic"] * 100, 2) for r in scored["per_scenario"]}
    runs = {sc: json.loads((E.CACHE / f"variant_{name}" / f"{sc}.json").read_text()) for sc in E.SCENES}
    ees = sorted(c["endpoint_error_cm"] for c in runs.values() if c.get("endpoint_error_cm") is not None)
    r = json.loads((E.EVAL / "results.json").read_text())
    r["systems"][name] = {"physics_iq_verified_pct": scored["physics_iq_verified_score_pct"],
                          "physics_iq_original_pct": scored["physics_iq_original_score_pct"],
                          "per_scenario_verified_pct": per, "n_ok": sum(c["ok"] for c in runs.values()),
                          "n_failed": sum(not c["ok"] for c in runs.values()),
                          "endpoint_error_cm_median": ees[len(ees) // 2] if ees else None,
                          "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "clean_run": True,
                          "perception_model": next(iter(runs.values())).get("perception_model")}
    r.setdefault("runs", {})[name] = runs
    (E.EVAL / "results.json").write_text(json.dumps(r, indent=2))
    (E.RESULTS / f"system_{name}.json").write_text(json.dumps(
        {k: r[k] for k in ("scenes", "gt_source", "gt_source_name", "view", "fps", "n_clips", "git_commit")}
        | {"systems": {name: r["systems"][name]}, "runs": {name: runs}}, indent=2))
    print(name, scored["physics_iq_verified_score_pct"])


if __name__ == "__main__":
    {"clip": lambda: clip(sys.argv[2], sys.argv[3]), "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
