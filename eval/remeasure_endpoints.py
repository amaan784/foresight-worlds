"""Re-measure endpoint errors of finished eval runs with the current reality tracker. No model calls.

Each Foresight / constant-velocity run saved its scene (scene.json) and its predicted path (predict_meta.json), so
the real object can be tracked again and the endpoint error recomputed after a change to how the real track ends
(Nov 22: it now ends when the object goes behind something standing in front of it).

  PYTHONPATH=backend:phase0 python eval/remeasure_endpoints.py main   # eval/results.json, latest clean run_*
  PYTHONPATH=backend:phase0 python eval/remeasure_endpoints.py wide   # eval/results_wide.json, latest wide_*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0"), str(ROOT / "eval")]

import run_eval as E  # noqa: E402
from foresight.pipeline import endpoint_error, track_in_video  # noqa: E402

SYSTEMS = ["foresight", "const_velocity"]


def remeasure(work: Path, scene_name: str, view: str) -> float | None:
    meta = json.loads((work / "predict_meta.json").read_text())
    scene = json.loads((work / "scene.json").read_text())
    main = max(scene["objects"], key=lambda ob: len(ob["track"]))["id"]
    tr = meta["trajectory"]
    traj = [{"t_s": t, "objects": {main: {"x": x, "y": y}}} for t, x, y in zip(tr["t_s"], tr["x"], tr["y"]) if x is not None]
    real = E.pq.benchmark_dir("gcs") / E.pq.needed_files(scene_name, view, E.FPS)[1]
    ee = endpoint_error(traj, track_in_video(real, scene), scene, main)
    return ee["cm"] if ee else None


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    if which == "main":
        path = E.EVAL / "results.json"
        r = json.loads(path.read_text())
        cache = sorted(p for p in E.CACHE.glob("run_*") if (p / "work" / "foresight").is_dir())[-1]
        keys = [(sc, sc, "center") for sc in r["scenes"]]
        # scene-reading variants (eval/run_variant.py) keep their own work folders
        workdir = lambda system, key: (E.CACHE / f"variant_{system}" / "work" / key if system.startswith("foresight_")  # noqa: E731
                                       else cache / "work" / system / key)
    else:
        path = E.EVAL / "results_wide.json"
        r = json.loads(path.read_text())
        cache = sorted(p for p in E.CACHE.glob("wide_*") if (p / "work").is_dir())[-1]
        keys = [(f"{sc}|{v}", sc, v) for sc in r["scenes"] for v in r["views"]]
        workdir = lambda system, key: cache / "work" / system / key.replace("|", "_")  # noqa: E731
    print("cache", cache)
    systems = SYSTEMS + ([k for k in r["systems"] if k.startswith("foresight_")] if which == "main" else [])
    for system in systems:
        runs = r["runs"].get(system, {})
        changed = 0
        for key, sc, view in keys:
            w = workdir(system, key)
            if key not in runs or not (w / "predict_meta.json").exists():
                continue
            old = runs[key].get("endpoint_error_cm")
            new = remeasure(w, sc, view)
            if new != old:
                changed += 1
                print(f"  {system} {key}: {old} -> {new}")
            runs[key]["endpoint_error_cm"] = new
        ees = sorted(c["endpoint_error_cm"] for c in runs.values() if c.get("endpoint_error_cm") is not None)
        r["systems"][system]["endpoint_error_cm_median"] = ees[len(ees) // 2] if ees else None
        r["systems"][system]["endpoint_rule"] = "real track ends at the frame edge, at loss, or behind an object in front (Nov 22)"
        print(f"{system}: {changed} changed, median {r['systems'][system]['endpoint_error_cm_median']}")
    path.write_text(json.dumps(r, indent=2))
    blk = E.RESULTS
    if which == "main":
        for system in systems:
            f = blk / f"system_{system}.json"
            if f.exists():
                b = json.loads(f.read_text())
                b["systems"][system] = r["systems"][system]
                b["runs"][system] = r["runs"][system]
                f.write_text(json.dumps(b, indent=2))


if __name__ == "__main__":
    main()
