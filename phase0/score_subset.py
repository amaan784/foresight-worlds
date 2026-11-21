"""Score a SUBSET of Physics-IQ (single perspective) with the official metric functions.

run_physics_iq.py hard-requires all 198 generated videos (validate_generations asserts
num_videos == 198) and process_videos() raises unless all 3 perspectives exist per scenario.
So we import the official pieces directly:
  - binary_mask_generator.generate_mask   (masks for our generated videos)
  - calculate_and_write_metrics_to_csv.process_view  (per-view IoU / MSE vs take-1, take-2)
  - calculate_iq_score_stable.IQTable     (variance-normalised scores; restricted to one view)

Prediction files must use the benchmark naming: {id}_perspective-{view}_trimmed-{scene}.mp4
and be exactly 5.000 s (+-0.001) at the benchmark FPS.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "physics-IQ-benchmark"))
sys.path.insert(0, str(ROOT / "phase0"))

import physiq.calculate_iq_score_stable as iq_stable  # noqa: E402
from physiq.binary_mask_generator import generate_mask  # noqa: E402
from physiq.calculate_and_write_metrics_to_csv import ViewPaths, process_view  # noqa: E402

from physiq_data import scene_ids  # noqa: E402
from vidio import probe  # noqa: E402


def one_view_table(df: pd.DataFrame, view: str) -> iq_stable.IQTable:
    # IQTable averages over the module-level VIEWS in two helpers; restrict both to our view.
    iq_stable.VIEWS = [view]

    class OneViewIQTable(iq_stable.IQTable):
        views = [view]

    return OneViewIQTable(df)


def score_folder(pred_dir: Path, bench_dir: Path, fps: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "generated_masks" / pred_dir.name
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows, checks, views = [], [], set()
    for f in sorted(pred_dir.glob("*.mp4")):
        id1, persp, scen = f.name.split("_")
        scene = scen[len("trimmed-"):-len(".mp4")]
        view = persp.split("-")[1]
        views.add(persp)
        _, id2 = scene_ids(scene, view)
        pr = probe(f)
        ok = math.isclose(pr["duration"], 5.0, abs_tol=0.001) and round(pr["fps"]) == fps
        checks.append({"file": f.name, **pr, "valid_5s": ok})
        if not ok:
            raise ValueError(f"{f.name}: duration {pr['duration']} fps {pr['fps']} (need 5.000s @ {fps})")
        gen_mask = mask_dir / f"{id1}_video-masks_{fps}FPS_{persp}_take-1_{scen}"
        if not gen_mask.exists():
            generate_mask(str(f), str(mask_dir / f.name), is_real=False)
        real = bench_dir / "split-videos" / "testing" / f"{fps}FPS"
        masks = bench_dir / "video-masks" / "real" / f"{fps}FPS"
        paths = ViewPaths(
            real_v1=str(real / f"{id1}_testing-videos_{fps}FPS_{persp}_take-1_{scen}"),
            real_v2=str(real / f"{id2}_testing-videos_{fps}FPS_{persp}_take-2_{scen}"),
            generated=str(f),
            mask_v1=str(masks / f"{id1}_video-masks_{fps}FPS_{persp}_take-1_{scen}"),
            mask_v2=str(masks / f"{id2}_video-masks_{fps}FPS_{persp}_take-2_{scen}"),
            mask_generated=str(gen_mask),
        )
        res = process_view(paths, persp, 0, fps * 5, fps * 5)
        rows.append({"scenario": scen, **res})
    if len(views) != 1:
        raise ValueError(f"expected exactly one perspective, got {views}")
    view = views.pop()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{pred_dir.name}.csv", index=False)

    table = one_view_table(df, view)
    pooled = {k: float(v) for k, v in table.get_output_dict().items()}
    per = iq_stable.clip(table.compute_scores_scenario_by_view())
    per.insert(0, "scenario", df["scenario"].values)
    per_rows = per.to_dict(orient="records")
    result = {
        "run": pred_dir.name,
        "view": view,
        "fps": fps,
        "n_scenarios": len(df),
        "physics_iq_verified_score_pct": round(pooled["final_score_view"] * 100, 2),
        "physics_iq_original_score_pct": round(pooled["final_score_orig"] * 100, 2),
        "pooled": pooled,
        "per_scenario": [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in per_rows],
        "input_checks": checks,
    }
    (out_dir / f"{pred_dir.name}_metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir", type=Path)
    ap.add_argument("--bench", type=Path, default=ROOT / "data" / "physics-IQ-benchmark-verified")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", type=Path, default=ROOT / "phase0" / "out" / "scores")
    a = ap.parse_args()
    r = score_folder(a.pred_dir, a.bench, a.fps, a.out)
    print(json.dumps({k: r[k] for k in ["run", "view", "n_scenarios", "physics_iq_verified_score_pct", "physics_iq_original_score_pct"]}, indent=1))
    for row in r["per_scenario"]:
        print(f"  {row['scenario']:40s} verified-per-scenario={row['final_arithmetic']*100:6.2f}")
