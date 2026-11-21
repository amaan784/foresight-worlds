"""Phase 2 eval: 15 rigid-body Physics-IQ clips (center view, 24 FPS), four systems, official metric code.

  PYTHONPATH=backend:phase0 python eval/run_eval.py --systems static const_velocity foresight omni [--clean]

Systems
  foresight       predict_clip(dynamics="physics")        perception + tracking + physics fit + render
  const_velocity  predict_clip(dynamics="const_velocity") same perception/tracking/renderer, no physics
  omni            gemini-omni-1.2-flash continues the 3 s clip (official base prompt + "Continue the scene.")
  static          last conditioning frame held for 5 s

A system that fails on a clip gets the static prediction for that clip (so pooled scores always cover all
clips) and the failure is listed in eval/FAILURES.md.
--clean: no reuse of earlier Gemini outputs; perception is computed once per clip in this run and shared by
foresight and const_velocity (so the only difference between them is the dynamics).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0")]

import physiq_data as pq  # noqa: E402
from score_subset import score_folder  # noqa: E402
from static_pred import static_prediction  # noqa: E402

EVAL = ROOT / "eval"
PREDS, RESULTS, CACHE = EVAL / "preds", EVAL / "results", EVAL / "cache"
SCENES = pq.RIGID_SINGLE + pq.RIGID_MULTI
VIEW, FPS = "center", 24
SYSTEMS = ["foresight", "omni", "const_velocity", "static"]


def gt_source(pref: str) -> str:
    if pref in ("hf", "gcs"):
        return pref
    try:  # Verified needs the HF gate accepted; fall back to the public original GT
        from huggingface_hub import hf_hub_download

        hf_hub_download(pq.HF_REPO, pq.needed_files("ball-rolls-off", VIEW, FPS)[-1], repo_type="dataset",
                        local_dir=pq.benchmark_dir("hf"))
        return "hf"
    except Exception:  # noqa: BLE001
        return "gcs"


def fetch(scenes: list[str], src: str) -> Path:
    bench = pq.fetch(scenes, VIEW, FPS, src)
    # 30 FPS conditioning only (fitting uses real timestamps; the 24 FPS files are frame-dropped)
    paths = [pq.needed_files(s, VIEW, 30)[0] for s in scenes]
    (pq.fetch_hf if src == "hf" else pq.fetch_gcs)(paths, bench)
    return bench


def cond_path(bench: Path, scene: str, fps: int) -> Path:
    return bench / pq.needed_files(scene, VIEW, fps)[0]


def pred_name(scene: str) -> str:
    id1, _ = pq.scene_ids(scene, VIEW)
    return f"{id1}_perspective-{VIEW}_trimmed-{scene}.mp4"


def benchmark_prompt(scene: str) -> str:
    import csv

    id1, _ = pq.scene_ids(scene, VIEW)
    key = f"{id1}_perspective-{VIEW}_take-1_trimmed-{scene}.mp4"
    with open(pq.PHYSIQ / "descriptions" / "best_practice" / "descriptions_base.csv") as f:
        for row in csv.DictReader(f):
            if row["scenario"] == key:
                return row["description"] + " Continue the scene."
    raise KeyError(key)


def run_system(system: str, scene: str, bench: Path, out: Path, cache_root: Path, omni_pace_s: float) -> dict:
    t0 = time.time()
    if system == "static":
        tmp = static_prediction(bench, scene, VIEW, FPS, out.parent)
        if tmp != out:
            shutil.move(tmp, out)
        return {"latency_s": round(time.time() - t0, 2)}
    if system in ("foresight", "const_velocity"):
        from foresight.pipeline import predict_clip

        real = bench / pq.needed_files(scene, VIEW, FPS)[1]
        meta = predict_clip(cond_path(bench, scene, FPS), out, out_fps=FPS, out_size=(960, 540),
                            dynamics="physics" if system == "foresight" else "const_velocity",
                            fit_video=cond_path(bench, scene, 30), cache_dir=cache_root / "scene" / scene,
                            reality_video=real, work_dir=cache_root / "work" / system / scene, clip_id=scene)
        m = meta["metrics"]
        return {"latency_s": round(time.time() - t0, 2), "endpoint_error_cm": m["endpoint_error_cm"]["foresight"],
                "fit_error_px": m.get("fit_error_px"), "decision1_confidences": m.get("decision1_confidences"),
                "repairs_used": m.get("repairs_used"), "renderer": meta["renderer"],
                "motion_model": (meta.get("physics") or {}).get("motion_model")}
    if system == "omni":
        from foresight import omni

        cached = cache_root / "omni" / f"{scene}.mp4"
        if cached.exists():
            shutil.copyfile(cached, out)
            return {"latency_s": 0.0, "cached": True}
        work = cache_root / "omni_work" / scene
        work.mkdir(parents=True, exist_ok=True)
        info = omni.continue_clip(cond_path(bench, scene, FPS), work / "omni_5s.mp4", prompt=benchmark_prompt(scene),
                                  fps=FPS, tag=f"eval:{scene}")
        shutil.copyfile(work / "omni_5s.mp4", out)
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, cached)
        time.sleep(max(0.0, omni_pace_s - (time.time() - t0)))  # pace under the shared spend cap
        return {"latency_s": info["latency_s"], "omni_raw_duration_s": info["raw_duration_s"],
                "frames_padded": info["frames_padded"]}
    raise ValueError(system)


def chart(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"foresight": "#2350d8", "const_velocity": "#7d8597", "omni": "#b86e00", "static": "#c4c6cc"}
    labels = {"foresight": "Foresight\n(physics)", "const_velocity": "Constant velocity\n(no physics)",
              "omni": "Gemini Omni 1.2\n(video model)", "static": "Last frame\nheld"}
    systems = [s for s in SYSTEMS if s in results["systems"]]
    vals = [results["systems"][s]["physics_iq_verified_pct"] for s in systems]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    bars = ax.bar([labels[s] for s in systems], vals, color=[colors[s] for s in systems], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(60, max(vals) + 12))
    ax.set_ylabel("Physics-IQ score (Verified formula, %)", fontsize=12)
    gt = "Physics-IQ Verified GT" if results["gt_source"] == "hf" else "original Physics-IQ GT"
    ax.set_title(f"{results['n_clips']} rigid-body side-view clips, official metric code, {gt}", fontsize=12, loc="left", color="#5f6470")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table(results: dict, path: Path) -> None:
    systems = [s for s in SYSTEMS if s in results["systems"]]
    lines = [f"# Results: {results['n_clips']} rigid-body Physics-IQ clips, center view, {FPS} FPS", "",
             f"Generated by `eval/run_eval.py` at {results['generated_at']} (commit `{results['git_commit']}`), "
             f"clean run: **{results['clean_run']}**, ground truth: **{results['gt_source_name']}**. "
             "Scores are the Physics-IQ Verified-formula score (%) from the official metric code; higher is better. "
             "Best per row in bold.", "",
             "| Scene | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for sc in results["scenes"]:
        row = [results["systems"][s]["per_scenario_verified_pct"].get(sc) for s in systems]
        best = max(v for v in row if v is not None)
        cells = [("**%.1f**" % v if v == best else "%.1f" % v) if v is not None else "n/a" for v in row]
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    pooled = [results["systems"][s]["physics_iq_verified_pct"] for s in systems]
    lines.append("| **pooled** | " + " | ".join(("**%.1f**" % v if v == max(pooled) else "%.1f" % v) for v in pooled) + " |")
    ee = [results["systems"][s].get("endpoint_error_cm_median") for s in systems]
    lines += ["", "Median endpoint error (cm, last frame the real object is visible; lower is better): " +
              ", ".join(f"{s} {v:.1f}" if v is not None else f"{s} n/a" for s, v in zip(systems, ee)) + "."]
    path.write_text("\n".join(lines) + "\n")


def merge_write(results: dict, system: str | None) -> dict:
    """Merge this run's systems into eval/results.json as it is on disk now (another run may be writing too)."""
    path = EVAL / "results.json"
    if system is not None:  # also keep this system's block on its own, for --merge-only
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"system_{system}.json").write_text(json.dumps(
            {k: results[k] for k in ("scenes", "gt_source", "gt_source_name", "view", "fps", "n_clips", "git_commit")}
            | {"systems": {system: results["systems"][system]}, "runs": {system: results["runs"][system]}}, indent=2))
    disk = json.loads(path.read_text()) if path.exists() else {}
    if disk.get("scenes") == results["scenes"] and disk.get("gt_source") == results["gt_source"]:
        for key in ("systems", "runs"):
            merged = dict(disk.get(key, {}))
            merged.update({k: v for k, v in results[key].items() if system is None or k == system})
            results[key] = {**results[key], **merged} if system is None else merged
    results["clean_run"] = all(v.get("clean_run") for v in results["systems"].values())
    results["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--gt", default="auto", choices=["auto", "hf", "gcs"])
    ap.add_argument("--clean", action="store_true", help="fresh Gemini outputs for this run (reported numbers)")
    ap.add_argument("--omni-pace", type=float, default=90.0, help="minimum seconds per Omni call (sequential mode)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent clips for the omni system")
    ap.add_argument("--merge-only", action="store_true", help="fold eval/results/system_*.json into results.json, redraw table + chart")
    a = ap.parse_args()
    if a.merge_only:
        r = json.loads((EVAL / "results.json").read_text())
        for f in sorted(RESULTS.glob("system_*.json")):
            blk = json.loads(f.read_text())
            if blk["scenes"] == r["scenes"] and blk["gt_source"] == r["gt_source"]:
                r["systems"].update(blk["systems"])
                r.setdefault("runs", {}).update(blk["runs"])
        r["clean_run"] = all(v.get("clean_run") for v in r["systems"].values())
        (EVAL / "results.json").write_text(json.dumps(r, indent=2))
        table(r, EVAL / "results_table.md")
        chart(r, EVAL / "chart.png")
        print(json.dumps({k: v["physics_iq_verified_pct"] for k, v in r["systems"].items()}, indent=1))
        return

    src = gt_source(a.gt)
    bench = fetch(a.scenes, src)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_root = CACHE / (f"run_{stamp}" if a.clean else "dev")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    prev = json.loads((EVAL / "results.json").read_text()) if (EVAL / "results.json").exists() else {}
    results = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit,
               "clean_run": a.clean, "gt_source": src,
               "gt_source_name": "physics-IQ-benchmark-verified" if src == "hf" else "physics-IQ-benchmark (original)",
               "view": VIEW, "fps": FPS, "scenes": a.scenes, "n_clips": len(a.scenes),
               "systems": dict(prev.get("systems", {})) if prev.get("scenes") == a.scenes and prev.get("gt_source") == src else {},
               "runs": dict(prev.get("runs", {})) if prev.get("scenes") == a.scenes and prev.get("gt_source") == src else {}}
    failures = []
    for system in a.systems:
        out_dir = PREDS / system
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        per_clip = {}

        def one(scene: str) -> None:
            out = out_dir / pred_name(scene)
            print(f"[{system}] {scene}", flush=True)
            try:
                pace = 0.0 if (system == "omni" and a.workers > 1) else a.omni_pace
                per_clip[scene] = {"ok": True, **run_system(system, scene, bench, out, cache_root, pace)}
            except Exception as e:  # noqa: BLE001
                if "SpendCapHit" in type(e).__name__:
                    raise
                traceback.print_exc()
                static_prediction(bench, scene, VIEW, FPS, out_dir)  # fallback keeps the pooled score over all clips
                per_clip[scene] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400], "fallback": "static"}
                failures.append({"system": system, "scene": scene, "error": per_clip[scene]["error"]})

        if system == "omni" and a.workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                list(ex.map(one, a.scenes))
        else:
            for scene in a.scenes:
                one(scene)
        scored = score_folder(out_dir, bench, FPS, RESULTS)
        per = {r["scenario"].removeprefix("trimmed-").removesuffix(".mp4"): round(r["final_arithmetic"] * 100, 2)
               for r in scored["per_scenario"]}
        ees = [c.get("endpoint_error_cm") for c in per_clip.values() if c.get("endpoint_error_cm") is not None]
        lats = sorted(c["latency_s"] for c in per_clip.values() if c.get("ok") and c.get("latency_s"))
        results["systems"][system] = {
            "physics_iq_verified_pct": scored["physics_iq_verified_score_pct"],
            "physics_iq_original_pct": scored["physics_iq_original_score_pct"],
            "per_scenario_verified_pct": per,
            "n_ok": sum(c["ok"] for c in per_clip.values()), "n_failed": sum(not c["ok"] for c in per_clip.values()),
            "latency_s_median": lats[len(lats) // 2] if lats else None,
            "endpoint_error_cm_median": sorted(ees)[len(ees) // 2] if ees else None,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "clean_run": a.clean,
        }
        results["runs"][system] = per_clip
        results = merge_write(results, system)
        print(f"== {system}: verified {scored['physics_iq_verified_score_pct']}", flush=True)

    results = merge_write(results, None)
    table(results, EVAL / "results_table.md")
    chart(results, EVAL / "chart.png")
    fl = EVAL / "failures_auto.json"
    old = json.loads(fl.read_text()) if fl.exists() else []
    fl.write_text(json.dumps([f for f in old if f["system"] not in a.systems] + failures, indent=2))
    print(json.dumps({s: v["physics_iq_verified_pct"] for s, v in results["systems"].items()}, indent=1))


if __name__ == "__main__":
    main()
