"""Fit a batch of eval scenes inside the Managed Agent's sandbox and check it reproduces the local fit.

The Managed Agent (antigravity-preview-09-2026) gets one remote Linux sandbox, installs pymunk/scipy and the
physics package once, and reuses that environment for every scene; it receives scene JSON (tracks, geometry),
never video. Each scene is also fitted locally with the same code; eval/agent_batch.json records both.

  PYTHONPATH=backend:phase0 python eval/run_agent_batch.py [--scenes ball-rolls-off ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0"), str(ROOT / "eval")]

import run_eval as E  # noqa: E402
from foresight import config  # noqa: E402
from physics.agent_runner import ManagedPhysicsBatchRunner, fit_batch_local  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=E.SCENES)
    a = ap.parse_args()
    run = sorted(p for p in E.CACHE.glob("run_*") if (p / "work" / "foresight").is_dir())[-1]
    scenes = [json.loads((run / "work" / "foresight" / sc / "scene.json").read_text()) for sc in a.scenes]
    t0 = time.time()
    local = fit_batch_local(scenes)
    t_local = time.time() - t0
    runner = ManagedPhysicsBatchRunner()
    t0 = time.time()
    runner.ensure_environment()
    t_setup = time.time() - t0
    rows = []
    for sc, scene, loc in zip(a.scenes, scenes, local):
        t1 = time.time()
        try:
            rem = runner.fit_scene_remote(scene)
            diffs = [abs(rem["fit_params"][oid][k] - loc["fit_params"][oid][k])
                     for oid in loc["fit_params"] for k in ("x0", "y0", "vx0", "vy0") if oid in rem["fit_params"]]
            rows.append({"scene": sc, "ok": True, "latency_s": round(time.time() - t1, 1),
                         "local_fit_error_px": loc["fit_error_px"], "remote_fit_error_px": rem["fit_error_px"],
                         "max_state_diff": round(max(diffs), 6) if diffs else None, "remote_notes": rem.get("notes", "")[:200]})
        except Exception as e:  # noqa: BLE001
            if "SpendCapHit" in type(e).__name__:
                raise
            rows.append({"scene": sc, "ok": False, "latency_s": round(time.time() - t1, 1), "error": f"{type(e).__name__}: {e}"[:300]})
        print(json.dumps(rows[-1])[:220], flush=True)
    ok = [r for r in rows if r["ok"]]
    same = [r for r in ok if r["max_state_diff"] is not None and r["max_state_diff"] < 1e-3]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": config.MANAGED_AGENT,
           "environment_id_reused": runner.environment_id is not None, "setup_s": round(t_setup, 1),
           "local_batch_s": round(t_local, 1), "n": len(rows), "n_ok": len(ok),
           "n_matching_local": len(same), "match_rule": "start position and velocity within 1e-3 m and m/s of the local fit",
           "scene_source": str(run.relative_to(ROOT)), "rows": rows}
    (E.EVAL / "agent_batch.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n", "n_ok", "n_matching_local", "setup_s", "local_batch_s")}))


if __name__ == "__main__":
    main()
