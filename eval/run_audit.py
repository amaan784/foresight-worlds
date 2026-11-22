"""Batch physicality audit (`foresight audit`, Task A2) over saved eval predictions.

Runs `foresight.audit.audit_video` in parallel across every saved continuation MP4 in:
  - `eval/preds/`      (15 center-view clips x 7 systems = 105 videos)
  - `eval/preds_wide/` (20 scenes x 3 camera views x 5 systems = 300 videos)

Uses cached C4-clean `scene.json` files (`c7_rebuild` and `c4_wide_scenes`) and local ground-truth
testing clips in `data/physics-IQ-benchmark/` with zero live Gemini calls. Writes concise committed
artifacts (`eval/audit_results.json`, `eval/audit_wide.json`, `eval/audit_table.md`), updates the
`<!-- audit: start/end -->` section in `README.md`, and stores full per-frame reports in gitignored
`eval/cache/audit/`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "phase0")]

import physiq_data as pq  # noqa: E402
from foresight.audit import audit_video  # noqa: E402

EVAL = ROOT / "eval"
CACHE_AUDIT = EVAL / "cache" / "audit"
AUDIT_START = "<!-- audit: start (written by eval/run_audit.py from eval/audit_results.json and eval/audit_wide.json; edits here are lost) -->"
AUDIT_END = "<!-- audit: end -->"

MAIN_ORDER = ["foresight", "foresight_carbon", "foresight_neon", "const_velocity", "genie", "omni", "static"]
WIDE_ORDER = ["foresight", "const_velocity", "genie", "omni", "static"]
NAMES = {
    "foresight": "Foresight (physics)", "foresight_carbon": "Foresight (Carbon reads)",
    "foresight_neon": "Foresight (Neon reads)", "const_velocity": "Constant velocity (no physics)",
    "genie": "Genie 4 (world model)", "omni": "Gemini Omni 1.2 Flash (video model)", "static": "Last frame held",
}


def median_or_none(vals: list[float]) -> float | None:
    return round(float(statistics.median(vals)), 1) if vals else None


def mean_or_none(vals: list[float]) -> float | None:
    return round(float(statistics.mean(vals)), 1) if vals else None


def clip_audit_score(report: dict[str, Any]) -> float:
    """Compute bounded [0, 100] physicality score: 30% permanence, 15% energy plausibility, 55% trajectory."""
    perm, plaus, pos = report["permanence"], report["plausibility"], report.get("position_error")
    perm_s = max(0.0, 1.0 - (0.65 if perm["vanished"] else 0.0) - (0.45 if perm["duplicated"] else 0.0)
                 - (0.30 if perm["size_unstable"] else 0.0))
    plaus_s = 1.0 if plaus["plausible"] else 0.0
    errs: list[float] = []
    if pos is not None:
        errs = [float(r["penalized_err_cm"]) for r in pos.get("by_time", []) if r.get("penalized_err_cm") is not None]
        errs += [float(r["lingered_err_cm"]) for r in pos.get("by_time", []) if r.get("lingered_err_cm") is not None]
        if not errs and pos.get("penalized_endpoint_error_cm") is not None:
            errs = [float(pos["penalized_endpoint_error_cm"])]
    traj_s = (1.0 / (1.0 + (sum(errs) / len(errs)) / 12.0)) if errs else 0.5
    return round(100.0 * (0.30 * perm_s + 0.15 * plaus_s + 0.55 * traj_s), 2)


def compact_clip_record(report: dict[str, Any]) -> dict[str, Any]:
    """Extract scalar per-clip audit metrics (omitting bulky frame arrays for committed JSON)."""
    perm, plaus, pos = report["permanence"], report["plausibility"], report.get("position_error") or {}
    ling = [float(r["lingered_err_cm"]) for r in pos.get("by_time", []) if r.get("lingered_err_cm") is not None]
    return {
        "physics_iq_pct": report.get("physics_iq_score_pct"), "audit_score_pct": clip_audit_score(report),
        "passed": bool(report["verdict"]["passed"]), "status": perm["status"],
        "vanished": bool(perm["vanished"]), "vanished_at_s": perm.get("vanished_at_s"),
        "duplicated": bool(perm["duplicated"]), "duplicate_at_s": perm.get("duplicate_at_s"),
        "size_unstable": bool(perm["size_unstable"]), "max_size_ratio": perm["max_size_change_ratio"],
        "plausible": bool(plaus["plausible"]), "max_speed_m_s": plaus["max_speed_m_s"], "bound_m_s": plaus["max_bound_m_s"],
        "raw_ep_cm": pos.get("raw_endpoint_error_cm"), "pen_ep_cm": pos.get("penalized_endpoint_error_cm"),
        "mean_pen_cm": pos.get("mean_penalized_cm"), "lingered_cm": round(max(ling), 1) if ling else None,
    }


def resolve_main_scene_dir() -> Path:
    for tag in ("c7_rebuild", "a1_check", "c6_rebuild", "c4_after_rebuild"):
        d = EVAL / "cache" / "offline" / tag / "work"
        if d.is_dir() and len(list(d.glob("*/scene.json"))) >= 15:
            return d
    runs = sorted(p / "work" / "foresight" for p in (EVAL / "cache").glob("run_*") if (p / "work" / "foresight").is_dir())
    if runs:
        return runs[-1]
    raise FileNotFoundError("No cached 15-clip scene directory found under eval/cache/.")


def resolve_wide_scene_dir() -> Path:
    c4 = EVAL / "cache" / "offline" / "c4_wide_scenes" / "work"
    if c4.is_dir() and len(list(c4.glob("*/scene.json"))) >= 60:
        return c4
    runs = sorted(p / "work" / "foresight" for p in (EVAL / "cache").glob("wide_*") if (p / "work" / "foresight").is_dir())
    if runs:
        return runs[-1]
    raise FileNotFoundError("No cached 60-clip wide scene directory found under eval/cache/.")


def _worker(job: tuple[str, str, str, str, str, str, float | None, str]) -> tuple[str, str, dict[str, Any]]:
    suite, system, key, cond_s, pred_s, real_s, piq, scene_s = job
    rep = audit_video(Path(cond_s), Path(pred_s), reality_video=Path(real_s), scene=Path(scene_s),
                      physics_iq_score_pct=piq, label=system)
    rep["audit_score_pct"] = clip_audit_score(rep)
    cache_out = CACHE_AUDIT / suite / system
    cache_out.mkdir(parents=True, exist_ok=True)
    (cache_out / f"{key.replace('|', '_')}.json").write_text(json.dumps(rep, indent=2))
    return system, key, compact_clip_record(rep)


def aggregate_system(clips: dict[str, dict[str, Any]], piq_pct: float, reported_ep_cm: float | None) -> dict[str, Any]:
    rows = list(clips.values())
    n, n_pass = len(rows), sum(1 for r in rows if r["passed"])
    raw_eps = [float(r["raw_ep_cm"]) for r in rows if r["raw_ep_cm"] is not None]
    pen_eps = [float(r["pen_ep_cm"]) for r in rows if r["pen_ep_cm"] is not None]
    hor_errs = [float(r["mean_pen_cm"]) for r in rows if r["mean_pen_cm"] is not None]
    aud_scores = [float(r["audit_score_pct"]) for r in rows]
    return {
        "n_clips": n, "physics_iq_verified_pct": round(float(piq_pct), 2),
        "audit_score_pct": round(float(statistics.mean(aud_scores)), 2) if aud_scores else 0.0,
        "pass_rate_pct": round(100.0 * n_pass / max(n, 1), 1), "n_passed": n_pass,
        "vanished_count": sum(1 for r in rows if r["vanished"]),
        "duplicated_count": sum(1 for r in rows if r["duplicated"]),
        "size_unstable_count": sum(1 for r in rows if r["size_unstable"]),
        "implausible_count": sum(1 for r in rows if not r["plausible"]),
        "lingered_count": sum(1 for r in rows if r["lingered_cm"] is not None),
        "reported_endpoint_cm_median": round(float(reported_ep_cm), 1) if reported_ep_cm is not None else None,
        "raw_endpoint_cm_median": median_or_none(raw_eps), "raw_endpoint_n": len(raw_eps),
        "penalized_endpoint_cm_median": median_or_none(pen_eps), "penalized_endpoint_n": len(pen_eps),
        "mean_horizon_error_cm_median": median_or_none(hor_errs), "mean_horizon_error_cm_mean": mean_or_none(hor_errs),
    }


def compute_disagreements(
    systems_summary: dict[str, dict[str, Any]], clips_by_system: dict[str, dict[str, dict[str, Any]]], order: list[str],
) -> dict[str, Any]:
    have = [s for s in order if s in systems_summary]
    piq_rank = sorted(have, key=lambda s: -systems_summary[s]["physics_iq_verified_pct"])
    audit_rank = sorted(have, key=lambda s: -systems_summary[s]["audit_score_pct"])
    pen_ep_rank = sorted([s for s in have if systems_summary[s]["penalized_endpoint_cm_median"] is not None],
                         key=lambda s: systems_summary[s]["penalized_endpoint_cm_median"])
    rep_ep_rank = sorted([s for s in have if systems_summary[s]["reported_endpoint_cm_median"] is not None],
                         key=lambda s: systems_summary[s]["reported_endpoint_cm_median"])

    inversions: list[dict[str, Any]] = []
    if "const_velocity" in systems_summary and "foresight" in systems_summary:
        cv, fh = systems_summary["const_velocity"], systems_summary["foresight"]
        if cv["physics_iq_verified_pct"] > fh["physics_iq_verified_pct"] and fh["audit_score_pct"] > cv["audit_score_pct"]:
            inversions.append({
                "kind": "pixel_stillness_and_early_exit_bias", "winner_on_physics_iq": "const_velocity", "winner_on_audit": "foresight",
                "physics_iq_pct": {"const_velocity": cv["physics_iq_verified_pct"], "foresight": fh["physics_iq_verified_pct"]},
                "audit_score_pct": {"const_velocity": cv["audit_score_pct"], "foresight": fh["audit_score_pct"]},
                "pass_rate_pct": {"const_velocity": cv["pass_rate_pct"], "foresight": fh["pass_rate_pct"]},
                "penalized_endpoint_cm_median": {"const_velocity": cv["penalized_endpoint_cm_median"], "foresight": fh["penalized_endpoint_cm_median"]},
                "mean_horizon_error_cm_median": {"const_velocity": cv["mean_horizon_error_cm_median"], "foresight": fh["mean_horizon_error_cm_median"]},
                "explanation": ("Physics-IQ rewards clean empty backgrounds after a straight-line sprite shoots off-screen early, "
                                "whereas foresight audit penalizes early border exits and mid-table pass-throughs."),
            })

    for vm in ("omni", "genie"):
        if vm in systems_summary and "foresight" in systems_summary:
            v, fh = systems_summary[vm], systems_summary["foresight"]
            rep_v, rep_f = v["reported_endpoint_cm_median"], fh["reported_endpoint_cm_median"]
            raw_v, pen_v, pen_f = v["raw_endpoint_cm_median"], v["penalized_endpoint_cm_median"], fh["penalized_endpoint_cm_median"]
            vanish_jump = raw_v is not None and pen_v is not None and pen_v - raw_v >= 10.0
            rep_inverted = rep_v is not None and rep_f is not None and rep_v < rep_f and pen_v is not None and pen_f is not None and pen_f < pen_v
            if rep_inverted or vanish_jump:
                inversions.append({
                    "kind": "stopped_track_vanish_bias", "video_model": vm,
                    "reported_endpoint_cm_median": {vm: rep_v, "foresight": rep_f},
                    "raw_endpoint_cm_median": {vm: raw_v, "foresight": fh["raw_endpoint_cm_median"]},
                    "penalized_endpoint_cm_median": {vm: pen_v, "foresight": pen_f},
                    "vanished_count": {vm: v["vanished_count"], "foresight": fh["vanished_count"]},
                    "duplicated_count": {vm: v["duplicated_count"], "foresight": fh["duplicated_count"]},
                    "explanation": (f"Stopped-track endpoint error stops measuring whenever {vm} drops or blurs the object "
                                    f"({v['vanished_count']}/{v['n_clips']} vanished, {v['duplicated_count']}/{v['n_clips']} duplicated); "
                                    "counting vanished objects against fixed-horizon reality flips the error ordering."),
                })

    clip_cases: list[dict[str, Any]] = []
    if "foresight" in clips_by_system and "const_velocity" in clips_by_system:
        for k, f_c in clips_by_system["foresight"].items():
            cv_c = clips_by_system["const_velocity"].get(k)
            if not cv_c:
                continue
            f_piq, cv_piq, f_err, cv_err = f_c["physics_iq_pct"], cv_c["physics_iq_pct"], f_c["mean_pen_cm"], cv_c["mean_pen_cm"]
            if f_piq is not None and cv_piq is not None and cv_piq >= f_piq and f_err is not None and cv_err is not None and cv_err >= f_err + 2.0:
                clip_cases.append({
                    "clip": k, "type": "const_velocity_beats_foresight_on_piq_but_loses_on_audit",
                    "physics_iq_pct": {"const_velocity": cv_piq, "foresight": f_piq},
                    "audit_score_pct": {"const_velocity": cv_c["audit_score_pct"], "foresight": f_c["audit_score_pct"]},
                    "mean_penalized_cm": {"const_velocity": cv_err, "foresight": f_err},
                })
    clip_cases.sort(key=lambda x: -(x["mean_penalized_cm"]["const_velocity"] - x["mean_penalized_cm"]["foresight"]))

    hallucinations: list[dict[str, Any]] = []
    for vm in ("omni", "genie"):
        for k, vc in clips_by_system.get(vm, {}).items():
            if vc["vanished"] or vc["duplicated"] or (not vc["plausible"]):
                if (vc["raw_ep_cm"] is not None and vc["pen_ep_cm"] is not None and vc["pen_ep_cm"] - vc["raw_ep_cm"] >= 15.0) or vc["duplicated"] or (not vc["plausible"]):
                    hallucinations.append({
                        "clip": k, "system": vm, "physics_iq_pct": vc["physics_iq_pct"],
                        "vanished": vc["vanished"], "duplicated": vc["duplicated"], "plausible": vc["plausible"],
                        "raw_ep_cm": vc["raw_ep_cm"], "pen_ep_cm": vc["pen_ep_cm"],
                    })
    hallucinations.sort(key=lambda x: -((x["pen_ep_cm"] or 0.0) - (x["raw_ep_cm"] or 0.0)))

    return {
        "physics_iq_ranking": piq_rank, "audit_score_ranking": audit_rank,
        "reported_endpoint_ranking": rep_ep_rank, "penalized_endpoint_ranking": pen_ep_rank,
        "system_rank_inversions": inversions,
        "const_velocity_vs_foresight_clip_inversions_count": len(clip_cases),
        "const_velocity_vs_foresight_top_clips": clip_cases[:10],
        "video_model_masked_failures_count": len(hallucinations),
        "video_model_top_masked_clips": hallucinations[:12],
    }


def dumps_compact_clips(payload: dict[str, Any]) -> str:
    """Serialize audit JSON cleanly with one compact JSON line per clip or disagreement row."""
    dis = payload.get("disagreements")
    top = {k: v for k, v in payload.items() if k not in ("disagreements", "clips")}
    lines = [json.dumps(top, indent=2)[:-2] + ","]
    if dis is not None:
        list_keys = ("system_rank_inversions", "const_velocity_vs_foresight_top_clips", "video_model_top_masked_clips")
        lines.append('  "disagreements": {')
        for k, v in dis.items():
            if k not in list_keys:
                lines.append(f'    "{k}": {json.dumps(v)},')
        for li, lk in enumerate(list_keys):
            rows, l_comma = dis.get(lk, []), ("," if li + 1 < len(list_keys) else "")
            if not rows:
                lines.append(f'    "{lk}": []{l_comma}')
            else:
                lines.append(f'    "{lk}": [')
                for ri, row in enumerate(rows):
                    lines.append(f"      {json.dumps(row, separators=(', ', ': '))}{',' if ri + 1 < len(rows) else ''}")
                lines.append(f"    ]{l_comma}")
        lines.append("  },")
    lines.append('  "clips": {')
    sys_items = list(payload["clips"].items())
    for si, (sys_name, clips) in enumerate(sys_items):
        lines.append(f'    "{sys_name}": {{')
        clip_items = list(clips.items())
        for ci, (ck, cv) in enumerate(clip_items):
            lines.append(f'      "{ck}": {json.dumps(cv, separators=(", ", ": "))}{"," if ci + 1 < len(clip_items) else ""}')
        lines.append(f"    }}{',' if si + 1 < len(sys_items) else ''}")
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def run_main_suite(workers: int) -> dict[str, Any]:
    r = json.loads((EVAL / "results.json").read_text())
    bench, scene_dir, preds_root = pq.benchmark_dir(r["gt_source"]), resolve_main_scene_dir(), EVAL / "preds"
    systems = [s for s in MAIN_ORDER if s in r["systems"] and (preds_root / s).is_dir()]
    jobs = []
    for s in systems:
        for sc in r["scenes"]:
            id1, _ = pq.scene_ids(sc, "center")
            pred = preds_root / s / f"{id1}_perspective-center_trimmed-{sc}.mp4"
            cond, real = bench / pq.needed_files(sc, "center", 24)[0], bench / pq.needed_files(sc, "center", 24)[1]
            jobs.append(("main", s, sc, str(cond), str(pred), str(real),
                         r["systems"][s]["per_scenario_verified_pct"].get(sc), str(scene_dir / sc / "scene.json")))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(_worker, jobs))
    elapsed = round(time.time() - t0, 2)
    clips_by_sys: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in systems}
    for s, sc, rec in recs:
        clips_by_sys[s][sc] = rec
    sys_summary = {s: aggregate_system(clips_by_sys[s], r["systems"][s]["physics_iq_verified_pct"],
                                       r["systems"][s].get("endpoint_error_cm_median")) for s in systems}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {
        "suite": "main_15_center", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "scene_cache": str(scene_dir.relative_to(ROOT)), "n_clips": len(r["scenes"]),
        "n_videos_audited": len(recs), "runtime_s": elapsed, "gemini_calls": 0, "scenes": r["scenes"],
        "systems": sys_summary, "disagreements": compute_disagreements(sys_summary, clips_by_sys, MAIN_ORDER), "clips": clips_by_sys,
    }


def run_wide_suite(workers: int) -> dict[str, Any]:
    w = json.loads((EVAL / "results_wide.json").read_text())
    bench, scene_dir, preds_root = pq.benchmark_dir(w["gt_source"]), resolve_wide_scene_dir(), EVAL / "preds_wide"
    systems = [s for s in WIDE_ORDER if s in w["systems"] and (preds_root / s).is_dir()]
    jobs = []
    for s in systems:
        for sc in w["scenes"]:
            for view in w["views"]:
                id1, _ = pq.scene_ids(sc, view)
                pred = preds_root / s / view / f"{id1}_perspective-{view}_trimmed-{sc}.mp4"
                cond, real = bench / pq.needed_files(sc, view, 24)[0], bench / pq.needed_files(sc, view, 24)[1]
                key = f"{sc}|{view}"
                jobs.append(("wide", s, key, str(cond), str(pred), str(real),
                             w["systems"][s]["per_scene_view_pct"].get(key), str(scene_dir / f"{sc}_{view}" / "scene.json")))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(_worker, jobs))
    elapsed = round(time.time() - t0, 2)
    clips_by_sys: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in systems}
    for s, key, rec in recs:
        clips_by_sys[s][key] = rec
    sys_summary = {s: aggregate_system(clips_by_sys[s], w["systems"][s]["pooled_verified_pct"],
                                       w["systems"][s].get("endpoint_error_cm_median")) for s in systems}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {
        "suite": "wide_60_all_views", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "scene_cache": str(scene_dir.relative_to(ROOT)),
        "n_scene_views": len(w["scenes"]) * len(w["views"]), "n_videos_audited": len(recs), "runtime_s": elapsed,
        "gemini_calls": 0, "scenes": w["scenes"], "views": w["views"], "systems": sys_summary,
        "disagreements": compute_disagreements(sys_summary, clips_by_sys, WIDE_ORDER), "clips": clips_by_sys,
    }


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def _sys_table_lines(systems_summary: dict[str, dict[str, Any]], order: list[str]) -> list[str]:
    ranked = sorted([s for s in order if s in systems_summary], key=lambda k: -systems_summary[k]["audit_score_pct"])
    best_piq = max(systems_summary[k]["physics_iq_verified_pct"] for k in ranked)
    best_aud = max(systems_summary[k]["audit_score_pct"] for k in ranked)
    best_pen = min(systems_summary[k]["penalized_endpoint_cm_median"] for k in ranked if systems_summary[k]["penalized_endpoint_cm_median"] is not None)
    best_hor = min(systems_summary[k]["mean_horizon_error_cm_median"] for k in ranked if systems_summary[k]["mean_horizon_error_cm_median"] is not None)
    lines = [
        "| System | Audit score (%) | Physics-IQ (%) | Audit pass | Vanished | Duplicated | Energy viol. | Stopped-track ep. (cm) | Penalized ep. (cm) | Horizon err. med (cm) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in ranked:
        s = systems_summary[k]
        aud = f"**{s['audit_score_pct']:.1f}**" if s["audit_score_pct"] == best_aud else f"{s['audit_score_pct']:.1f}"
        piq = f"**{s['physics_iq_verified_pct']:.1f}**" if s["physics_iq_verified_pct"] == best_piq else f"{s['physics_iq_verified_pct']:.1f}"
        pen = f"**{_fmt(s['penalized_endpoint_cm_median'])}**" if s["penalized_endpoint_cm_median"] == best_pen else _fmt(s["penalized_endpoint_cm_median"])
        hor = f"**{_fmt(s['mean_horizon_error_cm_median'])}**" if s["mean_horizon_error_cm_median"] == best_hor else _fmt(s["mean_horizon_error_cm_median"])
        lines.append(
            f"| {NAMES.get(k, k)} | {aud} | {piq} | {s['n_passed']}/{s['n_clips']} ({s['pass_rate_pct']:.1f}%) "
            f"| {s['vanished_count']} | {s['duplicated_count']} | {s['implausible_count']} "
            f"| {_fmt(s['reported_endpoint_cm_median'])} | {pen} | {hor} |"
        )
    return lines


def render_audit_table_md(main_res: dict[str, Any], wide_res: dict[str, Any]) -> str:
    m_sys, w_sys = main_res["systems"], wide_res["systems"]
    m_dis, w_dis = main_res["disagreements"], wide_res["disagreements"]
    lines = [
        "# Physics auditor (`foresight audit`): saved predictions vs. pixel score", "",
        f"Generated by `eval/run_audit.py` at {main_res['generated_at']} (commit `{main_res['git_commit']}`), "
        f"`0` live Gemini calls ({main_res['n_videos_audited']} videos in `eval/preds/`, "
        f"{wide_res['n_videos_audited']} videos in `eval/preds_wide/`).", "",
        "Composite **Audit score (%)** weights occlusion-aware object permanence (`30%`), mechanical energy plausibility "
        "(`15%`), and vanish/early-exit-penalized trajectory accuracy (`55%`, `1 / (1 + err_cm / 12)`).", "",
        f"## 1. Main eval: {main_res['n_clips']} rigid-body clips, center camera", "",
        *_sys_table_lines(m_sys, MAIN_ORDER), "",
        f"## 2. Wider eval: {len(wide_res['scenes'])} scenes x {len(wide_res['views'])} cameras ({wide_res['n_scene_views']} scene-views)", "",
        *_sys_table_lines(w_sys, WIDE_ORDER), "",
        "## 3. Where `foresight audit` and Physics-IQ / stopped-track endpoint disagree", "",
        f"- **Straight-line pixel stillness vs. trajectory accuracy**: On the 15-clip eval, Constant velocity leads "
        f"Physics-IQ (**{m_sys['const_velocity']['physics_iq_verified_pct']:.1f}%** vs. Foresight "
        f"**{m_sys['foresight']['physics_iq_verified_pct']:.1f}%**) because leaving the frame early yields a clean background, "
        f"but drops from `{m_sys['foresight']['audit_score_pct']:.1f}%` to `{m_sys['const_velocity']['audit_score_pct']:.1f}%` on "
        f"audit score (`{m_sys['const_velocity']['pass_rate_pct']:.1f}%` pass rate vs. `{m_sys['foresight']['pass_rate_pct']:.1f}%`, "
        f"median penalized horizon error `{m_sys['const_velocity']['mean_horizon_error_cm_median']:.1f} cm` vs. "
        f"`{m_sys['foresight']['mean_horizon_error_cm_median']:.1f} cm`). The same inversion holds on the 60-clip wide eval "
        f"(Physics-IQ `{w_sys['const_velocity']['physics_iq_verified_pct']:.1f}%` vs. `{w_sys['foresight']['physics_iq_verified_pct']:.1f}%`, "
        f"audit score `{w_sys['const_velocity']['audit_score_pct']:.1f}%` vs. `{w_sys['foresight']['audit_score_pct']:.1f}%`, "
        f"penalized endpoint `{w_sys['const_velocity']['penalized_endpoint_cm_median']:.1f} cm` vs. "
        f"`{w_sys['foresight']['penalized_endpoint_cm_median']:.1f} cm`).",
        f"- **Stopped-track vanish bias on the 60-clip wide eval**: Stopped-track endpoint error ranked Genie 4 "
        f"(`{w_sys['genie']['reported_endpoint_cm_median']:.1f} cm`) and Gemini Omni 1.2 Flash "
        f"(`{w_sys['omni']['reported_endpoint_cm_median']:.1f} cm`) ahead of Foresight "
        f"(`{w_sys['foresight']['reported_endpoint_cm_median']:.1f} cm` pre-C4) because tracking stopped whenever the object "
        f"vanished. On C4-clean scenes with vanish penalties, **Foresight leads at "
        f"`{w_sys['foresight']['penalized_endpoint_cm_median']:.1f} cm` penalized endpoint median "
        f"(`{w_sys['foresight']['mean_horizon_error_cm_median']:.1f} cm` horizon median, `{w_sys['foresight']['pass_rate_pct']:.1f}%` pass)**, "
        f"ahead of Genie 4 (`{w_sys['genie']['penalized_endpoint_cm_median']:.1f} cm`, `{w_sys['genie']['vanished_count']}` vanished, "
        f"`{w_sys['genie']['duplicated_count']}` duplicated, `{w_sys['genie']['implausible_count']}` energy violation) and "
        f"Gemini Omni 1.2 Flash (`{w_sys['omni']['penalized_endpoint_cm_median']:.1f} cm`, `{w_sys['omni']['vanished_count']}/60` vanished, "
        f"`{w_sys['omni']['duplicated_count']}/60` duplicated, `{w_sys['omni']['pass_rate_pct']:.1f}%` pass).", "",
        "### Top clip-level inversions (`const_velocity` >= `foresight` on Physics-IQ, worse on penalized error)", "",
        "| Clip | `const_velocity` PIQ (%) | `foresight` PIQ (%) | `const_velocity` horizon err (cm) | `foresight` horizon err (cm) | `const_velocity` audit (%) | `foresight` audit (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in m_dis["const_velocity_vs_foresight_top_clips"][:6]:
        lines.append(
            f"| {c['clip']} | {c['physics_iq_pct']['const_velocity']:.1f} | {c['physics_iq_pct']['foresight']:.1f} "
            f"| {c['mean_penalized_cm']['const_velocity']:.1f} | {c['mean_penalized_cm']['foresight']:.1f} "
            f"| {c['audit_score_pct']['const_velocity']:.1f} | {c['audit_score_pct']['foresight']:.1f} |"
        )
    lines += [
        "", "### Video-model hallucinations masked by stopped-track endpoint error (wide eval examples)", "",
        "| Clip | System | Physics-IQ (%) | Vanished | Duplicated | Plausible | Raw stopped ep. (cm) | Penalized ep. (cm) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in w_dis["video_model_top_masked_clips"][:8]:
        lines.append(
            f"| {h['clip']} | {h['system']} | {_fmt(h['physics_iq_pct'])} | {'yes' if h['vanished'] else 'no'} "
            f"| {'yes' if h['duplicated'] else 'no'} | {'yes' if h['plausible'] else 'no'} "
            f"| {_fmt(h['raw_ep_cm'])} | {_fmt(h['pen_ep_cm'])} |"
        )
    return "\n".join(lines) + "\n"


def render_readme_audit_section(main_res: dict[str, Any], wide_res: dict[str, Any]) -> str:
    ms, ws = main_res["systems"], wide_res["systems"]
    return "\n".join([
        AUDIT_START, "## Physics auditor", "",
        "Pixel masks reward an empty background after a straight-line sprite leaves the frame early, and stopped-track "
        "endpoint error stops measuring when a video model blurs or drops the object. `foresight audit` "
        "([CLI and Python API](backend/foresight/audit.py)) audits any 5 s continuation video on three physical checks — "
        "occlusion-aware **object permanence** (hidden behind an object in front vs. unexplained mid-frame vanish, "
        "duplicate respawning, size stability), **fixed-horizon position error** at `+0.5, 1.0, 2.0, 3.0, 5.0 s` "
        "(counting vanishes and early border exits as errors), and **free-fall mechanical energy bounds**.", "",
        "```bash",
        "foresight audit conditioning_3s.mp4 model_5s.mp4 --reality reality_5s.mp4 --out audit_report/",
        "```", "",
        f"Auditing all {main_res['n_videos_audited']} saved predictions from the 15-clip eval "
        f"([eval/audit_results.json](eval/audit_results.json)) and all {wide_res['n_videos_audited']} predictions from the "
        f"60-scene-view wide eval ([eval/audit_wide.json](eval/audit_wide.json), [full table](eval/audit_table.md)) "
        "resolves both metric inversions:", "",
        f"- **15-clip eval (`eval/preds/`)**: Constant velocity leads Physics-IQ (`{ms['const_velocity']['physics_iq_verified_pct']:.1f}%` "
        f"vs. Foresight `{ms['foresight']['physics_iq_verified_pct']:.1f}%`), yet passes physicality on only "
        f"`{ms['const_velocity']['n_passed']}/{ms['const_velocity']['n_clips']}` clips (`{ms['const_velocity']['pass_rate_pct']:.1f}%`) "
        f"with `{ms['const_velocity']['penalized_endpoint_cm_median']:.1f} cm` median penalized endpoint error (`{ms['const_velocity']['mean_horizon_error_cm_median']:.1f} cm` "
        f"horizon median, audit score `{ms['const_velocity']['audit_score_pct']:.1f}%`), while **Foresight passes "
        f"`{ms['foresight']['n_passed']}/{ms['foresight']['n_clips']}` (`{ms['foresight']['pass_rate_pct']:.1f}%`) at "
        f"`{ms['foresight']['penalized_endpoint_cm_median']:.1f} cm` penalized endpoint median (`{ms['foresight']['mean_horizon_error_cm_median']:.1f} cm` "
        f"horizon median, audit score `{ms['foresight']['audit_score_pct']:.1f}%`; Carbon `{ms['foresight_carbon']['audit_score_pct']:.1f}%`, "
        f"Neon `{ms['foresight_neon']['audit_score_pct']:.1f}%`)**. Gemini Omni 1.2 Flash passes `{ms['omni']['n_passed']}/{ms['omni']['n_clips']}` "
        f"(`{ms['omni']['vanished_count']}` vanished, `{ms['omni']['duplicated_count']}` duplicated, `{ms['omni']['penalized_endpoint_cm_median']:.1f} cm` penalized endpoint).",
        f"- **60-scene-view wide eval (`eval/preds_wide/`)**: Stopped-track endpoint error put Genie 4 "
        f"(`{ws['genie']['reported_endpoint_cm_median']:.1f} cm`) and Gemini Omni 1.2 Flash (`{ws['omni']['reported_endpoint_cm_median']:.1f} cm`) "
        f"below Foresight (`{ws['foresight']['reported_endpoint_cm_median']:.1f} cm`) because vanished tracks ended early. "
        f"With vanish and early-exit penalties on C4-clean scenes, **Foresight ranks first on audit score (`{ws['foresight']['audit_score_pct']:.1f}%`), "
        f"penalized endpoint median (`{ws['foresight']['penalized_endpoint_cm_median']:.1f} cm`), and horizon median "
        f"(`{ws['foresight']['mean_horizon_error_cm_median']:.1f} cm`, `{ws['foresight']['n_passed']}/{ws['foresight']['n_clips']}` passed)** — "
        f"ahead of Genie 4 (`{ws['genie']['audit_score_pct']:.1f}%`, `{ws['genie']['penalized_endpoint_cm_median']:.1f} cm`, "
        f"`{ws['genie']['vanished_count']}` vanished, `{ws['genie']['duplicated_count']}` duplicated, `{ws['genie']['implausible_count']}` energy violation), "
        f"Constant velocity (`{ws['const_velocity']['audit_score_pct']:.1f}%`, `{ws['const_velocity']['penalized_endpoint_cm_median']:.1f} cm`), "
        f"Last frame held (`{ws['static']['audit_score_pct']:.1f}%`, `{ws['static']['penalized_endpoint_cm_median']:.1f} cm`), "
        f"and Gemini Omni 1.2 Flash (`{ws['omni']['audit_score_pct']:.1f}%`, `{ws['omni']['penalized_endpoint_cm_median']:.1f} cm`, "
        f"`{ws['omni']['vanished_count']}/60` vanished, `{ws['omni']['duplicated_count']}/60` duplicated).", "",
        "| Suite | Foresight | Genie 4 | Constant velocity | Last frame held | Gemini Omni 1.2 Flash |",
        "|---|---|---|---|---|---|",
        f"| **15-clip audit score / Physics-IQ (%)** | **{ms['foresight']['audit_score_pct']:.1f}** / {ms['foresight']['physics_iq_verified_pct']:.1f} "
        f"| {ms['genie']['audit_score_pct']:.1f} / {ms['genie']['physics_iq_verified_pct']:.1f} "
        f"| {ms['const_velocity']['audit_score_pct']:.1f} / **{ms['const_velocity']['physics_iq_verified_pct']:.1f}** "
        f"| {ms['static']['audit_score_pct']:.1f} / {ms['static']['physics_iq_verified_pct']:.1f} "
        f"| {ms['omni']['audit_score_pct']:.1f} / {ms['omni']['physics_iq_verified_pct']:.1f} |",
        f"| **15-clip penalized / stopped ep. (cm)** | **{ms['foresight']['penalized_endpoint_cm_median']:.1f}** / {ms['foresight']['reported_endpoint_cm_median']:.1f} "
        f"| {ms['genie']['penalized_endpoint_cm_median']:.1f} / {ms['genie']['reported_endpoint_cm_median']:.1f} "
        f"| {ms['const_velocity']['penalized_endpoint_cm_median']:.1f} / {ms['const_velocity']['reported_endpoint_cm_median']:.1f} "
        f"| {ms['static']['penalized_endpoint_cm_median']:.1f} / {ms['static']['reported_endpoint_cm_median']:.1f} "
        f"| {ms['omni']['penalized_endpoint_cm_median']:.1f} / {ms['omni']['reported_endpoint_cm_median']:.1f} |",
        f"| **60-clip audit score / Physics-IQ (%)** | **{ws['foresight']['audit_score_pct']:.1f}** / {ws['foresight']['physics_iq_verified_pct']:.1f} "
        f"| {ws['genie']['audit_score_pct']:.1f} / {ws['genie']['physics_iq_verified_pct']:.1f} "
        f"| {ws['const_velocity']['audit_score_pct']:.1f} / **{ws['const_velocity']['physics_iq_verified_pct']:.1f}** "
        f"| {ws['static']['audit_score_pct']:.1f} / {ws['static']['physics_iq_verified_pct']:.1f} "
        f"| {ws['omni']['audit_score_pct']:.1f} / {ws['omni']['physics_iq_verified_pct']:.1f} |",
        f"| **60-clip penalized / stopped ep. (cm)** | **{ws['foresight']['penalized_endpoint_cm_median']:.1f}** / {ws['foresight']['reported_endpoint_cm_median']:.1f} "
        f"| {ws['genie']['penalized_endpoint_cm_median']:.1f} / {ws['genie']['reported_endpoint_cm_median']:.1f} "
        f"| {ws['const_velocity']['penalized_endpoint_cm_median']:.1f} / {ws['const_velocity']['reported_endpoint_cm_median']:.1f} "
        f"| {ws['static']['penalized_endpoint_cm_median']:.1f} / {ws['static']['reported_endpoint_cm_median']:.1f} "
        f"| {ws['omni']['penalized_endpoint_cm_median']:.1f} / {ws['omni']['reported_endpoint_cm_median']:.1f} |",
        AUDIT_END,
    ])


def update_readme(main_res: dict[str, Any], wide_res: dict[str, Any], readme_path: Path | None = None) -> None:
    path = readme_path or (ROOT / "README.md")
    text = path.read_text()
    block = render_readme_audit_section(main_res, wide_res)
    if AUDIT_START in text and AUDIT_END in text:
        a, b = text.index(AUDIT_START), text.index(AUDIT_END) + len(AUDIT_END)
        path.write_text(text[:a] + block + text[b:])
    else:
        idx = text.index("<!-- results: end -->") + len("<!-- results: end -->")
        path.write_text(text[:idx] + "\n\n" + block + text[idx:])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run foresight.audit over saved predictions in eval/preds and eval/preds_wide.")
    ap.add_argument("--suite", choices=["all", "main", "wide"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--render-only", action="store_true", help="Rebuild audit_table.md and README section from audit_*.json.")
    args = ap.parse_args()

    main_path, wide_path = EVAL / "audit_results.json", EVAL / "audit_wide.json"
    if not args.render_only:
        if args.suite in ("all", "main"):
            main_res = run_main_suite(args.workers)
            main_path.write_text(dumps_compact_clips(main_res))
            print("main:", json.dumps({k: v["audit_score_pct"] for k, v in main_res["systems"].items()}))
        if args.suite in ("all", "wide"):
            wide_res = run_wide_suite(args.workers)
            wide_path.write_text(dumps_compact_clips(wide_res))
            print("wide:", json.dumps({k: v["audit_score_pct"] for k, v in wide_res["systems"].items()}))

    if main_path.exists() and wide_path.exists():
        main_res, wide_res = json.loads(main_path.read_text()), json.loads(wide_path.read_text())
        if args.render_only:
            main_path.write_text(dumps_compact_clips(main_res))
            wide_path.write_text(dumps_compact_clips(wide_res))
        (EVAL / "audit_table.md").write_text(render_audit_table_md(main_res, wide_res))
        update_readme(main_res, wide_res)
        print("Wrote eval/audit_table.md and updated README.md physics auditor section.")


if __name__ == "__main__":
    main()
