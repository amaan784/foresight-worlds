"""Offline unit, artifact-consistency, and API tests for Task A2 (`eval/run_audit.py`)."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from foresight import api, config
from physics.tests.test_c2_session_capture import _asgi_request

ROOT = config.ROOT


def _load_run_audit():
    spec = importlib.util.spec_from_file_location("run_audit", ROOT / "eval" / "run_audit.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_report(
    *,
    piq: float,
    vanished: bool = False,
    duplicated: bool = False,
    size_unstable: bool = False,
    plausible: bool = True,
    raw_ep: float | None = 4.0,
    pen_ep: float | None = 4.0,
    by_time_pen: list[float] | None = None,
    lingered: list[float] | None = None,
) -> dict:
    pen_list = [4.0, 4.0] if by_time_pen is None else by_time_pen
    by_time = [{"dt_s": 0.5 * (i + 1), "penalized_err_cm": v, "lingered_err_cm": None} for i, v in enumerate(pen_list)]
    for lv in lingered or []:
        by_time.append({"dt_s": 3.0, "penalized_err_cm": None, "lingered_err_cm": lv})
    passed = (not vanished) and (not duplicated) and (not size_unstable) and plausible
    return {
        "physics_iq_score_pct": piq,
        "verdict": {"passed": passed, "reasons": []},
        "permanence": {
            "passed": (not vanished) and (not duplicated) and (not size_unstable),
            "status": "vanished" if vanished else "visible_to_end",
            "vanished": vanished,
            "vanished_at_s": 0.25 if vanished else None,
            "duplicated": duplicated,
            "duplicate_at_s": 0.58 if duplicated else None,
            "size_unstable": size_unstable,
            "max_size_change_ratio": 2.1 if size_unstable else 1.04,
        },
        "plausibility": {
            "plausible": plausible,
            "max_speed_m_s": 1.1 if plausible else 4.8,
            "max_bound_m_s": 1.8,
        },
        "position_error": {
            "by_time": by_time,
            "raw_endpoint_error_cm": raw_ep,
            "penalized_endpoint_error_cm": pen_ep,
            "mean_penalized_cm": round(sum(pen_list) / len(pen_list), 1) if pen_list else None,
        },
    }


def test_run_audit_scoring_medians_and_disagreement_logic() -> None:
    ra = _load_run_audit()

    # Exact mathematical median for both odd and even sample counts
    assert ra.median_or_none([]) is None
    assert ra.median_or_none([2.0, 10.0, 30.0]) == pytest.approx(10.0)
    assert ra.median_or_none([2.0, 6.0, 10.0, 30.0]) == pytest.approx(8.0)

    rep_good = _synthetic_report(piq=88.0, raw_ep=3.0, pen_ep=3.0, by_time_pen=[2.5, 3.5])
    rep_static = _synthetic_report(piq=35.0, raw_ep=25.0, pen_ep=25.0, by_time_pen=[22.0, 28.0], lingered=[30.0])
    rep_cv = _synthetic_report(piq=93.0, vanished=True, raw_ep=1.5, pen_ep=42.0, by_time_pen=[18.0, 42.0])
    rep_omni = _synthetic_report(piq=28.0, vanished=True, duplicated=True, raw_ep=5.0, pen_ep=48.0, by_time_pen=[40.0, 48.0])

    s_good = ra.clip_audit_score(rep_good)
    s_static = ra.clip_audit_score(rep_static)
    s_cv = ra.clip_audit_score(rep_cv)
    s_omni = ra.clip_audit_score(rep_omni)
    assert 0.0 <= s_omni < s_cv < s_static < s_good <= 100.0

    clips = {
        "foresight": {"c1": ra.compact_clip_record(rep_good)},
        "const_velocity": {"c1": ra.compact_clip_record(rep_cv)},
        "omni": {"c1": ra.compact_clip_record(rep_omni)},
    }
    summary = {
        "foresight": ra.aggregate_system(clips["foresight"], piq_pct=88.0, reported_ep_cm=18.2),
        "const_velocity": ra.aggregate_system(clips["const_velocity"], piq_pct=93.0, reported_ep_cm=154.8),
        "omni": ra.aggregate_system(clips["omni"], piq_pct=28.0, reported_ep_cm=14.7),
    }
    dis = ra.compute_disagreements(summary, clips, ["foresight", "const_velocity", "omni"])
    kinds = {inv["kind"] for inv in dis["system_rank_inversions"]}
    assert "pixel_stillness_and_early_exit_bias" in kinds
    assert "stopped_track_vanish_bias" in kinds
    assert dis["const_velocity_vs_foresight_clip_inversions_count"] == 1
    assert dis["video_model_masked_failures_count"] == 1

    # Compact clip JSON serializer round-trips cleanly as valid JSON
    payload = {"suite": "unit", "systems": summary, "disagreements": dis, "clips": clips}
    dumped = ra.dumps_compact_clips(payload)
    assert json.loads(dumped) == payload


def test_committed_audit_artifacts_readme_and_dockerfile_are_in_sync() -> None:
    ra = _load_run_audit()
    main_path = ROOT / "eval" / "audit_results.json"
    wide_path = ROOT / "eval" / "audit_wide.json"
    table_path = ROOT / "eval" / "audit_table.md"
    assert main_path.exists() and wide_path.exists() and table_path.exists()

    main_res = json.loads(main_path.read_text())
    wide_res = json.loads(wide_path.read_text())

    assert main_res["gemini_calls"] == 0
    assert wide_res["gemini_calls"] == 0
    assert main_res["n_clips"] == 15
    assert main_res["n_videos_audited"] == 105
    assert set(main_res["systems"].keys()) == set(ra.MAIN_ORDER)

    assert wide_res["n_scene_views"] == 60
    assert wide_res["n_videos_audited"] == 300
    assert set(wide_res["systems"].keys()) == set(ra.WIDE_ORDER)

    # Bulky per-frame arrays must never be committed inside eval/audit_*.json
    sample_clip = next(iter(main_res["clips"]["foresight"].values()))
    assert "tracks" not in sample_clip and "speed_series" not in sample_clip

    # Verify both key empirical inversions hold in the committed evaluation results
    ms, ws = main_res["systems"], wide_res["systems"]
    assert ms["const_velocity"]["physics_iq_verified_pct"] > ms["foresight"]["physics_iq_verified_pct"]
    assert ms["foresight"]["audit_score_pct"] > ms["const_velocity"]["audit_score_pct"]
    assert ms["foresight"]["penalized_endpoint_cm_median"] < ms["const_velocity"]["penalized_endpoint_cm_median"]

    assert ws["genie"]["reported_endpoint_cm_median"] < ws["foresight"]["reported_endpoint_cm_median"]
    assert ws["omni"]["reported_endpoint_cm_median"] < ws["foresight"]["reported_endpoint_cm_median"]
    assert ws["foresight"]["penalized_endpoint_cm_median"] < ws["genie"]["penalized_endpoint_cm_median"]
    assert ws["foresight"]["penalized_endpoint_cm_median"] < ws["omni"]["penalized_endpoint_cm_median"]
    assert ws["foresight"]["audit_score_pct"] > ws["genie"]["audit_score_pct"] > ws["omni"]["audit_score_pct"]

    # Table and README audit section must match programmatic output with zero hand-edited numbers
    assert table_path.read_text() == ra.render_audit_table_md(main_res, wide_res)

    readme_txt = (ROOT / "README.md").read_text()
    res_end = readme_txt.index("<!-- results: end -->")
    aud_start = readme_txt.index(ra.AUDIT_START)
    aud_end = readme_txt.index(ra.AUDIT_END) + len(ra.AUDIT_END)
    assert aud_start > res_end, "Physics auditor section must sit strictly outside <!-- results: start/end -->"
    assert readme_txt[aud_start:aud_end] == ra.render_readme_audit_section(main_res, wide_res)

    dockerfile_txt = (ROOT / "Dockerfile").read_text()
    assert "eval/audit_results.json" in dockerfile_txt
    assert "eval/audit_wide.json" in dockerfile_txt


def test_eval_audit_api_endpoints_serve_json_and_404_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_m, raw_m = asyncio.run(_asgi_request("GET", "/api/eval/audit"))
    assert code_m == 200
    assert json.loads(raw_m)["n_videos_audited"] == 105

    code_w, raw_w = asyncio.run(_asgi_request("GET", "/api/eval/audit/wide"))
    assert code_w == 200
    assert json.loads(raw_w)["n_videos_audited"] == 300

    monkeypatch.setattr(api, "ROOT", tmp_path)
    code_404_m, _ = asyncio.run(_asgi_request("GET", "/api/eval/audit"))
    code_404_w, _ = asyncio.run(_asgi_request("GET", "/api/eval/audit/wide"))
    assert code_404_m == 404
    assert code_404_w == 404
