"""Command-line interface for Foresight (`foresight audit ...`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .audit import audit_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foresight",
        description="Foresight CLI: physicality auditing and scene simulation tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ap_audit = sub.add_parser(
        "audit",
        help="Audit a 5 s continuation MP4 against object permanence, fixed-time position error, and energy bounds.",
    )
    ap_audit.add_argument("cond_video", type=Path, help="3 s conditioning video MP4.")
    ap_audit.add_argument("pred_video", type=Path, help="5 s model continuation video MP4.")
    ap_audit.add_argument("--reality", type=Path, default=None, help="Optional 5 s ground-truth continuation MP4.")
    ap_audit.add_argument("--scene", type=Path, default=None, help="Optional pre-built scene.json (skips perception).")
    ap_audit.add_argument("--fit-video", type=Path, default=None, help="Optional higher-FPS conditioning clip for tracking.")
    ap_audit.add_argument("--cache-dir", type=Path, default=None, help="Optional perception cache directory.")
    ap_audit.add_argument("--out", type=Path, default=None, help="Directory to write audit.json and audit.html.")
    ap_audit.add_argument("--physics-iq-score", type=float, default=None, help="Precomputed Physics-IQ score (%%).")
    ap_audit.add_argument("--bench-dir", type=Path, default=None, help="Physics-IQ benchmark root for single-clip scoring.")
    ap_audit.add_argument("--label", type=str, default=None, help="Display label for the model/system in audit.html.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "audit":
        report = audit_video(
            args.cond_video,
            args.pred_video,
            reality_video=args.reality,
            scene=args.scene,
            out_dir=args.out,
            physics_iq_score_pct=args.physics_iq_score,
            fit_video=args.fit_video,
            cache_dir=args.cache_dir,
            bench_dir=args.bench_dir,
            label=args.label,
        )
        pos = report["position_error"]
        summary = {
            "clip_id": report["clip_id"],
            "label": report["label"],
            "passed": report["verdict"]["passed"],
            "permanence": {
                "status": report["permanence"]["status"],
                "vanished": report["permanence"]["vanished"],
                "duplicated": report["permanence"]["duplicated"],
                "max_size_change_ratio": report["permanence"]["max_size_change_ratio"],
            },
            "plausibility": {
                "plausible": report["plausibility"]["plausible"],
                "max_speed_m_s": report["plausibility"]["max_speed_m_s"],
                "max_bound_m_s": report["plausibility"]["max_bound_m_s"],
            },
            "position_error": (
                {
                    "raw_endpoint_error_cm": pos["raw_endpoint_error_cm"],
                    "penalized_endpoint_error_cm": pos["penalized_endpoint_error_cm"],
                    "mean_penalized_cm": pos["mean_penalized_cm"],
                }
                if pos is not None
                else None
            ),
            "physics_iq_score_pct": report["physics_iq_score_pct"],
            "out_dir": str(args.out.resolve()) if args.out is not None else None,
        }
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
