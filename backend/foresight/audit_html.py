"""Self-contained editorial HTML + inline SVG report renderer for `foresight audit`."""
from __future__ import annotations

import html
from typing import Any


def _svg_trajectory_plot(report: dict[str, Any]) -> str:
    W, H = report["orig_size"]
    vw, vh = 640.0, max(240.0, round(640.0 * H / max(W, 1), 1))
    sx, sy = vw / max(W, 1), vh / max(H, 1)
    tr = report["tracks"]
    parts = [f'<svg viewBox="0 0 {vw:.0f} {vh:.0f}" width="100%" role="img" aria-label="2D trajectory comparison" '
             'style="background:#17191e;border-radius:12px;display:block;border:1px solid #d5d6d1">']
    for s in tr.get("surfaces") or []:
        (x0, y0), (x1, y1) = s["p0"], s["p1"]
        dash = ' stroke-dasharray="5 4"' if s.get("kind") == "container" else ""
        parts.append(f'<line x1="{x0 * sx:.1f}" y1="{y0 * sy:.1f}" x2="{x1 * sx:.1f}" y2="{y1 * sy:.1f}" '
                     f'stroke="#6e7482" stroke-width="2.5"{dash}/>')
    for x0, y0, x1, y1 in tr.get("occluders") or []:
        parts.append(f'<rect x="{max(0.0, x0 * sx):.1f}" y="{max(0.0, y0 * sy):.1f}" width="{max(2.0, (x1 - x0) * sx):.1f}" '
                     f'height="{max(2.0, (y1 - y0) * sy):.1f}" fill="#7b3fa8" fill-opacity="0.22" stroke="#9d6bc4" '
                     'stroke-width="1.5" stroke-dasharray="4 3"/>')

    def _poly(pts: list[dict[str, Any]], color: str, width: float, dash: str = "") -> None:
        if len(pts) >= 2:
            coords = " ".join(f"{p['x'] * sx:.1f},{p['y'] * sy:.1f}" for p in pts)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}"'
                         f'{f" stroke-dasharray={chr(34)}{dash}{chr(34)}" if dash else ""} points="{coords}"/>')

    _poly(tr.get("conditioning") or [], "#9aa0ad", 2.0, "4 3")
    if tr.get("reality"):
        _poly(tr["reality"], "#25a567", 2.8)
    _poly(tr.get("prediction") or [], "#4c7dff", 2.8)

    perm = report["permanence"]
    main_obj = next(iter(perm["per_object"].values()), None)
    if perm["vanished"] and main_obj and main_obj.get("last_seen_px"):
        vx, vy = main_obj["last_seen_px"][0] * sx, main_obj["last_seen_px"][1] * sy
        parts.append(f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="7" fill="none" stroke="#ff5a52" stroke-width="2.5"/>'
                     f'<line x1="{vx - 5:.1f}" y1="{vy - 5:.1f}" x2="{vx + 5:.1f}" y2="{vy + 5:.1f}" stroke="#ff5a52" stroke-width="2.5"/>'
                     f'<line x1="{vx - 5:.1f}" y1="{vy + 5:.1f}" x2="{vx + 5:.1f}" y2="{vy - 5:.1f}" stroke="#ff5a52" stroke-width="2.5"/>')
    if perm["duplicated"] and main_obj and main_obj.get("duplicate_pos_px"):
        dx, dy = main_obj["duplicate_pos_px"][0] * sx, main_obj["duplicate_pos_px"][1] * sy
        parts.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="8" fill="#ffb020" fill-opacity="0.35" stroke="#ffb020" stroke-width="2.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_speed_plot(report: dict[str, Any]) -> str:
    series = report["plausibility"].get("speed_series") or []
    vw, vh, pad_l, pad_r, pad_t, pad_b = 640.0, 220.0, 46.0, 18.0, 18.0, 32.0
    pw, ph = vw - pad_l - pad_r, vh - pad_t - pad_b
    dur = max(float(report.get("pred_duration_s") or 5.0), 0.5)
    ymax = max([1.5] + [float(p["speed_m_s"]) for p in series] + [float(p["bound_m_s"]) for p in series]) * 1.12
    xy = lambda dt, val: (pad_l + (dt / dur) * pw, pad_t + ph - (val / ymax) * ph)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {vw:.0f} {vh:.0f}" width="100%" role="img" aria-label="Speed vs free-fall energy bound" '
        'style="background:#ffffff;border-radius:12px;display:block;border:1px solid #d5d6d1">',
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{pad_l + pw}" y2="{pad_t + ph}" stroke="#d5d6d1"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#d5d6d1"/>',
        f'<text x="{pad_l}" y="{vh - 8}" fill="#5f6470" font-size="11" font-family="monospace">0.0 s</text>',
        f'<text x="{pad_l + pw - 34}" y="{vh - 8}" fill="#5f6470" font-size="11" font-family="monospace">+{dur:.1f} s</text>',
        f'<text x="6" y="{pad_t + 10}" fill="#5f6470" font-size="11" font-family="monospace">{ymax:.1f} m/s</text>',
    ]
    if len(series) >= 2:
        b_pts = " ".join(f"{xy(p['dt_s'], p['bound_m_s'])[0]:.1f},{xy(p['dt_s'], p['bound_m_s'])[1]:.1f}" for p in series)
        s_pts = " ".join(f"{xy(p['dt_s'], p['speed_m_s'])[0]:.1f},{xy(p['dt_s'], p['speed_m_s'])[1]:.1f}" for p in series)
        s_col = "#2350d8" if report["plausibility"]["plausible"] else "#b3261e"
        parts.append(f'<polyline fill="none" stroke="#b86e00" stroke-width="2" stroke-dasharray="5 4" points="{b_pts}"/>')
        parts.append(f'<polyline fill="none" stroke="{s_col}" stroke-width="2.4" points="{s_pts}"/>')
    else:
        parts.append(f'<text x="{vw / 2 - 75:.0f}" y="{vh / 2:.0f}" fill="#5f6470" font-size="13">Insufficient track points</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_audit_html(report: dict[str, Any]) -> str:
    """Render a self-contained editorial HTML report for a single `audit_video` result."""
    verdict, perm, plaus, pos = report["verdict"], report["permanence"], report["plausibility"], report["position_error"]
    piq = report.get("physics_iq_score_pct")

    def _badge(ok: bool, good_txt: str, bad_txt: str) -> str:
        bg, fg, txt = ("#e6f4ec", "#1f7a4d", good_txt) if ok else ("#fce8e6", "#b3261e", bad_txt)
        return f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;background:{bg};color:{fg};font:600 12px/1.4 sans-serif">{html.escape(txt)}</span>'

    rows_html = ""
    if pos is not None:
        all_rows = list(pos["by_time"])
        if pos.get("at_last_real_visible"):
            all_rows.append({**pos["at_last_real_visible"], "_label": f"+{pos['at_last_real_visible']['dt_s']:.2f} s (last real visible)"})
        for r in all_rows:
            lbl = r.get("_label") or f"+{r['dt_s']:.1f} s"
            raw_s = f"{r['raw_err_cm']:.1f} cm" if r["raw_err_cm"] is not None else "—"
            pen_s = f"{r['penalized_err_cm']:.1f} cm" if r["penalized_err_cm"] is not None else (
                f"{r['lingered_err_cm']:.1f} cm (lingered)" if r.get("lingered_err_cm") is not None else "—")
            rows_html += (f"<tr><td><code>{html.escape(lbl)}</code></td><td><code>{html.escape(str(r['status']))}</code></td>"
                          f"<td>{html.escape(raw_s)}</td><td><strong>{html.escape(pen_s)}</strong></td></tr>")

    reasons_li = "".join(f"<li>{html.escape(r)}</li>" for r in verdict["reasons"]) or "<li>No physicality violations detected.</li>"
    piq_str = f"{piq:.2f}%" if piq is not None else "—"
    mean_pen = f"{pos['mean_penalized_cm']:.1f} cm" if (pos and pos["mean_penalized_cm"] is not None) else "—"
    raw_ep = f"{pos['raw_endpoint_error_cm']:.1f} cm" if (pos and pos["raw_endpoint_error_cm"] is not None) else "—"
    pos_section = (
        f'<section class="card" style="margin-top:18px"><h2 style="margin:0 0 8px;font-size:18px">Fixed-Time Position Error vs. Reality</h2>'
        f'<p style="margin:0 0 12px;color:#5f6470;font-size:14px">Stopped-track raw endpoint error: <strong>{raw_ep}</strong> · '
        f'Penalized mean error across visible real timestamps: <strong>{mean_pen}</strong> (vanish floor: {pos["off_course_floor_cm"]:.1f} cm)</p>'
        f'<table><thead><tr><th>Offset</th><th>Track Status</th><th>Raw Error</th><th>Penalized Error</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table></section>'
        if pos is not None else
        '<section class="card" style="margin-top:18px"><p style="margin:0;color:#5f6470">No ground-truth 5 s clip provided; position error table skipped.</p></section>'
    )
    title_lbl = f" · {html.escape(str(report['label']))}" if report.get("label") else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Foresight Audit — {html.escape(str(report['clip_id']))}</title>
<style>
  body {{ margin:0; padding:28px 20px 48px; background:#f2f2ef; color:#15171c; font:15px/1.5 "IBM Plex Sans",-apple-system,sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; }}
  .card {{ background:#ffffff; border:1px solid #d5d6d1; border-radius:14px; padding:18px 20px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-top:16px; }}
  .kpi {{ font:700 28px/1.1 "Bricolage Grotesque",sans-serif; margin-top:6px; }}
  .lbl {{ color:#5f6470; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e6e7e2; }}
  th {{ color:#5f6470; font-weight:600; font-size:12px; text-transform:uppercase; }}
  code {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:13px; }}
</style></head><body><div class="wrap">
  <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:12px;margin-bottom:16px">
    <div><div class="lbl">Foresight Physicality Auditor</div><h1 style="margin:4px 0 0;font-size:28px">{html.escape(str(report['clip_id']))}{title_lbl}</h1></div>
    <div>{_badge(verdict['passed'], 'AUDIT PASS', 'AUDIT VIOLATION')}</div>
  </div>
  <div class="grid4">
    <div class="card"><div class="lbl">Object Permanence</div><div class="kpi">{html.escape(perm['status'])}</div>
      <div style="margin-top:8px">{_badge(not perm['vanished'], 'No vanish', 'Vanished')} {_badge(not perm['duplicated'], 'No duplicate', 'Duplicated')} {_badge(not perm['size_unstable'], f"Size {perm['max_size_change_ratio']:.2f}x", f"Size {perm['max_size_change_ratio']:.2f}x")}</div></div>
    <div class="card"><div class="lbl">Energy Plausibility</div><div class="kpi">{plaus['max_speed_m_s']:.2f} m/s</div>
      <div style="margin-top:8px;color:#5f6470;font-size:13px">Bound {plaus['max_bound_m_s']:.2f} m/s (v0={plaus['v0_cond_m_s']:.2f} m/s) · {_badge(plaus['plausible'], 'Plausible', 'Violated')}</div></div>
    <div class="card"><div class="lbl">Penalized Position Error</div><div class="kpi">{html.escape(mean_pen)}</div>
      <div style="margin-top:8px;color:#5f6470;font-size:13px">Stopped-track endpoint: {html.escape(raw_ep)}</div></div>
    <div class="card"><div class="lbl">Physics-IQ Score</div><div class="kpi">{html.escape(piq_str)}</div>
      <div style="margin-top:8px;color:#5f6470;font-size:13px">Pixel mask IoU / MSE metric</div></div>
  </div>
  <section class="card" style="margin-top:16px"><div class="lbl">Audit Findings</div><ul style="margin:8px 0 0;padding-left:20px">{reasons_li}</ul></section>
  <div class="grid2" style="margin-top:16px">
    <section class="card"><h2 style="margin:0 0 6px;font-size:17px">2D Trajectory &amp; Occlusion Layers</h2>
      <p style="margin:0 0 10px;color:#5f6470;font-size:13px">Dashed grey = 3 s conditioning · Green = reality · Blue = prediction · Purple = foreground occluder · Red × = vanish · Amber ○ = duplicate</p>{_svg_trajectory_plot(report)}</section>
    <section class="card"><h2 style="margin:0 0 6px;font-size:17px">Speed vs. Free-Fall Energy Bound</h2>
      <p style="margin:0 0 10px;color:#5f6470;font-size:13px">Solid = tracked speed · Dashed amber = 1.3·sqrt(v0² + 2g·Δh) + 0.3 m/s bound</p>{_svg_speed_plot(report)}</section>
  </div>
  {pos_section}
</div></body></html>
"""
