# Foresight — interface contract (all agents)

Four agents build in parallel. Each owns ONE directory and codes against this file. If something here is wrong or missing, tell the user; do not silently change another agent's interface.

| Owner | Directory | Builds |
|---|---|---|
| Claude Code | `backend/foresight/`, `phase0/`, `CONTRACT.md`, `README.md` | ingest, perceive (Hydrogen), track (CSRT), scale, scene assembly (background, sprites, keyframes), orchestrator `predict_clip`, const-velocity baseline, FastAPI + SSE |
| Physics agent | `backend/physics/` | pymunk world, fit, simulate, render, Decision 1 verify, Carbon repair, Managed Agent batch runner |
| Frontend agent | `frontend/` | Vite + React + Tailwind demo (phone capture, progress, 3 synced panels) |
| Eval agent | `eval/` | Physics-IQ subset eval: 4 systems, official metrics, results.json, chart, FAILURES.md |

Shared files: `PROGRESS.md` (edit only your own `##` section), `CLAUDE.md` (rules; read-only for others), `.gitignore` (root: Claude only; put your own ignores in `<your dir>/.gitignore`).

Repo facts every agent needs:
- Python env: `.venv` at repo root (`source .venv/bin/activate`), Python 3.12, google-genai 3.6.1, opencv-contrib, pymunk, scipy, fastapi. Add packages with `uv pip install <pkg>` and tell the user.
- Run Python from the repo root with `PYTHONPATH=backend` so `import foresight` and `import physics` work.
- Model IDs live ONLY in `backend/foresight/config.py`. All Gemini calls go through `foresight.gemini.call(...)` (retry/backoff on 503, 429 raises `SpendCapHit` → stop and tell the user, JSONL log in `logs/`). Preview models (`gemini-decision-1`, `gemini-omni-1.2-flash`, `gemini-robotics-er-2-preview`, the Managed Agent) need the v1beta client: use `foresight.gemini.client_for(model)`.
- Verified API shapes (google-genai 3.6.1): `client.interactions.create(model=..., input=[...], generation_config={"thinking_level": "low"|"medium"|...}, response_format={"type": "text", "mime_type": "application/json", "schema": Model.model_json_schema()})`, read `resp.output_text` / `resp.output_video.data` (base64). Video part: `{"type": "video", "uri": f.uri, "mime_type": f.mime_type, "processing": "agentic"}` (or `{"type":"static","fps":N}`). Image part: `{"type": "image", "data": <base64>, "mime_type": "image/jpeg"}`. Decisions: `beta_client.decisions.create(model=..., input=..., questions={key: Question})` — confirm the Question shape in the SDK (`google/genai/_gaos/types/decisions/`) or the docs MCP (`python phase0/docs_mcp.py search "..."`) before use.
- Video I/O: `phase0/vidio.py` → `write_mp4(path, frames_bgr, fps)` writes H.264 with an exact frame count (120 frames @ 24 FPS → ffprobe duration 5.000 s, verified); `probe(path)` returns duration/fps/frames/size.
- Benchmark data (gitignored, never commit): `data/physics-IQ-benchmark-verified/` (HF, gated — user must accept terms) or `data/physics-IQ-benchmark/` (public original GT, same videos). Fetch with `python phase0/physiq_data.py <scenes...> --source hf|gcs --fps 24` (add `--fps 30` for the 30 FPS conditioning clips).
- Benchmark FPS is **24**. The Physics-IQ 24 FPS files are frame-dropped from 30 FPS (per-frame displacement alternates ~1x/2x), so tracks carry real timestamps from the **30 FPS** conditioning clip when available.

---------------------------------------------------------------------------------------------------

## 1. Scene JSON ("paths JSON") — `backend/foresight` → `backend/physics`

Produced by `foresight.scene.build_scene(...)`, written to `<work_dir>/scene.json`, passed to `physics.predict(scene)`.
**All pixel coordinates are in the ORIGINAL input resolution, origin top-left, y DOWN.** Physics flips to y-up internally and flips back on output.

```jsonc
{
  "clip_id": "0023_perspective-center_trimmed-ball-rolls-off",
  "work_dir": "/abs/path/outputs/<job_id>",      // physics writes all outputs here
  "orig_size": [3840, 2160],                      // [W, H]
  "fps_out": 24,                                  // prediction FPS (benchmark FPS)
  "t_cond_end_s": 3.0,                            // end of the conditioning window (= n_cond_frames / fps_cond)
  "pred_seconds": 5.0,                            // prediction covers t in [t_cond_end_s, t_cond_end_s + 5.0)
                                                  // output frame k (0..119) is at t = t_cond_end_s + k / fps_out
  "px_per_m": 3700.0,                             // pixels per metre in the plane of motion
  "px_per_m_source": "size",                      // "size" | "gravity" | "size+gravity"
  "gravity_m_s2": 9.81,                           // do NOT fit gravity; use this
  "motion_type_hint": "roll",                     // Hydrogen: roll | slide | fall | bounce | other
  "objects": [                                    // dynamic bodies (moving objects only)
    {
      "id": "obj0",
      "name": "grey tennis ball",
      "shape": "circle",                          // "circle" | "box"
      "radius_px": 124.0,                         // circle only
      "w_px": null, "h_px": null,                 // box only
      "est_real_size_cm": 6.7,
      "track": [                                  // ONLY frames where tracking is trusted; centre of the object
        {"t_s": 2.500, "x": 288.0, "y": 1744.0}
      ],
      "visible_at_end": true,                     // visible in the last conditioning frame
      "sprite_png": "/abs/.../obj0.png"           // RGBA crop, orig resolution, object centre = image centre
    }
  ],
  "static_objects": [                             // large static things the body can hit (become static pymunk shapes)
    {"name": "black pipe", "shape": "box", "bbox": [0, 1520, 260, 1905]}   // [x0, y0, x1, y1]
  ],
  "surfaces": [                                   // support lines (top edges); static segments
    {"name": "wooden table", "kind": "table", "p0": [0.0, 1875.0], "p1": [3840.0, 1875.0]}
  ],                                              // kind: table | floor | ramp | wall | container | other
  "background_png": "/abs/.../background.png",   // last conditioning frame with moving objects inpainted (orig res)
  "last_frame_png": "/abs/.../last_frame.png",   // raw last conditioning frame (orig res)
  "keyframes": [                                  // conditioning frames for Decision 1 verification (orig res PNG)
    {"t_s": 2.542, "png": "/abs/.../kf_061.png"}
  ]
}
```

Notes the physics agent MUST handle:
- Tracks are short: in Physics-IQ the object usually moves only in the last 5–14 conditioning frames (0.2–0.6 s). Use strong priors and bounds; gravity is fixed; scale is given.
- Friction / elasticity / damping are often **not identifiable** from the conditioning window (no bounce, little deceleration). Fit them only when the data constrains them; otherwise keep priors (e.g. tennis ball on wood: elasticity ~0.6–0.75, friction ~0.5). The synthetic fitter test should require **position/velocity within 5%**, and friction/elasticity within 5% only on synthetic tracks that contain a bounce / measurable deceleration.
- pymunk has no rolling resistance: add a per-body velocity damping (fit or prior) so balls do not roll forever.
- Surfaces are finite segments: a ball that passes the end of the table must fall off (Physics-IQ `ball-rolls-off`).
- Objects can leave the frame; render nothing for them once off-screen.
- Verification frames: draw the **fitted** trajectory (the model's positions at the keyframe times inside the conditioning window), not the 5 s prediction.

## 2. Physics output — `physics.predict(scene, on_event=None) -> dict`

```python
from physics import predict                      # backend/physics/__init__.py re-exports predict
result = predict(scene: dict, on_event: Callable[[str, dict], None] | None = None) -> dict
```
`on_event(name, payload)` is how physics reports progress to the SSE stream (names in §4: `fit`, `verify`, `repair`, `render`). Must be safe to call with `on_event=None`.

```jsonc
{
  "prediction_mp4": "/abs/<work_dir>/foresight.mp4",   // exactly 5.000 s ±0.001 at fps_out, orig_size (or out_size)
  "fit_params": {"obj0": {"x0": 0.1, "y0": 0.5, "vx0": 1.9, "vy0": 0.0, "friction": 0.5, "elasticity": 0.7, "damping": 0.3}},
  "fit_error_px": 6.3,                                  // RMSE over fitted track points, orig px
  "motion_model": "roll",                               // model actually used after repairs
  "decision1_confidences": {"outline_on_object": 0.91, "motion": 0.84},
  "decision1_answers": {"outline_on_object": "yes", "motion": "roll"},
  "repairs_used": 0,
  "trajectory": [                                       // predicted, one entry per output frame (120), orig px, y DOWN
    {"t_s": 3.0, "objects": {"obj0": {"x": 2310.0, "y": 1752.0, "angle": 1.2, "visible": true}}}
  ],
  "fit_trajectory": [                                   // fitted model evaluated at the track timestamps
    {"t_s": 2.5, "objects": {"obj0": {"x": 290.0, "y": 1745.0, "angle": 0.0}}}
  ]
}
```

Also exported for baselines (const-velocity uses the same renderer so only the dynamics differ):
```python
from physics.render import render_trajectory
render_trajectory(scene: dict, trajectory: list[dict], out_path: str | Path, out_size: tuple[int, int] | None = None) -> Path
```

## 3. Orchestrator — `foresight.pipeline.predict_clip` (Claude) — used by the API and by eval

```python
from foresight.pipeline import predict_clip
meta = predict_clip(
    cond_video: Path,                  # 3.0 s conditioning clip (any FPS; Physics-IQ: the 24 FPS file)
    out_video: Path,                   # writes exactly 5.000 s at out_fps
    out_fps: int = 24,
    out_size: tuple[int, int] | None = None,   # default: input size; eval may pass (960, 540) for speed
    dynamics: str = "physics",         # "physics" (Foresight) | "const_velocity" (physics-free baseline)
    fit_video: Path | None = None,     # optional 30 FPS version of the same 3 s, used for tracking/fitting timestamps
    cache_dir: Path | None = None,     # if set, Gemini outputs are cached here per clip (dev only)
    reality_video: Path | None = None, # optional hidden 5 s; enables endpoint error
    on_event: Callable[[str, dict], None] | None = None,
) -> dict                              # == the SSE `result` payload (§4) without URLs
```

## 4. HTTP API (Claude) — `uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port 8000`

Frontend dev server proxies `/api` and `/outputs` to `http://localhost:8000` (so the phone only needs the frontend's HTTPS origin).

- `GET /api/info` → `{"lan_ip": "192.168.x.y", "models": {...config IDs...}, "omni_enabled": true}`
- `GET /api/samples` → `[{"id": "ball-rolls-off", "title": "Ball rolls off table", "thumb_url": "/outputs/samples/ball-rolls-off.jpg"}]` (Physics-IQ clips available locally)
- `POST /api/predict` multipart form:
  - `video`: file (mp4 / webm / mov; phone recording ~8 s) — **or** `sample_id`: string
  - `session`: optional string (links a phone upload to the laptop view)
  - `cond_seconds`: optional float, default 3.0. For an uploaded video, `[0, cond_seconds)` is conditioning and the next 5.0 s is the hidden reality.
  - `omni`: optional `"true"|"false"`, default `"true"`
  - → `202 {"job_id": "j_ab12cd", "events_url": "/api/jobs/j_ab12cd/events", "result_url": "/api/jobs/j_ab12cd"}`
- `GET /api/jobs?session=<s>` → `[{"job_id": "...", "created": "ISO8601", "status": "running"|"done"|"error"}]` newest first
- `GET /api/jobs/{job_id}` → the `result` payload once done, else `{"job_id", "status", "events_so_far": [...]}`
- `GET /api/jobs/{job_id}/events` → `text/event-stream` (EventSource-compatible). Replays past events on connect, then streams. Each message: `event: <name>` + `data: <json>`. Stream ends after `result` or `error`.
- Static files: `/outputs/<job_id>/{conditioning.mp4, reality.mp4, foresight.mp4, omni.mp4, overlay.mp4, result.json, events.jsonl}` — H.264 yuv420p, browser-playable.

SSE events (every payload also has `"t": seconds since job start`):

| event | payload |
|---|---|
| `stage` | `{"stage": "ingest"\|"perceive"\|"track"\|"scale"\|"fit"\|"simulate"\|"render"\|"verify"\|"repair"\|"omni", "status": "start"\|"done"\|"error", "message": str}` |
| `ingest` | `{"fps": 24, "cond_frames": 72, "reality_frames": 120, "width": 1920, "height": 1080, "conditioning_url": "/outputs/<id>/conditioning.mp4", "has_reality": true}` |
| `perception` | `{"model": "gemini-4-hydrogen", "latency_s": 12.5, "objects": [{"name", "shape", "est_real_size_cm", "moving", "bbox_last_frame"}], "support_surfaces": [{"name", "kind", "line"}], "motion_type": "roll", "motion_start_s": 2.4}` (bbox/line normalized 0–1000, [y, x] order) |
| `tracking` | `{"object": "grey tennis ball", "frames_ok": 12, "frames_total": 72, "points": [[t_s, x_px, y_px], ...]}` (orig px) |
| `scale` | `{"px_per_m": 3700.0, "source": "size", "gravity_check": null \| {"g_est": 9.6}}` |
| `fit` | `{"attempt": 0, "motion_model": "roll", "params": {...}, "fit_error_px": 6.3, "n_points": 12}` |
| `verify` | `{"model": "gemini-decision-1", "attempt": 0, "confidences": {"outline_on_object": 0.91, "motion": 0.84}, "answers": {...}, "passed": true}` |
| `repair` | `{"model": "gemini-4-carbon", "attempt": 1, "proposal": {"motion_model": "bounce", "reason": "..."}}` |
| `render` | `{"url": "/outputs/<id>/foresight.mp4", "frames": 120, "fps": 24, "duration_s": 5.0}` |
| `omni` | `{"status": "start"\|"done"\|"error"\|"skipped", "url": "/outputs/<id>/omni.mp4" \| null, "latency_s": 39.1, "error": null}` |
| `result` | final payload below |
| `error` | `{"stage": "track", "message": "no moving object found"}` |

`result` payload:
```jsonc
{
  "job_id": "j_ab12cd",
  "status": "done",
  "urls": {"conditioning": "...", "reality": "..." , "foresight": "...", "omni": "..." , "overlay": "..."},  // reality/omni may be null
  "metrics": {
    "latency_s_total": 31.2,
    "latency_s_by_stage": {"perceive": 12.5, "track": 0.2, "fit": 3.1, "render": 2.0, "verify": 4.0},
    "fit_error_px": 6.3,
    "decision1_confidences": {"outline_on_object": 0.91, "motion": 0.84},
    "repairs_used": 0,
    "endpoint_error_cm": {"foresight": 3.2, "omni": 41.0},   // null when no reality video or object not found
    "endpoint_error_px": {"foresight": 118.0, "omni": 1517.0}
  },
  "scene": {"objects": [...], "motion_type": "roll", "px_per_m": 3700.0},
  "trajectory": {"t_s": [...], "x": [...], "y": [...]}       // Foresight prediction of the main object, orig px
}
```
Endpoint error = distance between the object's position in the LAST visible frame of the reality video (CSRT-tracked) and the prediction at the same time, converted to cm with `px_per_m`.

## 5. Eval layout (eval agent)

- Clips: the 15 rigid-body scenes in `phase0/physiq_data.py` (`RIGID_SINGLE + RIGID_MULTI`), perspective **center**, 24 FPS.
- Prediction folders: `eval/preds/<system>/<id>_perspective-center_trimmed-<scene>.mp4`, systems `foresight`, `omni`, `const_velocity`, `static`.
- Scoring: `phase0/score_subset.py` → `score_folder(pred_dir, bench_dir, fps=24, out_dir)` (official Physics-IQ functions, one view).
- Outputs (committed): `eval/results.json`, `eval/results_table.md`, `eval/chart.png`, `eval/FAILURES.md`, `eval/results/*_metrics.json`. Cache (gitignored): `eval/cache/`.
