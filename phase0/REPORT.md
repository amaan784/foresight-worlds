# Phase 0 go/no-go report (Nov 21 2026)

All numbers below are copied from files in `phase0/out/` and `logs/genai_calls_20261121.jsonl`.
Test clip for 0b/0c/0d: Physics-IQ `ball-rolls-off`, center perspective (the side view), 24 FPS conditioning (3.0 s, 72 frames).

| Test | Result | Evidence |
|---|---|---|
| 0a data + scoring | **PASS (mechanics)**, Verified GT pending HF access | `out/scores_gcs/*_metrics.json` |
| 0b Hydrogen perception + CSRT | **PASS** | `out/0b_ball-rolls-off/` |
| 0d Robotics-ER 2 vs CSRT | ER 2 works but is **not accurate enough**; CSRT stays primary | `out/0d_ball-rolls-off/compare.json` |
| 0c Omni continuation | **PASS** | `out/0c_ball-rolls-off/` |

## 0a — data + scoring
- `run_physics_iq.py` does **not** accept a partial set: `validate_generations()` asserts exactly 198 videos, and `process_videos()` raises unless all 3 perspectives exist per scenario.
- Importing the official functions works: `score_subset.py` uses `generate_mask`, `process_view`, and `IQTable` restricted to one view.
- Trivial prediction (last conditioning frame held 120 frames = 5.000 s @ 24 FPS), 3 center-view scenes, **original** Physics-IQ GT (public GCS bucket): Physics-IQ verified-formula score **33.9** (per scenario: hits-nothing 33.08, in-basket 28.98, rolls-off 39.66). Scoring took 23 s.
- Physics-IQ-Verified on HF is gated; the token authenticates but video files return 403 until the terms are accepted on the dataset page.
- Rigid-body scenes: **15 of 66** (12 single-object roll/fall/bounce + 3 two-ball). Center perspective is the side view, so 15 side-view clips.

## 0b — perception + tracking
- `gemini-4-hydrogen`, agentic video + first/last frames, `thinking_level: low`, native structured output: **12.48 s** latency.
- Output: grey tennis ball (circle, 6.7 cm, moving, hidden in frame 0), table line at y=868/1000, `motion_type: roll`, motion start 2.4 s.
- CSRT (appearance-gated, backward from the last frame): **12/12** fully-visible frames tracked (frames 60–71), partially-occluded frames 57–59 correctly rejected, **0.137 s** for the clip. Overlay: `out/0b_ball-rolls-off/overlay.mp4`.
- Independent check (background-subtraction centroid, no model): CSRT median error **20.1 px** at 960×540.

## 0d — Robotics-ER 2 vs CSRT
- `gemini-robotics-er-2-preview` needs the **v1beta** API (404 on v1). Video at static 24 FPS sampling, native structured output.
- Latency **8.54 s** (CSRT 0.137 s). 12 points over 0.458 s = **26.2 points/s** (CSRT 24/s).
- Median error vs CSRT **38.7 px** (1.17 ball radii); vs the independent reference ER 2 **39.1 px** vs CSRT **20.1 px**. Errors swing ±40 px in x frame-to-frame (≈ one frame of ball motion) and sit ~20 px high.
- Decision: **CSRT is the primary tracker**; ER 2 is not used for trajectory fitting.

## 0c — Omni baseline
- `gemini-omni-1.2-flash` (v1beta) extends an uploaded clip: **39.1 s** latency, returns input + 10 s (13.0 s, 1280×720, 24 FPS). Its first 72 frames match the input (mean abs diff 2.0/255), so the 5 s after t=3.0 s is extracted.
- Qualitatively: the ball disappears at the cut and a new ball emerges from the pipe ~1 s later and rolls slowly, while the real ball has already rolled off the table.
- Score on this one clip (original GT): Omni **9.17** vs static last frame **39.66**.

## Risks for Phase 1
1. The object moves only in the last 5–14 conditioning frames (0.2–0.6 s): fits must lean on gravity + size-based scale priors.
2. The 24 FPS files are frame-dropped from 30 FPS (per-frame displacement alternates ~35 px / ~67 px): fit on the 30 FPS conditioning video, render at 24 FPS.
3. The static last frame scores ~34 because the MSE component rewards no change; Foresight must beat that, not just Omni.
4. `gemini-decision-1` is not in `models.list()`; untested so far.
