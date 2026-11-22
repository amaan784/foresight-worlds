# Failures, causes, and what we changed

Every failure we hit while building and evaluating Foresight, in the order we found it. Clips are Physics-IQ,
center camera. Failures from the clean eval run are listed at the end, generated from `eval/failures_auto.json`.

## Found while building (Nov 21)

| # | What happened | Cause | What we changed |
|---|---|---|---|
| 1 | `run_physics_iq.py` refused to score our subset | it asserts exactly 198 videos and all three cameras per scene | `phase0/score_subset.py` calls the same official functions (`generate_mask`, `process_view`, `IQTable`) on one camera |
| 2 | CSRT reported "tracked" on all 72 frames of `ball-rolls-off`, including frames where the ball was still inside the pipe | CSRT almost never reports loss; it latches onto background | a frame counts only if the tracked patch still looks like the object (appearance gate), forward and backward runs merged (`backend/foresight/track.py`) |
| 3 | The tracked ball moved ~35 px per frame, then ~67 px every third or fourth frame | the 24 FPS Physics-IQ files are frame-dropped from 30 FPS | the fit uses the 30 FPS conditioning clip and its real timestamps; prediction is still rendered at 24 FPS |
| 4 | Robotics-ER 2 as the tracker: 39 px median error against 20 px for CSRT (Phase 0, 0d) | its points jitter by about one frame of motion and sit ~20 px high | CSRT stays the tracker; ER 2 is not in the pipeline |
| 5 | The background under the ball was an X-shaped smear with the shadow left behind | `cv2.inpaint` over a large region | with a still camera, the region is copied from the first frame when the object was elsewhere; inpaint only as a fallback (`scene.py`) |
| 6 | One run of `ball-rolls-off` ended 560 cm away; the same clip had scored 7 cm earlier | a fresh Hydrogen call drew the table's **back** edge as the support line, 344 px above where the ball rolls, so the simulated ball started under the table | for rolling and sliding, the support line is snapped to a line through the tracked ball's bottom; if Hydrogen's line was off by more than two radii, its ends are not trusted either (`snap_support`) |
| 7 | After that fix the fit still diverged (149 px error, the ball launched downward at 6 m/s) | the first tracked point was the ball half inside the pipe, a wrong centre; the pipe was also a static obstacle overlapping the ball | track points where the object overlaps the object it emerges from are dropped, and that object is removed from the world (`drop_source_statics`) |
| 8 | Endpoint errors for the same prediction disagreed (83 cm against 7 cm) | the reality tracker "re-found" a still patch at the frame edge after losing the ball | a reality track ends at its first lost frame or when the object reaches the frame edge (`track_in_video`) |
| 9 | Omni's continuation of `ball-rolls-off` (Phase 0, 0c) | the video model made the ball vanish and a new one roll out of the pipe ~1 s later | not ours to fix: this is the failure Foresight is built to show |

## Found after the hackathon (Nov 22)

Found with `eval/offline_fit.py`, which reruns fit, simulation, render and the official score on the scene readings
cached by the Nov 21 clean run, so no Gemini call is needed. Those development runs use the same 15 center-view
clips as the reported eval, so they are not a held-out test; the left and right cameras and the five extra scenes
in the wider eval (`eval/run_wide.py`) were not used while developing.

| # | What happened | Cause | What we changed |
|---|---|---|---|
| 10 | roll-in-box: the simulated ball fell through the table and kept falling (161 m off in the eval) | the first two tracked points were the ball half inside its pipe; they looked like a bounce, so the ball started inside the zero-width table line and was pushed out downwards | tracks keep only free motion (a held prefix is cut at release; first/last points the smooth path misses by over 0.2 object sizes are dropped), and surfaces are solid below their line (`track.clean_track`, `world._build_surfaces`) |
| 11 | two-balls-pass: the nearer ball fell forever | the two balls roll at different distances from the camera; the table line through the far ball sits above the near one, so the near ball started "under the table" | rolling objects at the same depth share one line fitted through all their bottoms; another depth gets its own line (`scene.snap_support`) |
| 12 | roll-behind-box, roll-in-box: the ball bounced off a box it should pass behind or roll into | a side view flattens depth: a box in front of the path looked like a box in it | depth layers from where things touch the table: a static object whose bottom is on the ball's line blocks it, one lower in the image is in front and hides it, one higher is behind (`physics/layers.py`). On the 15 clips the two groups are 0.6 and 2.4+ ball radii from the line, so the cut is 1.2 |
| 13 | ball-rolls-on-glass broke after surfaces became solid | Hydrogen also drew the table's back edge, a line above the ball; solid below it, it swallowed the ball | a surface that runs above an object never touches it; collisions are now per object, not per layer |
| 14 | balls-collide: the two balls passed through each other | the second ball's line was 0.53 radii off a line fitted through the first ball alone, just over the "another depth" cut | the shared line is fitted through both balls (row 11), and objects join a depth group within 1.2 radii |
| 15 | two balls stuck together after colliding | the physics engine multiplies the two bounce coefficients (0.65 x 0.65 = 0.42) | body-body contacts use the geometric mean |
| 16 | two-balls-pass: the far ball slowed, stopped and rolled back into view | a table running away from the camera is drawn as a sloped line; the simulation treated the slope as an incline | tables and floors are level: gravity along their drawn slope is cancelled while a body touches them; ramps keep their slope |
| 17 | ball-train, ball-hits-duck: the ball bounced off resting balls and a rubber duck as if they were walls | static objects were always fixed | a static object standing on a moving object's line, under 30 times its mass, is simulated as a body at rest and drawn where it is knocked to |
| 18 | balls kept creeping across the frame for seconds after real ones stopped | only exponential damping, which never stops a ball | constant rolling resistance of 0.04 g for balls (a typical felt-ball-on-wood value, not tuned on the clips); the exponential damping prior for rolling went from 0.25/s to 0 |
| 19 | duck-falls-in-box: 82 pixels per metre, about 35 times too few | the gravity check fitted a parabola through the 0.6 s the duck was still held | the held prefix is cut before scale (row 10) |
| 20 | ball-and-block-fall: the ball and block bounced and rolled off the pillows | nothing knew a pillow is soft | Hydrogen now reports each surface as hard or soft and each object's mass; soft surfaces absorb landings |
| 21 | ball-in-basket, duck-falls-in-box: objects landed on top of the crate or box | a container's top line was a lid | a container is two walls and an open top, and hides what falls in |
| 22 | repairs took 5 to 7 minutes on two clips | Carbon at thinking level medium: 44 s median, up to 411 s a call (`logs/genai_calls_20261121.jsonl`) | repairs use thinking level low inside a 90 s budget for the whole check-and-repair loop |
| 23 | tried and reverted: keeping Hydrogen's slope for short tracks | a few points cannot measure a slope, but Hydrogen's line is often a table edge, not the rolling path | reverted; it hurt ball-ramp |
| 24 | the Nov 22 clean run first scored Foresight 59.9 while the same videos scored about 74 offline | the scorer caches each prediction's motion masks by file name and only made a mask when none existed, so the new videos were scored with masks of the Nov 21 videos (the Nov 21 numbers were not affected: each system's masks were made in that run, and the held last frame is the same video both days) | a mask is regenerated whenever its video is newer (`phase0/score_subset.py`); `eval/run_eval.py --rescore` scored the Nov 22 videos again, with no model calls |

| 25 | a live roll-behind-box run showed Foresight 1.11 m off and Genie 2.5 cm | the real ball went behind the container and the reality tracker latched onto the container's edge for the rest of the 5 s, so "where it really ended up" was a still patch any static prediction matched | the real track also ends when the object reaches something standing in front of it (with two radii of margin: perception's box is loose); `eval/remeasure_endpoints.py` recomputed the eval's endpoint errors from the saved paths, no model calls |

| 26 | ball-hits-nothing: Foresight's plastic ball lagged the real one, which barely slowed | every ball braked at 0.04 g, a felt tennis-ball value | Hydrogen now reports each object's material and balls brake by material (felt 0.04, rubber 0.015, hard plastic and wood 0.005, metal and glass 0.002; textbook orders of magnitude, not tuned). On the same fresh scene readings (offline): 73.75 with the fixed value, 74.47 by material; only ball-hits-nothing changed (58.6 to 69.4). The reported clean runs predate this change |
| 27 | the Managed Agent could no longer fit after the physics changes | its sandbox setup shipped world.py and fit.py only, and both now import layers.py | the setup ships all three modules and each object's mass; 15 of 15 eval scenes then fitted remotely, identical to the local fit (`eval/agent_batch.json`) |

A simulated object that gains energy (moves faster than its start speed plus free fall allows) now makes the run
fall back to constant velocity, and the run says so (`dynamics_used`, `fallback_reason`).

## Clean eval run

<!-- failures: start (written by eval/readme_results.py; edits here are lost) -->
No system failed on any clip.
<!-- failures: end -->
