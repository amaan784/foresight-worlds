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

## Clean eval run

<!-- failures: start (written by eval/readme_results.py; edits here are lost) -->
Not run yet.
<!-- failures: end -->
