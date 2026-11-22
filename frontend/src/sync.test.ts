import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceClock,
  clampVideoTime,
  DEFAULT_TIMELINE_S,
  HOLD_OFFSET_S,
  syncAllVideos,
  type SyncableVideo,
  timelineDuration,
} from "./sync.ts";

class FakeVideo implements SyncableVideo {
  src: string;
  currentTime = 0;
  duration: number;
  paused = true;
  ended = false;
  playbackRate = 1;
  readyState = 4;
  playCalls = 0;
  pauseCalls = 0;

  constructor(src: string, duration: number) {
    this.src = src;
    this.duration = duration;
  }

  play(): Promise<void> {
    this.paused = false;
    this.ended = false;
    this.playCalls += 1;
    return Promise.resolve();
  }

  pause(): void {
    this.paused = true;
    this.pauseCalls += 1;
  }

  /** Simulate native media advancement over `dtMs` when unpaused (no loop attribute). */
  tickMedia(dtMs: number): void {
    if (this.paused || this.duration <= 0) return;
    const next = this.currentTime + (dtMs / 1000) * this.playbackRate;
    if (next >= this.duration) {
      this.currentTime = this.duration;
      this.paused = true;
      this.ended = true;
    } else {
      this.currentTime = next;
    }
  }
}

test("short 1.0 s reality video holds at end while 5.0 s predictions play to 5.0 s and wrap cleanly", () => {
  const reality = new FakeVideo("/outputs/j_short/reality.mp4", 1.0);
  const foresight = new FakeVideo("/outputs/j_short/foresight.mp4", 5.0);
  const omni = new FakeVideo("/outputs/j_short/omni.mp4", 5.0);
  const videos = { reality, foresight, omni };

  const dur = timelineDuration([reality.duration, foresight.duration, omni.duration]);
  assert.equal(dur, 5.0);

  let t = 0;
  syncAllVideos(videos, t, { playing: true, rate: 1 });
  assert.equal(reality.paused, false);
  assert.equal(foresight.paused, false);

  // Advance in 50 ms frames up to 4.95 s (99 steps): foresight & omni must never reset at t = 1.0 s!
  let sawMidRollout = false;
  let sawNearFive = false;
  for (let i = 0; i < 99; i++) {
    reality.tickMedia(50);
    foresight.tickMedia(50);
    omni.tickMedia(50);
    const step = advanceClock(t, 50, true, 1, dur);
    t = step.t;
    assert.equal(step.wrapped, false);
    const res = syncAllVideos(videos, t, { playing: true, rate: 1, wrapped: step.wrapped });

    if (Math.abs(t - 0.5) < 0.001) {
      assert.equal(res.reality.heldAtEnd, false);
      assert.equal(reality.paused, false);
      assert.ok(Math.abs(foresight.currentTime - 0.5) < 0.02);
    }
    if (Math.abs(t - 2.5) < 0.001) {
      sawMidRollout = true;
      assert.equal(res.reality.heldAtEnd, true);
      assert.equal(reality.paused, true);
      assert.ok(Math.abs(reality.currentTime - (1.0 - HOLD_OFFSET_S)) < 0.02);
      assert.ok(Math.abs(foresight.currentTime - 2.5) < 0.02);
      assert.ok(Math.abs(omni.currentTime - 2.5) < 0.02);
    }
    if (Math.abs(t - 4.95) < 0.001) {
      sawNearFive = true;
      assert.equal(res.reality.heldAtEnd, true);
      assert.ok(Math.abs(foresight.currentTime - 4.95) < 0.02);
      assert.ok(Math.abs(omni.currentTime - 4.95) < 0.02);
    }
  }
  assert.equal(sawMidRollout, true);
  assert.equal(sawNearFive, true);

  // Next 100 ms crosses 5.0 s -> wraps to ~0.05 s and resumes reality alongside predictions.
  foresight.tickMedia(100);
  omni.tickMedia(100);
  const wrapStep = advanceClock(t, 100, true, 1, dur);
  assert.equal(wrapStep.wrapped, true);
  assert.ok(wrapStep.t < 0.1);
  const afterWrap = syncAllVideos(videos, wrapStep.t, { playing: true, rate: 1, wrapped: wrapStep.wrapped });
  assert.equal(afterWrap.reality.heldAtEnd, false);
  assert.equal(reality.paused, false);
  assert.ok(Math.abs(reality.currentTime - wrapStep.t) < 0.01);
  assert.ok(Math.abs(foresight.currentTime - wrapStep.t) < 0.01);
  assert.ok(Math.abs(omni.currentTime - wrapStep.t) < 0.01);
});

test("timeline stays 5.0 s even when all loaded videos are short (e.g. 1 s reality + 1 s overlay)", () => {
  const reality = new FakeVideo("/outputs/j_short/reality.mp4", 1.0);
  const overlay = new FakeVideo("/outputs/j_short/overlay.mp4", 1.0);
  assert.equal(timelineDuration([reality.duration, overlay.duration]), DEFAULT_TIMELINE_S);
  assert.equal(timelineDuration([NaN, 0]), DEFAULT_TIMELINE_S);
  assert.equal(clampVideoTime(3.4, 1.0), 1.0 - HOLD_OFFSET_S);
});

test("scrubbing past short reality duration clamps reality without snapping foresight back", () => {
  const reality = new FakeVideo("/outputs/j_short/reality.mp4", 1.0);
  const foresight = new FakeVideo("/outputs/j_short/foresight.mp4", 5.0);
  const videos = { reality, foresight };

  syncAllVideos(videos, 3.6, { playing: true, rate: 0.5, forceSeek: true });
  assert.equal(reality.paused, true);
  assert.equal(reality.currentTime, 1.0 - HOLD_OFFSET_S);
  assert.equal(foresight.paused, false);
  assert.equal(foresight.currentTime, 3.6);
  assert.equal(foresight.playbackRate, 0.5);

  // Scrub back into [0, 1.0): reality seeks to 0.4 s and resumes playing.
  syncAllVideos(videos, 0.4, { playing: true, rate: 0.5, forceSeek: true });
  assert.equal(reality.paused, false);
  assert.equal(reality.currentTime, 0.4);
  assert.equal(foresight.currentTime, 0.4);
});

test("late-arriving baseline video snaps immediately to current timeline position", () => {
  const foresight = new FakeVideo("/outputs/j_late/foresight.mp4", 5.0);
  foresight.currentTime = 2.8;
  foresight.paused = false;
  const genie = new FakeVideo("/outputs/j_late/genie.mp4", 5.0); // newly loaded at currentTime = 0

  syncAllVideos({ foresight, genie }, 2.8, { playing: true, rate: 1 });
  assert.equal(genie.currentTime, 2.8);
  assert.equal(genie.paused, false);
});
