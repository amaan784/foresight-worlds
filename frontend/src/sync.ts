// Shared 5 s playback timeline for SyncedPanels (Task C3).
// Decouples synchronized reveal playback from short uploaded reality/overlay clips (< 5 s)
// so predictions always play through the full 5.0 s window while shorter clips hold on their last frame.

export const DEFAULT_TIMELINE_S = 5;
export const END_EPSILON_S = 0.02;
export const HOLD_OFFSET_S = 0.01;
export const DRIFT_TOLERANCE_S = 0.06;

export interface SyncableVideo {
  src: string;
  currentTime: number;
  duration: number;
  paused: boolean;
  ended?: boolean;
  playbackRate: number;
  readyState?: number;
  play(): Promise<void> | void;
  pause(): void;
}

export interface VideoSyncResult {
  heldAtEnd: boolean;
  clipDuration: number | null;
  targetTime: number;
}

const pendingPlays = new WeakSet<SyncableVideo>();

function safePause(v: SyncableVideo): void {
  pendingPlays.delete(v);
  if (!v.paused) v.pause();
}

function safePlay(v: SyncableVideo): void {
  if (!v.paused || pendingPlays.has(v)) return;
  if (typeof v.readyState === "number" && v.readyState < 1) return;
  pendingPlays.add(v);
  try {
    const res = v.play();
    if (res && typeof (res as Promise<void>).finally === "function") {
      (res as Promise<void>).catch(() => {}).finally(() => pendingPlays.delete(v));
    } else {
      pendingPlays.delete(v);
    }
  } catch {
    pendingPlays.delete(v);
  }
}

/** Target timeline duration in seconds: at least 5.0 s, or longer if a loaded video exceeds 5.0 s. */
export function timelineDuration(durations: readonly number[], minSeconds = DEFAULT_TIMELINE_S): number {
  let max = minSeconds;
  for (const d of durations) {
    if (Number.isFinite(d) && d > max) max = d;
  }
  return max;
}

/** Clamp a timeline position `t` into a video's playable range `[0, duration - HOLD_OFFSET_S]`. */
export function clampVideoTime(t: number, clipDuration: number): number {
  const safeT = Math.max(0, t);
  if (!Number.isFinite(clipDuration) || clipDuration <= 0) return safeT;
  return Math.min(safeT, Math.max(0, clipDuration - HOLD_OFFSET_S));
}

/** Advance the monotonic timeline clock by `dtMs` milliseconds, wrapping cleanly at `duration`. */
export function advanceClock(
  prevT: number,
  dtMs: number,
  playing: boolean,
  rate: number,
  duration = DEFAULT_TIMELINE_S,
): { t: number; wrapped: boolean } {
  const dur = duration > 0 ? duration : DEFAULT_TIMELINE_S;
  const clampedPrev = Math.min(Math.max(0, prevT), dur);
  if (!playing || !Number.isFinite(dtMs) || dtMs <= 0) {
    return { t: clampedPrev, wrapped: false };
  }
  const cappedDtMs = Math.min(dtMs, 1000);
  const next = clampedPrev + (cappedDtMs / 1000) * rate;
  if (next >= dur) {
    return { t: next % dur, wrapped: true };
  }
  return { t: next, wrapped: false };
}

/** Synchronize one video element to the shared timeline position `clockT`. */
export function syncVideo(
  v: SyncableVideo,
  clockT: number,
  opts: { playing: boolean; rate: number; wrapped?: boolean; forceSeek?: boolean },
): VideoSyncResult {
  if (v.playbackRate !== opts.rate) v.playbackRate = opts.rate;
  const hasDur = Number.isFinite(v.duration) && v.duration > 0;
  const clipDuration = hasDur ? v.duration : null;

  // Clip is shorter than the current timeline position: pause and hold its final frame.
  if (hasDur && clockT >= v.duration - END_EPSILON_S) {
    const holdT = clampVideoTime(clockT, v.duration);
    safePause(v);
    if (Math.abs(v.currentTime - holdT) > END_EPSILON_S || v.ended) {
      v.currentTime = holdT;
    }
    return { heldAtEnd: true, clipDuration, targetTime: holdT };
  }

  const target = Math.max(0, clockT);
  if (opts.forceSeek || opts.wrapped || Math.abs(v.currentTime - target) > DRIFT_TOLERANCE_S) {
    v.currentTime = target;
  }
  if (opts.playing) {
    safePlay(v);
  } else {
    safePause(v);
  }
  return { heldAtEnd: false, clipDuration, targetTime: target };
}

/** Synchronize all active video elements to `clockT` and return per-key status. */
export function syncAllVideos(
  videosByKey: Readonly<Record<string, SyncableVideo | null | undefined>>,
  clockT: number,
  opts: { playing: boolean; rate: number; wrapped?: boolean; forceSeek?: boolean },
): Record<string, VideoSyncResult> {
  const out: Record<string, VideoSyncResult> = {};
  for (const [key, v] of Object.entries(videosByKey)) {
    if (!v || !v.src) continue;
    out[key] = syncVideo(v, clockT, opts);
  }
  return out;
}
