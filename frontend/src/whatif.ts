// Pure geometry and request-diff helpers for interactive counterfactual ("what-if") rollouts.

export type WhatIfObjBaseline = {
  object_id: string;
  name: string;
  shape: string;
  start_px: [number, number];
  vx0: number;
  vy0: number;
  speed_m_s: number;
  friction: number;
  bounce: number;
};

export type WhatIfRequest = {
  object_id?: string;
  start_px?: [number, number];
  vx0?: number;
  vy0?: number;
  friction?: number;
  bounce?: number;
};

export type WhatIfDraft = {
  startPx: [number, number];
  vx0: number;
  vy0: number;
  friction: number;
  bounce: number;
};

export type BoxRect = { left: number; top: number; width: number; height: number };

export function draftFromBaseline(b: WhatIfObjBaseline): WhatIfDraft {
  return {
    startPx: [b.start_px[0], b.start_px[1]],
    vx0: b.vx0,
    vy0: b.vy0,
    friction: b.friction,
    bounce: b.bounce,
  };
}

export function clientToScenePx(
  clientX: number,
  clientY: number,
  rect: BoxRect,
  origSize: [number, number],
): [number, number] | null {
  if (rect.width <= 0 || rect.height <= 0) return null;
  const [w, h] = origSize;
  const x = Math.max(0, Math.min(w, ((clientX - rect.left) / rect.width) * w));
  const y = Math.max(0, Math.min(h, ((clientY - rect.top) / rect.height) * h));
  return [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
}

export function velocityBound(v0: number): number {
  return Math.min(12, Math.max(4, Math.ceil(Math.abs(v0) * 2.5)));
}

export function buildWhatIfRequest(
  objectId: string,
  draft: WhatIfDraft,
  base: WhatIfObjBaseline,
): WhatIfRequest | null {
  const req: WhatIfRequest = { object_id: objectId };
  let changed = false;
  if (Math.hypot(draft.startPx[0] - base.start_px[0], draft.startPx[1] - base.start_px[1]) > 0.25) {
    req.start_px = [draft.startPx[0], draft.startPx[1]];
    changed = true;
  }
  if (Math.abs(draft.vx0 - base.vx0) > 1e-3) {
    req.vx0 = draft.vx0;
    changed = true;
  }
  if (Math.abs(draft.vy0 - base.vy0) > 1e-3) {
    req.vy0 = draft.vy0;
    changed = true;
  }
  if (Math.abs(draft.friction - base.friction) > 1e-3) {
    req.friction = draft.friction;
    changed = true;
  }
  if (Math.abs(draft.bounce - base.bounce) > 1e-3) {
    req.bounce = draft.bounce;
    changed = true;
  }
  return changed ? req : null;
}
