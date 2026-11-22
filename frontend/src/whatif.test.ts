import assert from "node:assert/strict";
import test from "node:test";
import { buildWhatIfRequest, clientToScenePx, draftFromBaseline, velocityBound } from "./whatif.ts";

const BASE = {
  object_id: "obj0",
  name: "tennis ball",
  shape: "circle",
  start_px: [480, 540] as [number, number],
  vx0: 1.25,
  vy0: -0.1,
  speed_m_s: 1.254,
  friction: 0.5,
  bounce: 0.65,
};

test("clientToScenePx maps pointer coordinates into clamped scene pixels", () => {
  const rect = { left: 100, top: 50, width: 640, height: 360 };
  assert.deepEqual(clientToScenePx(420, 230, rect, [1920, 1080]), [960, 540]);
  assert.deepEqual(clientToScenePx(20, 999, rect, [1920, 1080]), [0, 1080]);
  assert.equal(clientToScenePx(100, 100, { left: 0, top: 0, width: 0, height: 360 }, [1920, 1080]), null);
});

test("buildWhatIfRequest omits unchanged fields and returns null at baseline", () => {
  const d0 = draftFromBaseline(BASE);
  assert.equal(buildWhatIfRequest("obj0", d0, BASE), null);

  const d1 = { ...d0, bounce: 0.9 };
  assert.deepEqual(buildWhatIfRequest("obj0", d1, BASE), { object_id: "obj0", bounce: 0.9 });

  const d2 = { ...d0, startPx: [600, 420] as [number, number], vx0: 2.5 };
  assert.deepEqual(buildWhatIfRequest("obj0", d2, BASE), {
    object_id: "obj0",
    start_px: [600, 420],
    vx0: 2.5,
  });
});

test("velocityBound scales cleanly with baseline speed", () => {
  assert.equal(velocityBound(0.2), 4);
  assert.equal(velocityBound(-2.6), 7);
  assert.equal(velocityBound(9.0), 12);
});
