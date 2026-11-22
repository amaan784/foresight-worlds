"""Pymunk 2D rigid-body world construction from Foresight scene JSON.

Coordinate convention:
- Scene JSON uses original-resolution pixels with origin at top-left, y DOWN.
- Internally, this module flips to y-UP metres with origin at bottom-left:
    x_m = x_px / px_per_m
    y_m = (H - y_px) / px_per_m
- Gravity is fixed at (0, -gravity_m_s2) and is never fitted.
- Static surfaces and static boxes have friction=1.0 and elasticity=1.0 so that
  Chipmunk2D's multiplicative contact combination makes each dynamic body's
  friction and elasticity equal to the effective physical coefficients.
- Pymunk lacks rolling resistance; per-body tangential velocity damping is
  applied during support contact so rolling balls decelerate realistically
  without artificially slowing vertical free-fall once they leave a surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pymunk

from .layers import assign_layers

_BODY = 1  # collision type of every simulated body
_LEVEL = 2  # collision type of tables and floors: level in 3D whatever their slope in the image
_SOFT = 3  # soft surfaces (cushions, bedding): level too, and they absorb motion
SOFT_DAMPING = 8.0  # s^-1 while touching a soft surface: a landed object stops within a fraction of a second
# Rolling resistance coefficient for balls (braking = 0.04 g): a typical value for a felt ball on wood, set from
# physics and not tuned on the eval clips. Not fitted: 0.1-0.4 s of conditioning cannot separate it from damping.
ROLLING_RESISTANCE = 0.04


def _mean_coefficients(arb: pymunk.Arbiter, space: pymunk.Space, data: Any) -> None:
    a, b = arb.shapes
    arb.restitution = math.sqrt(max(0.0, a.elasticity) * max(0.0, b.elasticity))
    arb.friction = math.sqrt(max(0.0, a.friction) * max(0.0, b.friction))

DEFAULT_SUBSTEP_DT = 1.0 / 480.0
DEFAULT_SOLVER_ITERATIONS = 20

# Priors indexed by motion_type_hint: (friction, elasticity, damping_s^-1). Rolling balls already brake through
# ROLLING_RESISTANCE, so their extra exponential damping starts at 0 (the fitter raises it when a track shows it).
_MOTION_PRIORS: dict[str, dict[str, float]] = {
    "roll": {"friction": 0.50, "elasticity": 0.65, "damping": 0.0},
    "slide": {"friction": 0.35, "elasticity": 0.30, "damping": 0.00},
    "fall": {"friction": 0.50, "elasticity": 0.65, "damping": 0.02},
    "bounce": {"friction": 0.50, "elasticity": 0.70, "damping": 0.05},
    "other": {"friction": 0.50, "elasticity": 0.60, "damping": 0.0},
}


def default_priors(motion_model: str = "roll", shape: str = "circle") -> dict[str, float]:
    """Return prior (friction, elasticity, damping) for a motion model and body shape."""
    base = dict(_MOTION_PRIORS.get(motion_model, _MOTION_PRIORS["other"]))
    if shape == "box" and motion_model not in ("bounce",):
        base["elasticity"] = min(base["elasticity"], 0.35)
    return base


@dataclass
class CoordTransform:
    """Bidirectional transform between y-down pixels and y-up metres."""

    width_px: int
    height_px: int
    px_per_m: float

    @classmethod
    def from_scene(cls, scene: dict[str, Any]) -> "CoordTransform":
        w, h = scene["orig_size"]
        ppm = float(scene["px_per_m"])
        if ppm <= 0:
            raise ValueError(f"px_per_m must be positive, got {ppm}")
        return cls(width_px=int(w), height_px=int(h), px_per_m=ppm)

    def px_to_m(self, x_px: float, y_px: float) -> tuple[float, float]:
        return (float(x_px) / self.px_per_m, (self.height_px - float(y_px)) / self.px_per_m)

    def m_to_px(self, x_m: float, y_m: float) -> tuple[float, float]:
        return (float(x_m) * self.px_per_m, self.height_px - float(y_m) * self.px_per_m)

    def vec_px_to_m(self, vx_px: float, vy_px: float) -> tuple[float, float]:
        return (float(vx_px) / self.px_per_m, -float(vy_px) / self.px_per_m)

    def vec_m_to_px(self, vx_m: float, vy_m: float) -> tuple[float, float]:
        return (float(vx_m) * self.px_per_m, -float(vy_m) * self.px_per_m)

    def len_px_to_m(self, d_px: float) -> float:
        return float(d_px) / self.px_per_m

    def len_m_to_px(self, d_m: float) -> float:
        return float(d_m) * self.px_per_m

    def is_visible(self, x_px: float, y_px: float, extent_px: float = 0.0) -> bool:
        """True if an object centred at (x_px, y_px) with half-extent intersects the frame."""
        return (
            x_px + extent_px >= 0.0
            and x_px - extent_px <= self.width_px
            and y_px + extent_px >= 0.0
            and y_px - extent_px <= self.height_px
        )


@dataclass
class BodyParams:
    """Initial state and physical parameters of one dynamic body in y-up metres."""

    x0: float
    y0: float
    vx0: float
    vy0: float
    friction: float = 0.50
    elasticity: float = 0.65
    damping: float = 0.25
    angle0: float = 0.0
    omega0: float | None = None

    def to_dict(self) -> dict[str, float]:
        return {
            "x0": float(self.x0),
            "y0": float(self.y0),
            "vx0": float(self.vx0),
            "vy0": float(self.vy0),
            "friction": float(self.friction),
            "elasticity": float(self.elasticity),
            "damping": float(self.damping),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BodyParams":
        return cls(
            x0=float(d["x0"]),
            y0=float(d["y0"]),
            vx0=float(d["vx0"]),
            vy0=float(d["vy0"]),
            friction=float(d.get("friction", 0.50)),
            elasticity=float(d.get("elasticity", 0.65)),
            damping=float(d.get("damping", 0.25)),
            angle0=float(d.get("angle0", 0.0)),
            omega0=None if d.get("omega0") is None else float(d["omega0"]),
        )


def _obj_half_extent_px(obj: dict[str, Any]) -> float:
    if obj.get("shape") == "box":
        w = float(obj.get("w_px") or 0.0)
        h = float(obj.get("h_px") or 0.0)
        return 0.5 * math.hypot(w, h)
    return float(obj.get("radius_px") or 0.0)


def _obj_min_half_extent_px(obj: dict[str, Any]) -> float:
    if obj.get("shape") == "box":
        w = float(obj.get("w_px") or 0.0)
        h = float(obj.get("h_px") or 0.0)
        return 0.5 * min(w, h)
    return float(obj.get("radius_px") or 0.0)


def _static_box_overlaps_track(
    bbox: list[float],
    objects: list[dict[str, Any]],
    penetration_frac: float = 0.35,
) -> bool:
    """Return True if a static box deeply overlaps any dynamic object's observed track.

    For example, in Physics-IQ `ball-rolls-off` the ball emerges from a hollow pipe
    whose 2D bbox overlaps the first tracked ball positions. Treating that pipe as a
    solid box would violently eject the ball on step 0.
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bx0, bx1 = min(x0, x1), max(x0, x1)
    by0, by1 = min(y0, y1), max(y0, y1)
    for obj in objects:
        r = _obj_min_half_extent_px(obj)
        if r <= 0:
            continue
        thresh = penetration_frac * r
        for pt in obj.get("track", []):
            cx, cy = float(pt["x"]), float(pt["y"])
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                return True
            qx = min(max(cx, bx0), bx1)
            qy = min(max(cy, by0), by1)
            dist = math.hypot(cx - qx, cy - qy)
            if (r - dist) > thresh:
                return True
    return False


class PhysicsWorld:
    """Pymunk space built from a Foresight scene JSON and per-body parameters."""

    def __init__(
        self,
        scene: dict[str, Any],
        body_params: dict[str, BodyParams | dict[str, Any]],
        t_start_s: float = 0.0,
        motion_model: str | None = None,
        substep_dt: float = DEFAULT_SUBSTEP_DT,
    ) -> None:
        self.scene = scene
        self.coord = CoordTransform.from_scene(scene)
        self.motion_model = motion_model or scene.get("motion_type_hint", "roll")
        self.substep_dt = float(substep_dt)
        self.time_s = float(t_start_s)

        self.space = pymunk.Space()
        self.space.gravity = (0.0, -float(scene.get("gravity_m_s2", 9.81)))
        self.space.iterations = DEFAULT_SOLVER_ITERATIONS

        self.bodies: dict[str, pymunk.Body] = {}
        self.shapes: dict[str, pymunk.Shape] = {}
        self.damping_rates: dict[str, float] = {}
        self.object_meta: dict[str, dict[str, Any]] = {}
        self.skipped_static: list[str] = []
        self.layers = assign_layers(scene)
        ids = [o["id"] for o in scene.get("objects", [])] + [r["id"] for r in self.layers.resting]
        self._bit = {oid: 1 << i for i, oid in enumerate(ids[:30])}
        # Body-body contacts: Chipmunk multiplies the two coefficients (0.65 x 0.65 = 0.42 for two balls, which
        # makes them stick together); use the geometric mean so two equal balls keep their own coefficient.
        self.space.on_collision(_BODY, _BODY, pre_solve=_mean_coefficients)

        self._build_surfaces()
        self._build_static_objects()
        self._build_dynamic_bodies(body_params)

    # Collision filters from layers.py: every moving object has its own category bit; a static shape lists the
    # objects it collides with, a moving object lists the moving objects it collides with.
    def _filter_hits(self, oids: set) -> pymunk.ShapeFilter:
        bits = 0
        for oid in oids:
            bits |= self._bit.get(oid, 0)
        return pymunk.ShapeFilter(categories=bits, mask=bits)

    def _filter_body(self, oid: str) -> pymunk.ShapeFilter:
        mask = self._bit.get(oid, 0)
        for pair in self.layers.pair_hits:
            if oid in pair:
                for other in pair - {oid}:
                    mask |= self._bit.get(other, 0)
        return pymunk.ShapeFilter(categories=self._bit.get(oid, 0), mask=mask)

    def _surface_thickness_m(self) -> float:
        radii = [self.coord.len_px_to_m(_obj_half_extent_px(o)) for o in self.scene.get("objects", [])]
        return max(0.05, 3.0 * max(radii, default=0.03))

    def _build_surfaces(self) -> None:
        """Surfaces are solid below their top line (a zero-width segment lets a body that starts slightly inside it
        be pushed out downwards and fall forever). A table or floor that reaches the frame edge continues off-screen.
        Containers are built as two walls with an open top."""
        W = self.coord.width_px
        depth = self._surface_thickness_m()
        for si, surf in enumerate(self.scene.get("surfaces", [])):
            if surf.get("kind") == "container":
                continue
            (x0, y0), (x1, y1) = sorted([tuple(surf["p0"]), tuple(surf["p1"])])
            if surf.get("kind") in ("table", "floor", "other") and abs(x1 - x0) > 1e-6:
                slope = (y1 - y0) / (x1 - x0)
                if x0 <= 0.02 * W:
                    y0, x0 = y0 - slope * (x0 + W), -W
                if x1 >= 0.98 * W:
                    y1, x1 = y1 + slope * (2 * W - x1), 2 * W
            p0 = self.coord.px_to_m(x0, y0)
            p1 = self.coord.px_to_m(x1, y1)
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-6:
                continue
            if abs(p1[0] - p0[0]) > 1e-6:
                shape: pymunk.Shape = pymunk.Poly(self.space.static_body,
                                                  [p0, p1, (p1[0], p1[1] - depth), (p0[0], p0[1] - depth)])
            else:  # vertical line (wall): a thin segment
                shape = pymunk.Segment(self.space.static_body, p0, p1, 0.0)
            shape.friction = 1.0
            shape.elasticity = 1.0
            shape.filter = self._filter_hits(self.layers.surface_hits.get(si, set()))
            if surf.get("material") == "soft":  # objects sink in: no bounce, stop quickly (see contact damping)
                shape.elasticity = 0.0
                shape.collision_type = _SOFT
            elif surf.get("kind") in ("table", "floor"):
                shape.collision_type = _LEVEL
            self.space.add(shape)
        for c in self.layers.containers:
            for x in (c.x0, c.x1):
                wall = pymunk.Segment(self.space.static_body, self.coord.px_to_m(x, c.top),
                                      self.coord.px_to_m(x, c.floor), self.coord.len_px_to_m(4.0))
                wall.friction = 1.0
                wall.elasticity = 1.0
                self.space.add(wall)

    def _build_static_objects(self) -> None:
        dyn_objects = self.scene.get("objects", [])
        for ti, sobj in enumerate(self.scene.get("static_objects", [])):
            bbox = sobj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            if ti in self.layers.container_statics:
                continue  # an open box: built as container walls in _build_surfaces
            if any(r["static"] == ti for r in self.layers.resting):
                continue  # a body at rest (built in _build_dynamic_bodies)
            if _static_box_overlaps_track(bbox, dyn_objects):
                self.skipped_static.append(str(sobj.get("name", "static")))
                continue
            blocks = self.layers.static_hits.get(ti, set())
            x0, y0, x1, y1 = [float(v) for v in bbox]
            p_bl = self.coord.px_to_m(min(x0, x1), max(y0, y1))
            p_tr = self.coord.px_to_m(max(x0, x1), min(y0, y1))
            w_m = p_tr[0] - p_bl[0]
            h_m = p_tr[1] - p_bl[1]
            if w_m < 1e-5 or h_m < 1e-5:
                continue
            verts = [
                (p_bl[0], p_bl[1]),
                (p_tr[0], p_bl[1]),
                (p_tr[0], p_tr[1]),
                (p_bl[0], p_tr[1]),
            ]
            poly = pymunk.Poly(self.space.static_body, verts)
            poly.friction = 1.0
            poly.elasticity = 1.0
            poly.filter = self._filter_hits(blocks)
            self.space.add(poly)

    def _build_dynamic_bodies(
        self,
        body_params: dict[str, BodyParams | dict[str, Any]],
    ) -> None:
        objects = self.scene.get("objects", [])
        sizes = [float(o.get("est_real_size_cm") or 6.7) for o in objects]
        ref_size = sizes[0] if sizes else 6.7
        self._ref_mass_g = float(objects[0].get("est_mass_g") or 0.0) if objects else 0.0

        for obj in objects:
            oid = obj["id"]
            if oid not in body_params:
                continue
            raw_p = body_params[oid]
            params = raw_p if isinstance(raw_p, BodyParams) else BodyParams.from_dict(raw_p)
            self._add_body(oid, obj, params, ref_size, spin_from_velocity=True)

        # Static objects standing in a moving object's path (layers.py step 3): bodies at rest that can be knocked.
        for rest in self.layers.resting:
            x0, y0 = self.coord.px_to_m(rest["x"], rest["y"])
            priors = default_priors("roll" if rest["shape"] == "circle" else "slide", rest["shape"])
            self._add_body(rest["id"], rest, BodyParams(x0=x0, y0=y0, vx0=0.0, vy0=0.0, **priors), ref_size,
                           spin_from_velocity=False)

    def _add_body(self, oid: str, obj: dict[str, Any], params: BodyParams, ref_size: float,
                  spin_from_velocity: bool) -> None:
        shape_kind = obj.get("shape", "circle")
        size_cm = max(0.5, float(obj.get("est_real_size_cm") or ref_size))
        mass = max(0.05, (size_cm / max(0.5, ref_size)) ** 3)  # same density as the first object
        if self._ref_mass_g > 0 and float(obj.get("est_mass_g") or 0.0) > 0:  # perception's mass estimate
            mass = max(0.01, float(obj["est_mass_g"]) / self._ref_mass_g)

        if shape_kind == "box":
            w_px = float(obj.get("w_px") or 20.0)
            h_px = float(obj.get("h_px") or 20.0)
            w_m = max(1e-4, self.coord.len_px_to_m(w_px))
            h_m = max(1e-4, self.coord.len_px_to_m(h_px))
            moment = pymunk.moment_for_box(mass, (w_m, h_m))
            body = pymunk.Body(mass, moment)
            shape: pymunk.Shape = pymunk.Poly.create_box(body, (w_m, h_m))
            radius_m = 0.5 * min(w_m, h_m)
        else:
            r_px = float(obj.get("radius_px") or 10.0)
            radius_m = max(1e-4, self.coord.len_px_to_m(r_px))
            moment = pymunk.moment_for_circle(mass, 0.0, radius_m)
            body = pymunk.Body(mass, moment)
            shape = pymunk.Circle(body, radius_m)

        body.position = (params.x0, params.y0)
        body.velocity = (params.vx0, params.vy0)
        body.angle = params.angle0
        if params.omega0 is not None:
            body.angular_velocity = params.omega0
        elif spin_from_velocity and shape_kind == "circle" and self.motion_model in ("roll", "other"):
            body.angular_velocity = -params.vx0 / radius_m
        else:
            body.angular_velocity = 0.0

        shape.friction = max(0.0, float(params.friction))
        shape.elasticity = max(0.0, min(1.0, float(params.elasticity)))
        shape.filter = self._filter_body(oid)
        shape.collision_type = _BODY

        self.space.add(body, shape)
        self.bodies[oid] = body
        self.shapes[oid] = shape
        self.damping_rates[oid] = max(0.0, float(params.damping))
        self.object_meta[oid] = {
            "shape": shape_kind,
            "radius_m": radius_m,
            "extent_px": _obj_half_extent_px(obj),
        }

    def _apply_contact_damping(self, dt: float) -> None:
        """While a body is on a support surface: rolling/sliding damping, and level tables stay level.

        A table or floor that runs away from the camera is drawn as a sloped line in the image (perspective), but
        it is level: gravity has no component along it. While a body touches such a surface, the part of gravity
        along the line is cancelled, so a ball rolling "up" the drawn line does not slow and roll back. Ramps keep
        their real slope.
        """
        g = self.space.gravity
        for oid, body in self.bodies.items():
            gamma = self.damping_rates[oid]
            normals: list[tuple[float, float]] = []
            level: list[tuple[float, float]] = []
            soft: list[bool] = []

            def _collect(arb: pymunk.Arbiter) -> None:
                n = arb.contact_point_set.normal
                # Support contact (table, floor, ramp) rather than purely vertical wall
                if abs(n.y) > 0.25:
                    # Orient normal consistently upward before averaging
                    sign = 1.0 if n.y >= 0 else -1.0
                    normals.append((n.x * sign, n.y * sign))
                    if any(sh.collision_type in (_LEVEL, _SOFT) for sh in arb.shapes):
                        level.append((n.x * sign, n.y * sign))
                    if any(sh.collision_type == _SOFT for sh in arb.shapes):
                        soft.append(True)

            body.each_arbiter(_collect)
            if not normals:
                continue

            if level:
                lx, ly = sum(n[0] for n in level), sum(n[1] for n in level)
                ll = math.hypot(lx, ly)
                if ll > 1e-8:
                    lx, ly = lx / ll, ly / ll
                    gn = g.x * lx + g.y * ly
                    gtx, gty = g.x - gn * lx, g.y - gn * ly  # gravity along the level surface
                    body.velocity = (body.velocity.x - gtx * dt, body.velocity.y - gty * dt)
            if soft:
                gamma = max(gamma, SOFT_DAMPING)
            rr = ROLLING_RESISTANCE if self.object_meta[oid]["shape"] == "circle" else 0.0
            if gamma <= 0.0 and rr <= 0.0:
                continue

            nx = sum(n[0] for n in normals)
            ny = sum(n[1] for n in normals)
            nlen = math.hypot(nx, ny)
            if nlen < 1e-8:
                continue
            nx /= nlen
            ny /= nlen

            vx, vy = body.velocity.x, body.velocity.y
            vn = vx * nx + vy * ny
            vnx, vny = vn * nx, vn * ny
            vtx, vty = vx - vnx, vy - vny

            decay = math.exp(-gamma * dt)
            # Rolling resistance: a constant braking (rr * g) on top of the exponential damping, so a slow ball
            # stops instead of creeping on for seconds.
            vt = math.hypot(vtx, vty) * decay
            keep = max(0.0, vt - rr * abs(g.y) * dt) / vt if vt > 1e-9 else 0.0
            body.velocity = (vnx + vtx * decay * keep, vny + vty * decay * keep)
            body.angular_velocity = body.angular_velocity * decay * keep

    def step(self, dt: float) -> None:
        """Advance the simulation by `dt` seconds using fixed-size substeps."""
        if dt <= 0.0:
            return
        n_steps = max(1, int(math.ceil(dt / self.substep_dt)))
        sub_dt = dt / n_steps
        for _ in range(n_steps):
            self.space.step(sub_dt)
            self._apply_contact_damping(sub_dt)
        self.time_s += dt

    def step_to(self, target_t_s: float) -> None:
        """Advance the simulation from current `time_s` to `target_t_s`."""
        dt = float(target_t_s) - self.time_s
        if dt < -1e-9:
            raise ValueError(f"Cannot step backwards from {self.time_s} to {target_t_s}")
        if dt > 1e-12:
            self.step(dt)
        self.time_s = float(target_t_s)

    def get_state_m(self) -> dict[str, dict[str, float]]:
        """Current state of every dynamic body in y-up metres."""
        out: dict[str, dict[str, float]] = {}
        for oid, body in self.bodies.items():
            out[oid] = {
                "x": float(body.position.x),
                "y": float(body.position.y),
                "vx": float(body.velocity.x),
                "vy": float(body.velocity.y),
                "angle": float(body.angle),
                "omega": float(body.angular_velocity),
            }
        return out

    def get_state_px(self) -> dict[str, dict[str, Any]]:
        """Current state of every dynamic body in orig y-down pixels (CONTRACT.md §2)."""
        out: dict[str, dict[str, Any]] = {}
        for oid, body in self.bodies.items():
            x_px, y_px = self.coord.m_to_px(body.position.x, body.position.y)
            extent_px = self.object_meta[oid]["extent_px"]
            out[oid] = {
                "x": x_px,
                "y": y_px,
                "angle": float(body.angle),
                "visible": self.coord.is_visible(x_px, y_px, extent_px=extent_px),
            }
        return out
