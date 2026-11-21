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

DEFAULT_SUBSTEP_DT = 1.0 / 480.0
DEFAULT_SOLVER_ITERATIONS = 20

# Priors indexed by motion_type_hint: (friction, elasticity, damping_s^-1)
_MOTION_PRIORS: dict[str, dict[str, float]] = {
    "roll": {"friction": 0.50, "elasticity": 0.65, "damping": 0.25},
    "slide": {"friction": 0.35, "elasticity": 0.30, "damping": 0.05},
    "fall": {"friction": 0.50, "elasticity": 0.65, "damping": 0.02},
    "bounce": {"friction": 0.50, "elasticity": 0.70, "damping": 0.05},
    "other": {"friction": 0.50, "elasticity": 0.60, "damping": 0.15},
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

        self._build_surfaces()
        self._build_static_objects()
        self._build_dynamic_bodies(body_params)

    def _build_surfaces(self) -> None:
        for surf in self.scene.get("surfaces", []):
            p0 = self.coord.px_to_m(surf["p0"][0], surf["p0"][1])
            p1 = self.coord.px_to_m(surf["p1"][0], surf["p1"][1])
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 1e-6:
                continue
            seg = pymunk.Segment(self.space.static_body, p0, p1, 0.0)
            seg.friction = 1.0
            seg.elasticity = 1.0
            self.space.add(seg)

    def _build_static_objects(self) -> None:
        dyn_objects = self.scene.get("objects", [])
        for sobj in self.scene.get("static_objects", []):
            bbox = sobj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            if _static_box_overlaps_track(bbox, dyn_objects):
                self.skipped_static.append(str(sobj.get("name", "static")))
                continue
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
            self.space.add(poly)

    def _build_dynamic_bodies(
        self,
        body_params: dict[str, BodyParams | dict[str, Any]],
    ) -> None:
        objects = self.scene.get("objects", [])
        sizes = [float(o.get("est_real_size_cm") or 6.7) for o in objects]
        ref_size = sizes[0] if sizes else 6.7

        for obj in objects:
            oid = obj["id"]
            if oid not in body_params:
                continue
            raw_p = body_params[oid]
            params = raw_p if isinstance(raw_p, BodyParams) else BodyParams.from_dict(raw_p)
            shape_kind = obj.get("shape", "circle")

            size_cm = max(0.5, float(obj.get("est_real_size_cm") or ref_size))
            mass = max(0.05, (size_cm / max(0.5, ref_size)) ** 3)

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
            elif shape_kind == "circle" and self.motion_model in ("roll", "other"):
                body.angular_velocity = -params.vx0 / radius_m
            else:
                body.angular_velocity = 0.0

            shape.friction = max(0.0, float(params.friction))
            shape.elasticity = max(0.0, min(1.0, float(params.elasticity)))

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
        """Apply tangential velocity + spin damping while a body is on a support surface."""
        for oid, body in self.bodies.items():
            gamma = self.damping_rates[oid]
            if gamma <= 0.0:
                continue
            normals: list[tuple[float, float]] = []

            def _collect(arb: pymunk.Arbiter) -> None:
                n = arb.contact_point_set.normal
                # Support contact (table, floor, ramp) rather than purely vertical wall
                if abs(n.y) > 0.25:
                    # Orient normal consistently upward before averaging
                    sign = 1.0 if n.y >= 0 else -1.0
                    normals.append((n.x * sign, n.y * sign))

            body.each_arbiter(_collect)
            if not normals:
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
            body.velocity = (vnx + vtx * decay, vny + vty * decay)
            body.angular_velocity = body.angular_velocity * decay

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
