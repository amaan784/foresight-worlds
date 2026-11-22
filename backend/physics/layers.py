"""Depth layers for a 2D simulation of a 3D scene (all in original image pixels, y down).

A side-view camera flattens depth: a box in front of a ball's path and a box in its path look the same in 2D.
The cue we use is where things touch the ground. Objects resting on one table at the same distance from the
camera touch it along the same image line; a nearer object touches it lower in the image, a farther one higher.

  * Each moving object's layer is the surface it rolls on (the line under its tracked bottoms). Objects with no
    surface under them (falling, thrown) are "free".
  * Objects in the same layer collide with each other and with that layer's surface; objects in different layers
    pass each other (two balls rolling at different depths).
  * A static object whose bottom sits on a layer's line (within 1.2 radii of the layer's objects) is an obstacle
    in that layer. One whose bottom is lower in the image is in front: it hides the objects of that layer instead
    of blocking them. One whose bottom is higher is behind: no contact, drawn behind. On the 15 eval clips the two
    groups are far apart: obstacles sit within 0.6 radii of the line, objects in front/behind 2.4 radii or more.
  * A surface of kind "container" is an opening, not a lid: objects fall in between its walls and are hidden by its
    front (the static object of the same box, or the region from its rim down to the surface below).
Surfaces nobody rolls on (ramps, walls, floors under a drop) are shared, except with objects that are below
them in the image (a table's back edge drawn above a ball rolling in front of it is at another depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEPTH_TOL_RADII = 1.2
SUPPORT_TOL_RADII = 0.5


def y_on(surf: dict[str, Any], x: float) -> float:
    (x0, y0), (x1, y1) = surf["p0"], surf["p1"]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if abs(x1 - x0) > 1e-9 else min(y0, y1)


def spans(surf: dict[str, Any], x: float, margin: float = 0.0) -> bool:
    return min(surf["p0"][0], surf["p1"][0]) - margin <= x <= max(surf["p0"][0], surf["p1"][0]) + margin


def half_height_px(obj: dict[str, Any]) -> float:
    if obj.get("shape") == "box":
        return 0.5 * float(obj.get("h_px") or 20.0)
    return float(obj.get("radius_px") or 10.0)


@dataclass
class Container:
    surface: int
    x0: float
    x1: float
    top: float
    floor: float
    static: int | None  # index into static_objects of the container's own body, if perception listed it


@dataclass
class Layers:
    layer_of: dict[str, int | None] = field(default_factory=dict)  # object id -> layer (support surface idx) or None
    support_layer: dict[int, int] = field(default_factory=dict)  # surface idx -> layer, for surfaces someone rolls on
    surface_hits: dict[int, set] = field(default_factory=dict)  # surface idx -> object ids it collides with
    static_hits: dict[int, set] = field(default_factory=dict)  # static idx -> object ids it collides with
    pair_hits: set = field(default_factory=set)  # frozenset({a, b}) of moving objects that collide with each other
    occluders: dict[str, list[list[float]]] = field(default_factory=dict)  # object id -> [x0, y0, x1, y1] that hide it
    containers: list[Container] = field(default_factory=list)
    container_statics: set = field(default_factory=set)
    resting: list = field(default_factory=list)  # static objects simulated as bodies at rest (see step 3)
    notes: list[str] = field(default_factory=list)


def surfaces_touching(scene: dict[str, Any], layers: "Layers", oid: str) -> list[int]:
    """Surface indices object `oid` collides with (containers excluded: they are walls, not lines)."""
    return [si for si, hits in layers.surface_hits.items() if oid in hits]


def _overlaps_track(bbox: list[float], objects: list[dict[str, Any]]) -> bool:
    """True if a tracked path runs through the box (the pipe a ball rolls out of): it is not something to knock."""
    x0, y0, x1, y1 = bbox
    for ob in objects:
        r = 0.65 * half_height_px(ob)
        if any(x0 - r < p["x"] < x1 + r and y0 - r < p["y"] < y1 + r for p in ob.get("track", [])):
            return True
    return False


def _below(ob: dict[str, Any], surf: dict[str, Any]) -> bool:
    """True if the object's top is below the surface line where they overlap (the line is at another depth)."""
    r = half_height_px(ob)
    pts = ob.get("track") or ([{"x": ob["x"], "y": ob["y"]}] if "x" in ob else [])
    gaps = [(p["y"] - r) - y_on(surf, p["x"]) for p in pts if spans(surf, p["x"], r)]
    return bool(gaps) and sorted(gaps)[len(gaps) // 2] > -0.5 * r


def _movers(objects: list[dict[str, Any]], out: "Layers", lay: int) -> str:
    names = [o.get("name", o["id"]) for o in objects if out.layer_of.get(o["id"]) == lay]
    return " and ".join(names) or "moving object"


def assign_layers(scene: dict[str, Any]) -> Layers:
    surfaces = scene.get("surfaces", [])
    statics = scene.get("static_objects", [])
    objects = scene.get("objects", [])  # an object without a track is treated as free (touches everything)
    H = float(scene["orig_size"][1])
    out = Layers()

    # 1. layer of each moving object: the non-container surface right under its tracked bottoms
    for ob in objects:
        r = half_height_px(ob)
        best, best_err = None, None
        for si, s in enumerate(surfaces):
            if s.get("kind") == "container":
                continue
            errs = [abs(y_on(s, p["x"]) - (p["y"] + r)) for p in ob.get("track", []) if spans(s, p["x"], r)]
            if len(errs) < max(2, len(ob.get("track", [])) // 2):
                continue
            err = sorted(errs)[len(errs) // 2]
            if best_err is None or err < best_err:
                best, best_err = si, err
        layer = best if best is not None and best_err <= SUPPORT_TOL_RADII * r else None
        out.layer_of[ob["id"]] = layer
        if layer is not None:
            out.support_layer[layer] = layer
    radius_of_layer = {}
    for ob in objects:
        lay = out.layer_of[ob["id"]]
        if lay is not None:
            radius_of_layer[lay] = max(radius_of_layer.get(lay, 0.0), half_height_px(ob))

    # 2. containers: walls at the rim ends down to the next surface below (or the box's own bottom)
    for si, s in enumerate(surfaces):
        if s.get("kind") != "container":
            continue
        x0, x1 = sorted([float(s["p0"][0]), float(s["p1"][0])])
        top = min(float(s["p0"][1]), float(s["p1"][1]))
        xc = 0.5 * (x0 + x1)
        body = None
        for ti, st in enumerate(statics):
            bx0, by0, bx1, by1 = st["bbox"]
            overlap = max(0.0, min(x1, bx1) - max(x0, bx0)) / max(1.0, x1 - x0)
            if overlap >= 0.6 and abs(by0 - top) <= 0.25 * max(1.0, by1 - by0):
                body = ti
                break
        below = [y_on(o, xc) for oi, o in enumerate(surfaces)
                 if oi != si and o.get("kind") != "container" and spans(o, xc) and y_on(o, xc) > top + 1]
        floor = min(below) if below else (statics[body]["bbox"][3] if body is not None else H)
        out.containers.append(Container(si, x0, x1, top, floor, body))
        if body is not None:
            out.container_statics.add(body)
        out.notes.append(f"The {s['name']} is open on top: things that fall in stay inside, hidden by its front.")

    # 3. resting objects: a static object standing on a layer's line can be knocked by that layer's objects
    #    (balls in a row, a rubber duck). It becomes a body at rest; mass from its size like the moving objects.
    #    Very heavy ones (over 30x the moving object's mass) and ones the tracked path runs through stay static.
    ref_cm = max(0.5, float((objects[0].get("est_real_size_cm") if objects else None) or 6.7))
    for ti, st in enumerate(statics):
        if ti in out.container_statics or _overlaps_track(st["bbox"], objects):
            continue
        bx0, by0, bx1, by1 = st["bbox"]
        xc = 0.5 * (bx0 + bx1)
        size_cm = float(st.get("est_real_size_cm") or 0.0)
        ref_g = float((objects[0].get("est_mass_g") if objects else None) or 0.0)
        ratio = (float(st.get("est_mass_g") or 0.0) / ref_g) if ref_g > 0 and st.get("est_mass_g") else (size_cm / ref_cm) ** 3
        if size_cm <= 0 or ratio > 30:
            continue
        same = [lay for lay, r in radius_of_layer.items() if abs(by1 - y_on(surfaces[lay], xc)) <= DEPTH_TOL_RADII * r]
        if not same:
            continue
        lay = same[0]
        w, h = bx1 - bx0, by1 - by0
        circle = st.get("perceived_shape") == "circle"
        half = 0.5 * min(w, h) if circle else 0.5 * h
        out.resting.append({
            "id": f"rest{ti}", "static": ti, "name": st.get("name", "static"), "shape": "circle" if circle else "box",
            "radius_px": 0.5 * min(w, h) if circle else None, "w_px": None if circle else w, "h_px": None if circle else h,
            "est_real_size_cm": size_cm, "x": xc, "y": y_on(surfaces[lay], xc) - half, "bbox_center": [xc, 0.5 * (by0 + by1)],
            "sprite_png": st.get("sprite_png"), "track": [],
        })
        out.layer_of[f"rest{ti}"] = lay
        out.notes.append(f"The {st['name']} rests in the path of the {_movers(objects, out, lay)}: it can be knocked.")
    bodies = list(objects) + out.resting
    rest_statics = {r["static"] for r in out.resting}

    # 4. surfaces: own layer's line, or a line nobody rolls on that is not above the object
    for si, s in enumerate(surfaces):
        if s.get("kind") == "container":
            continue
        hits = set()
        for ob in bodies:
            lay = out.layer_of[ob["id"]]
            if lay == si:
                hits.add(ob["id"])
            elif (si not in out.support_layer or lay is None) and not _below(ob, s):
                hits.add(ob["id"])
        out.surface_hits[si] = hits

    # 5. bodies collide in the same layer; falling objects collide with anything
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            la, lb = out.layer_of[a["id"]], out.layer_of[b["id"]]
            if la is None or lb is None or la == lb:
                out.pair_hits.add(frozenset((a["id"], b["id"])))

    # 6. static objects: obstacle, in front, or behind, per layer; falling objects hit every non-container static
    for ti, st in enumerate(statics):
        if ti in out.container_statics or ti in rest_statics:
            continue
        bx0, by0, bx1, by1 = st["bbox"]
        xc = 0.5 * (bx0 + bx1)
        blocks = {ob["id"] for ob in objects if out.layer_of[ob["id"]] is None}
        for lay, r in radius_of_layer.items():
            d = by1 - y_on(surfaces[lay], xc)
            if abs(d) <= DEPTH_TOL_RADII * r:
                blocks |= {ob["id"] for ob in bodies if out.layer_of[ob["id"]] == lay}
            elif d > 0:  # touches the ground nearer the camera: in front of this layer
                for ob in objects:
                    if out.layer_of[ob["id"]] == lay:
                        out.occluders.setdefault(ob["id"], []).append([bx0, by0, bx1, by1])
                out.notes.append(f"The {st['name']} is nearer the camera than the {_movers(objects, out, lay)}: it hides it, "
                                 f"it does not stop it.")
            else:
                out.notes.append(f"The {st['name']} is farther back than the {_movers(objects, out, lay)}: they cannot touch.")
        out.static_hits[ti] = blocks

    # 7. containers hide what falls into them (free objects) and objects of layers they stand on or in front of
    for c in out.containers:
        rect = list(statics[c.static]["bbox"]) if c.static is not None else [c.x0, c.top, c.x1, c.floor]
        for ob in objects:
            lay = out.layer_of[ob["id"]]
            if lay is None or rect[3] - y_on(surfaces[lay], 0.5 * (rect[0] + rect[2])) >= -DEPTH_TOL_RADII * radius_of_layer[lay]:
                out.occluders.setdefault(ob["id"], []).append(rect)
    return out
