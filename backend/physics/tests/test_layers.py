"""Depth layers (backend/physics/layers.py) on small synthetic scenes, original image pixels (y down)."""
from __future__ import annotations

from physics.layers import assign_layers
from physics.world import BodyParams, PhysicsWorld


def _scene(statics=(), surfaces=None, objects=None) -> dict:
    ball = {"id": "obj0", "name": "ball", "shape": "circle", "radius_px": 100.0, "w_px": None, "h_px": None,
            "est_real_size_cm": 6.7, "track": [{"t_s": 2.9 + 0.033 * i, "x": 300.0 + 100 * i, "y": 900.0} for i in range(4)]}
    return {"orig_size": [3840, 2160], "px_per_m": 3000.0, "gravity_m_s2": 9.81, "fps_out": 24, "t_cond_end_s": 3.0,
            "pred_seconds": 5.0, "motion_type_hint": "roll", "objects": objects or [ball], "static_objects": list(statics),
            "surfaces": surfaces or [{"name": "table", "kind": "table", "p0": [0.0, 1000.0], "p1": [3840.0, 1000.0]}]}


def _box(name, x0, y0, x1, y1, size_cm=40.0, shape="box"):
    return {"name": name, "shape": "box", "bbox": [x0, y0, x1, y1], "perceived_shape": shape, "est_real_size_cm": size_cm}


def test_object_rolls_in_the_layer_of_the_line_under_it():
    L = assign_layers(_scene())
    assert L.layer_of["obj0"] == 0 and 0 in L.support_layer
    assert "obj0" in L.surface_hits[0]


def test_box_in_front_hides_and_does_not_block():
    L = assign_layers(_scene([_box("near box", 1500, 400, 1900, 1400)]))  # bottom 400 px below the line: nearer
    assert "obj0" not in L.static_hits[0]
    assert L.occluders["obj0"] == [[1500, 400, 1900, 1400]]


def test_box_on_the_line_blocks():
    L = assign_layers(_scene([_box("heavy box", 1500, 500, 1900, 1010, size_cm=60.0)]))  # bottom on the line
    assert "obj0" in L.static_hits[0]
    assert "obj0" not in L.occluders


def test_box_behind_neither_blocks_nor_hides():
    L = assign_layers(_scene([_box("far box", 1500, 300, 1900, 600)]))  # bottom 400 px above the line: farther
    assert "obj0" not in L.static_hits[0]
    assert "obj0" not in L.occluders


def test_light_ball_on_the_line_becomes_a_body_at_rest():
    L = assign_layers(_scene([_box("other ball", 1500, 810, 1700, 1000, size_cm=6.7, shape="circle")]))
    assert [r["id"] for r in L.resting] == ["rest0"]
    assert L.layer_of["rest0"] == 0 and frozenset(("obj0", "rest0")) in L.pair_hits


def test_balls_at_two_depths_pass_each_other():
    near = {"id": "obj1", "name": "near ball", "shape": "circle", "radius_px": 130.0, "w_px": None, "h_px": None,
            "est_real_size_cm": 6.7, "track": [{"t_s": 2.9 + 0.033 * i, "x": 3000.0 - 100 * i, "y": 1370.0} for i in range(4)]}
    far = _scene()["objects"][0]
    surfaces = [{"name": "table", "kind": "table", "p0": [0.0, 1000.0], "p1": [3840.0, 1000.0]},
                {"name": "near line", "kind": "table", "p0": [0.0, 1500.0], "p1": [3840.0, 1500.0]}]
    L = assign_layers(_scene(surfaces=surfaces, objects=[far, near]))
    assert L.layer_of == {"obj0": 0, "obj1": 1}
    assert frozenset(("obj0", "obj1")) not in L.pair_hits
    assert "obj1" not in L.surface_hits[0]  # the far table line is above the near ball


def test_container_is_open_on_top():
    falling = {"id": "obj0", "name": "duck", "shape": "circle", "radius_px": 80.0, "w_px": None, "h_px": None,
               "est_real_size_cm": 8.0, "track": [{"t_s": 2.9 + 0.033 * i, "x": 1900.0, "y": 300.0 + 40 * i} for i in range(4)]}
    surfaces = [{"name": "table", "kind": "table", "p0": [1000.0, 1750.0], "p1": [2800.0, 1750.0]},
                {"name": "box", "kind": "container", "p0": [1600.0, 1200.0], "p1": [2200.0, 1200.0]}]
    scene = _scene([_box("box", 1550, 1200, 2250, 1750, size_cm=30.0)], surfaces=surfaces, objects=[falling])
    L = assign_layers(scene)
    assert L.layer_of["obj0"] is None and len(L.containers) == 1 and L.container_statics == {0}
    # dropped from above the rim, the duck falls into the box (below the rim) instead of landing on top of it
    c = PhysicsWorld(scene, {}).coord
    x0, y0 = c.px_to_m(1900.0, 400.0)
    w = PhysicsWorld(scene, {"obj0": BodyParams(x0=x0, y0=y0, vx0=0.0, vy0=0.0)}, motion_model="fall")
    w.step_to(1.0)
    assert w.get_state_px()["obj0"]["y"] > 1300.0
