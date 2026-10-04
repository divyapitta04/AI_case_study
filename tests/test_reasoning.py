import numpy as np

from floorscan import reasoning
from floorscan.damage import Region


def _reg(cls, w, h, u, v):
    poly = [[u, v], [u + w, v], [u + w, v + h], [u, v + h]]
    return Region(cls, np.zeros((2, 2), bool), 0.9, w * h, w, h, 0.05, (0, 0, 1, 1), poly)


def test_rules_fire_and_scope_is_keyed_to_surfaces():
    regs = [("r1.ceiling", "ceiling", 0, _reg("water_stain", 0.8, 0.7, 1, 1), "r1.dmg0"),
            ("r1.w2", "wall", 2, _reg("water_stain", 0.8, 0.8, 1, 0.1), "r1.dmg1"),
            ("r1.w2", "wall", 2, _reg("mold", 0.5, 0.4, 1.4, 0.5), "r1.dmg2"),
            ("r1.w3", "wall", 3, _reg("hole", 0.24, 0.2, 2, 1), "r1.dmg3")]
    flags = reasoning.concealed_flags("r1", regs, "lidar")
    rules = sorted(f["rule"] for f in flags)
    for r in ("R-CEIL-01", "R-WALL-02", "R-MOLD-03", "R-HOLE-04", "R-COMB-05"):
        assert r in rules, r
    surf = {"r1.w2": {"area": {"value": 10.0, "sigma": 0.1}}, "r1.w3": {"area": {"value": 9.0, "sigma": 0.1}},
            "r1.ceiling": {"area": {"value": 15.0, "sigma": 0.1}}}
    sc = reasoning.scope_items("r1", regs, flags, surf, "lidar")
    codes = {s["code"] for s in sc}
    assert {"DRY-RR", "INS-RR", "PNT-WALL", "PNT-CEIL", "MOLD-REM", "HOLE-PATCH", "DRY-OUT", "BB-RR"} <= codes
    assert all(s["surface_id"].startswith("r1.") for s in sc)
    for s in sc:
        assert s["quantity"]["ci95"][0] <= s["quantity"]["value"] <= s["quantity"]["ci95"][1]
