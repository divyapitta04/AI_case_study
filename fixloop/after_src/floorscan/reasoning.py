"""Concealed-damage flags (explicit rule base) and scope line items keyed to surfaces.

The rules are transparent heuristics from common restoration practice (wicking behind baseboards, wet
insulation above stained ceilings, hidden growth behind visible mould).  They are NOT code citations and are
exposed as data (RULES) so a reviewer can read, change or disable each one.  Every flag reports the rule id
that fired and the damage regions that triggered it.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import uncertainty as U

SQM_TO_SF = 10.7639

RULES: List[Dict] = [
    dict(id="R-CEIL-01", trigger="water_stain on ceiling",
         text="Ceiling water stain: drywall and insulation above the stain are likely saturated beyond the visible edge.",
         margin=0.30, confidence=0.70),
    dict(id="R-WALL-02", trigger="water_stain on wall with lower edge < 0.45 m above floor",
         text="Low wall stain: moisture wicks up the cavity; wet drywall/insulation likely behind the baseboard and above the visible line.",
         margin=0.30, confidence=0.70),
    dict(id="R-MOLD-03", trigger="mold on wall or ceiling",
         text="Visible mould: growth is likely present in the wall cavity / back of the board beyond the visible spots.",
         margin=0.30, confidence=0.65),
    dict(id="R-HOLE-04", trigger="hole on wall",
         text="Opening in the wall surface: cavity (insulation, wiring) is exposed; check for moisture and damaged lining around the opening.",
         margin=0.20, confidence=0.55),
    dict(id="R-COMB-05", trigger="mold and water_stain on the same surface within 0.6 m",
         text="Stain and mould together indicate an ongoing moisture source: treat the whole surface as potentially wet.",
         margin=0.0, confidence=0.80),
    dict(id="R-FLOOR-06", trigger="water_stain on floor within 0.6 m of a wall with stain/mould",
         text="Floor staining next to a damaged wall: subfloor / flooring underlay may be wet.",
         margin=0.30, confidence=0.60),
]
RULE_BY_ID = {r["id"]: r for r in RULES}


def _bbox_uv(poly_uv: List[List[float]]) -> Tuple[float, float, float, float]:
    a = np.array(poly_uv)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def concealed_flags(room_id: str, regions: List[Tuple[str, str, int, object, str]], tier: str,
                    extra_rel: float = 0.0, wall_height: float = 2.5) -> List[Dict]:
    """regions: (surface_id, kind, idx, Region, damage_id).  Returns concealed_damage list."""
    flags: List[Dict] = []
    n = 0

    def add(rule_id, sid, ext_area, ext_w, ext_h, trig, extra_conf=0.0):
        nonlocal n
        r = RULE_BY_ID[rule_id]
        sig = 0.05 * ext_area + 0.02 * (ext_w + ext_h)
        flags.append(dict(
            id=f"{room_id}.cd{n}", rule=rule_id, surface_id=sid, description=r["text"],
            confidence=round(min(0.95, r["confidence"] + extra_conf), 2), triggered_by=list(trig),
            extent=dict(width=U.measure(ext_w, 0.03, tier, "extent", extra_rel),
                        height=U.measure(ext_h, 0.03, tier, "extent", extra_rel),
                        area=U.measure(ext_area, sig, tier, "extent", extra_rel))))
        n += 1

    by_surface: Dict[str, List] = {}
    for item in regions:
        by_surface.setdefault(item[0], []).append(item)
    for sid, items in by_surface.items():
        for (_, k, idx, reg, did) in items:
            u0, v0, u1, v1 = _bbox_uv(reg.polygon_uv)
            m = None
            if reg.cls == "water_stain" and k == "ceiling":
                m = RULE_BY_ID["R-CEIL-01"]["margin"]
                add("R-CEIL-01", sid, (reg.width + 2 * m) * (reg.height + 2 * m), reg.width + 2 * m, reg.height + 2 * m, [did])
            elif reg.cls == "water_stain" and k == "wall" and v0 < 0.45:
                m = RULE_BY_ID["R-WALL-02"]["margin"]
                top = v1 + m
                add("R-WALL-02", sid, (reg.width + 2 * m) * top, reg.width + 2 * m, top, [did])
            elif reg.cls == "mold":
                m = RULE_BY_ID["R-MOLD-03"]["margin"]
                add("R-MOLD-03", sid, (reg.width + 2 * m) * (reg.height + 2 * m), reg.width + 2 * m, reg.height + 2 * m, [did])
            elif reg.cls == "hole" and k == "wall":
                m = RULE_BY_ID["R-HOLE-04"]["margin"]
                add("R-HOLE-04", sid, (reg.width + 2 * m) * (reg.height + 2 * m), reg.width + 2 * m, reg.height + 2 * m, [did])
        stains = [i for i in items if i[3].cls == "water_stain"]
        molds = [i for i in items if i[3].cls == "mold"]
        for s_ in stains:
            for m_ in molds:
                us0, vs0, us1, vs1 = _bbox_uv(s_[3].polygon_uv)
                um0, vm0, um1, vm1 = _bbox_uv(m_[3].polygon_uv)
                gap = max(max(um0 - us1, us0 - um1, 0), max(vm0 - vs1, vs0 - vm1, 0))
                if gap < 0.6:
                    w = max(us1, um1) - min(us0, um0)
                    h = max(vs1, vm1) - min(vs0, vm0)
                    add("R-COMB-05", sid, w * h, w, h, [s_[4], m_[4]])
    # floor stain next to damaged wall (distance not resolvable without plan placement -> same room test)
    floor_st = [i for i in regions if i[1] == "floor" and i[3].cls == "water_stain"]
    wall_dmg = [i for i in regions if i[1] == "wall" and i[3].cls in ("water_stain", "mold")]
    if floor_st and wall_dmg:
        for f in floor_st:
            add("R-FLOOR-06", f[0], f[3].area + 1.2, f[3].width + 0.6, f[3].height + 0.6, [f[4], wall_dmg[0][4]])
    return flags


# ---------------------------------------------------------------------------
CATALOG = {
    "DRY-RR": ("Drywall - remove and replace", "m2"),
    "INS-RR": ("Insulation - remove and replace", "m2"),
    "PNT-WALL": ("Paint wall surface (full wall for colour continuity)", "m2"),
    "PNT-CEIL": ("Paint ceiling surface", "m2"),
    "BB-RR": ("Baseboard - remove and replace", "m"),
    "MOLD-REM": ("Mould remediation (HEPA clean / treat)", "m2"),
    "HOLE-PATCH": ("Patch hole in drywall", "ea"),
    "FLR-RR": ("Flooring - remove and replace (affected area)", "m2"),
    "DRY-OUT": ("Structural drying (equipment, per room-day)", "room-day"),
}


def scope_items(room_id: str, regions, flags: List[Dict], surfaces: Dict[str, Dict], tier: str,
                extra_rel: float = 0.0) -> List[Dict]:
    """Line items keyed to surfaces. `surfaces`: id -> surface dict (with area measure) from the assembler."""
    items: List[Dict] = []
    n = 0

    def add(code, sid, qty, sig, basis, ids):
        nonlocal n
        desc, unit = CATALOG[code]
        q = U.measure(qty, sig, tier, "extent", extra_rel, lo_clip=0.0) if unit in ("m2", "m") else \
            dict(value=float(qty), ci95=[float(qty), float(qty)], sigma=0.0)
        row = dict(id=f"{room_id}.sc{n}", surface_id=sid, code=code, description=desc, unit=unit, quantity=q,
                   basis=basis, evidence=list(ids))
        if unit == "m2":
            row["quantity_sf"] = round(qty * SQM_TO_SF, 1)
        items.append(row)
        n += 1

    flag_by_surface: Dict[str, List[Dict]] = {}
    for f in flags:
        flag_by_surface.setdefault(f["surface_id"], []).append(f)
    seen_paint = set()
    for (sid, kind, idx, reg, did) in regions:
        fl = flag_by_surface.get(sid, [])
        if reg.cls == "hole":
            add("HOLE-PATCH", sid, 1, 0, f"hole {did}", [did])
        if reg.cls == "mold":
            add("MOLD-REM", sid, (reg.width + 0.6) * (reg.height + 0.6), 0.1 * (reg.width + reg.height), f"mould {did}", [did])
    # flags drive removal work
    for sid, fl in flag_by_surface.items():
        kind = "ceiling" if sid.endswith(".ceiling") else ("floor" if sid.endswith(".floor") else "wall")
        for f in fl:
            if f["rule"] == "R-COMB-05":
                continue
            a = f["extent"]["area"]["value"]
            sg = f["extent"]["area"]["sigma"]
            w = f["extent"]["width"]["value"]
            if kind in ("wall", "ceiling"):
                add("DRY-RR", sid, a, sg, f"concealed-damage flag {f['rule']}", [f["id"]])
            if kind == "ceiling" or (kind == "wall" and f["rule"] in ("R-WALL-02", "R-MOLD-03")):
                add("INS-RR", sid, a, sg, f"concealed-damage flag {f['rule']}", [f["id"]])
            if kind == "wall" and f["rule"] == "R-WALL-02":
                add("BB-RR", sid, w, 0.03 * w + 0.03, f"concealed-damage flag {f['rule']}", [f["id"]])
            if kind == "floor":
                add("FLR-RR", sid, a, sg, f"concealed-damage flag {f['rule']}", [f["id"]])
        if sid not in seen_paint and kind in ("wall", "ceiling") and sid in surfaces:
            seen_paint.add(sid)
            a = surfaces[sid]["area"]
            add("PNT-CEIL" if kind == "ceiling" else "PNT-WALL", sid, a["value"], a["sigma"],
                "repaint whole surface after repair", [f["id"] for f in fl])
    if flags:
        add("DRY-OUT", f"{room_id}.floor", 3, 0, "3 days default when concealed damage is flagged", [f["id"] for f in flags])
    return items
