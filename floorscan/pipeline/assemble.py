"""Build the published JSON contract from room models, placements and links."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .. import uncertainty as U
from ..geometry.room import RoomModel, Wall, polygon_from_walls
from ..geometry.stitch import Link, Placement

SCHEMA_VERSION = "floorscan-1.0"


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _area_sigma(room: RoomModel, n: int = 120) -> float:
    rng = np.random.default_rng(1234)
    vals = []
    for _ in range(n):
        ws = [Wall(w.idx, w.orient, w.coord + rng.normal(0, w.sigma), w.s0, w.s1, w.out_n) for w in room.walls]
        vals.append(_poly_area(polygon_from_walls(ws)))
    return float(np.std(vals))


def wall_length_sigma(room: RoomModel, i: int) -> float:
    m = len(room.walls)
    a, b = room.walls[(i - 1) % m], room.walls[(i + 1) % m]
    return float(np.hypot(a.sigma, b.sigma))


def room_height_sigma(room: RoomModel) -> float:
    return float(np.hypot(room.sigma_zf, room.sigma_zc))


def assemble(tier: str, source: str, rooms: Sequence[RoomModel], places: Sequence[Placement],
             links: Sequence[Link], wall_t: float, meta: Dict, timings: Dict, notes: List[str],
             drift_info: Dict, device: Optional[str] = None, extra_rel: Optional[Dict[str, float]] = None) -> Dict:
    extra_rel = extra_rel or {}
    # plan-frame polygons
    plan_polys = []
    for r, pl in zip(rooms, places):
        plan_polys.append((pl.R @ r.polygon.T).T + pl.p)
    origin = np.concatenate(plan_polys).min(0) if plan_polys else np.zeros(2)
    if not rooms:
        notes = list(notes) + ["NO ROOM RECONSTRUCTED: no closed free-space region was found (capture too short, held too low, "
                               "or floor and ceiling never seen) - recapture following capture/PROTOCOL.md"]
    out_rooms = []
    plan_rooms = []
    for ri, (r, pl, pp) in enumerate(zip(rooms, places, plan_polys)):
        pp = pp - origin
        xr = extra_rel.get(r.id, 0.0)
        height = r.height
        walls_j = []
        wall_ends = []
        for i, w in enumerate(r.walls):
            p0, p1 = pp[i - 1], pp[i]
            wall_ends.append((p0, p1))
            L = float(np.linalg.norm(p1 - p0))
            sig = wall_length_sigma(r, i)
            walls_j.append(dict(id=f"{r.id}.w{i}", p0=[round(float(p0[0]), 4), round(float(p0[1]), 4)],
                                p1=[round(float(p1[0]), 4), round(float(p1[1]), 4)],
                                length=U.measure(L, sig, tier, "length", xr),
                                observed=bool(w.observed)))
        area = _poly_area(r.polygon)
        a_sig = _area_sigma(r)
        h_m = U.measure(height, room_height_sigma(r), tier, "height", xr)
        ops_j = []
        link_by_open = {}
        for l in links:
            link_by_open[(l.a, l.oa)] = (l.b, l.ob)
            link_by_open[(l.b, l.ob)] = (l.a, l.oa)
        dims = r.polygon.max(0) - r.polygon.min(0)
        narrow = bool(dims.min() < 1.8)  # corridors: doors are seen at grazing angles, widths are the weakest measurement
        extra_w = 0.20 if narrow else 0.0
        for oi, o in enumerate(r.openings):
            w = r.walls[o.wall]
            # offset measured from the wall start p0 (CCW direction)
            # wall runs from vertex (o.wall-1) to vertex o.wall; its start coordinate along the wall axis:
            poly = r.polygon
            v0, v1 = poly[o.wall - 1], poly[o.wall]
            ax = 1 if w.orient == "V" else 0
            start = v0[ax]
            direction = np.sign(v1[ax] - v0[ax]) or 1.0
            c = 0.5 * (o.s0 + o.s1)
            off = float((c - start) * direction)
            lk = link_by_open.get((ri, oi))
            ops_j.append(dict(
                id=f"{r.id}.o{oi}", wall_id=f"{r.id}.w{o.wall}", type=o.kind,
                center_offset=U.measure(off, float(np.hypot(0.01, 0.75 * extra_w)), tier, "offset", xr),
                width=U.measure(o.s1 - o.s0, float(np.hypot(o.sigma_w, extra_w)), tier, "opening_width", xr),
                height=U.measure(o.z1 - o.z0, float(np.hypot(o.sigma_h, extra_w)), tier, "opening_height", xr),
                sill=U.measure(max(o.z0, 0.0), float(np.hypot(o.sigma_h, extra_w)), tier, "opening_height", xr),
                confidence=round(float(o.confidence), 3),
                quality=dict(incidence_cos=round(float(o.evidence.get("cosq", 0.0)), 3), wall_hits_inside=round(float(o.evidence.get("conflict", 0.0)), 3)),
                links_to=(rooms[lk[0]].id if lk else None)))
        surfaces = [dict(id=f"{r.id}.floor", kind="floor", area=U.measure(area, a_sig, tier, "area", xr)),
                    dict(id=f"{r.id}.ceiling", kind="ceiling", area=U.measure(area, a_sig, tier, "area", xr))]
        for i, w in enumerate(r.walls):
            L = float(np.linalg.norm(wall_ends[i][1] - wall_ends[i][0]))
            surfaces.append(dict(id=f"{r.id}.w{i}", kind="wall",
                                 area=U.measure(L * height, np.hypot(L * room_height_sigma(r), height * wall_length_sigma(r, i)),
                                                tier, "area", xr)))
        dmg = []
        lo_al = r.polygon.min(0)
        for d_ in getattr(r, "damage_json", []):
            d_ = dict(d_)
            uv = np.array(d_["polygon_uv"])
            uc, vc = uv.mean(0)
            sid = d_["surface_id"]
            if sid.endswith(".ceiling") or sid.endswith(".floor"):
                al = lo_al + np.array([uc, vc])
                xy = pl.R @ al + pl.p - origin
                z = float(height if sid.endswith(".ceiling") else 0.0)
            else:
                wi = int(sid.split(".w")[-1])
                p0, p1 = wall_ends[wi]
                dirv = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
                xy = p0 + dirv * uc
                z = float(vc)
            d_["location"] = dict(plan_xy=[round(float(xy[0]), 3), round(float(xy[1]), 3)], z=round(z, 3))
            dmg.append(d_)
        out_rooms.append(dict(
            id=r.id, name=r.name or None,
            polygon=[[round(float(x), 4), round(float(y), 4)] for x, y in pp],
            walls=walls_j, ceiling_height=h_m,
            floor_area=U.measure(area, a_sig, tier, "area", xr),
            openings=ops_j, surfaces=surfaces,
            damage=dmg, concealed_damage=getattr(r, "concealed_json", []),
            scope=getattr(r, "scope_json", []),
            notes=list(r.notes)))
        rot_deg = float(np.rad2deg(np.arctan2(pl.R[1, 0], pl.R[0, 0])))
        plan_rooms.append(dict(room_id=r.id, rotation_deg=round(rot_deg, 3),
                               translation=[round(float(pl.p[0] - origin[0]), 4), round(float(pl.p[1] - origin[1]), 4)]))
    # footprint
    polys = [Polygon(p - origin) for p in plan_polys]
    foot = unary_union([p.buffer(wall_t / 2.0, join_style=2) for p in polys])
    f_area = float(foot.area)
    sig_rooms = float(np.sqrt(sum(_area_sigma(r) ** 2 for r in rooms)))
    f_sig = float(np.hypot(sig_rooms, 0.03 * foot.length * 0.5))
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            a = polys[i].intersection(polys[j]).area
            if a > 0.02:
                overlaps.append(dict(a=rooms[i].id, b=rooms[j].id, area=round(float(a), 3)))
    adjacency = [dict(a=rooms[l.a].id, b=rooms[l.b].id, opening_a=f"{rooms[l.a].id}.o{l.oa}",
                      opening_b=f"{rooms[l.b].id}.o{l.ob}", source=l.source,
                      residual_m=round(float(l.residual), 4), consistent=bool(l.consistent)) for l in links]
    plan = dict(rooms=plan_rooms, adjacency=adjacency,
                footprint_area=U.measure(f_area, f_sig, tier, "footprint"),
                wall_thickness=round(float(wall_t), 3), overlaps=overlaps, drift=drift_info)
    return dict(schema_version=SCHEMA_VERSION,
                capture=dict(tier=tier, source=source, device=device, **{k: v for k, v in meta.items()
                                                                          if k in ("frames", "photos", "path_length")}),
                units="m", rooms=out_rooms, plan=plan,
                quality=dict(warnings=list(notes), hazards=list(meta.get("hazards", []))),
                timing={k: round(float(v), 3) for k, v in timings.items()})
