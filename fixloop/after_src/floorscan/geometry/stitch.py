"""Whole-property plan: door-anchored pose graph over rooms (loop closure) + overlap checks.

Rooms are rectilinear polygons in their own (Manhattan-aligned) frame.  Each room gets a placement
(rotation k*90 deg, translation).  Doors that connect two rooms give two constraints each:
  tangential: door centres coincide along the wall,
  normal:     the two inner wall faces are one wall-thickness apart.
Cycles in the door graph (e.g. hall-living-kitchen) are what close the loop; translations are solved by
weighted least squares, rotations are fixed by the door-wall orientation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .room import Opening, RoomModel


def rot(k: float) -> np.ndarray:
    c, s = np.cos(k), np.sin(k)
    return np.array([[c, -s], [s, c]])


@dataclass
class Link:
    a: int  # room index
    oa: int  # opening index within room a
    b: int
    ob: int
    source: str = "geometry"  # geometry | photo
    residual: float = 0.0
    weight_t: float = 1.0
    consistent: bool = True
    sigma_tan: Optional[float] = None  # per-link override (inferred doorways are located only to ~0.3 m)


@dataclass
class Placement:
    R: np.ndarray = field(default_factory=lambda: np.eye(2))
    p: np.ndarray = field(default_factory=lambda: np.zeros(2))


def door_geom(room: RoomModel, op: Opening) -> Tuple[np.ndarray, np.ndarray]:
    """Door centre on the wall (room-local aligned frame) and wall outward normal."""
    w = room.walls[op.wall]
    s = 0.5 * (op.s0 + op.s1)
    c = np.array([w.coord, s]) if w.orient == "V" else np.array([s, w.coord])
    return c, np.array(w.out_n)


def match_links(rooms: Sequence[RoomModel], place: Sequence[Placement], max_dist: float = 1.2) -> List[Link]:
    """Mutual-best door pairs between different rooms from the current placement (global-frame tiers)."""
    cand = []
    for i, ra in enumerate(rooms):
        for ia, oa in enumerate(ra.openings):
            if oa.kind not in ("door", "passage"):
                continue
            ca, na = door_geom(ra, oa)
            Pa, Na = place[i].R @ ca + place[i].p, place[i].R @ na
            for j, rb in enumerate(rooms):
                if j <= i:
                    continue
                for ib, ob in enumerate(rb.openings):
                    if ob.kind not in ("door", "passage"):
                        continue
                    cb, nb = door_geom(rb, ob)
                    Pb, Nb = place[j].R @ cb + place[j].p, place[j].R @ nb
                    if Na @ Nb > -0.9:
                        continue
                    dv = Pb - Pa
                    if dv @ Na < -0.25 or dv @ Na > 0.6:  # b must lie just beyond a
                        continue
                    tang = abs(dv @ np.array([-Na[1], Na[0]]))
                    if tang > max_dist:
                        continue
                    wd = abs((oa.s1 - oa.s0) - (ob.s1 - ob.s0))
                    cand.append((tang + 0.5 * wd, i, ia, j, ib))
    cand.sort()
    used_a, used_b, links = set(), set(), []
    for _, i, ia, j, ib in cand:
        if (i, ia) in used_a or (j, ib) in used_b or (i, ia) in used_b or (j, ib) in used_a:
            continue
        used_a.add((i, ia))
        used_b.add((j, ib))
        links.append(Link(i, ia, j, ib))
    return links


def _link_rows(rooms, place, link: Link, t: float):
    ra, rb = rooms[link.a], rooms[link.b]
    ca, na = door_geom(ra, ra.openings[link.oa])
    cb, nb = door_geom(rb, rb.openings[link.ob])
    Ra, Rb = place[link.a].R, place[link.b].R
    Na = Ra @ na
    tau = np.array([-Na[1], Na[0]])
    # P_b - P_a = (Rb cb + pb) - (Ra ca + pa)
    d0 = (Rb @ cb + place[link.b].p) - (Ra @ ca + place[link.a].p)
    return Na, tau, d0


def solve_translations(rooms: Sequence[RoomModel], place: List[Placement], links: List[Link], t: float,
                       sigma_tan: float = 0.015, sigma_norm: float = 0.03, sigma_prior: float = 1.5,
                       anchor: int = 0, iters: int = 2) -> None:
    """Weighted LS for room translations; modifies `place` in place; sets link residuals."""
    n = len(rooms)
    active = [l for l in links]
    for _ in range(iters):
        rows, rhs, wts = [], [], []
        for l in active:
            Na, tau, d0 = _link_rows(rooms, place, l, t)
            for vec, target, sg in ((tau, 0.0, l.sigma_tan or sigma_tan), (Na, t, sigma_norm)):
                r = np.zeros(2 * n)
                r[2 * l.b:2 * l.b + 2] = vec
                r[2 * l.a:2 * l.a + 2] = -vec
                rows.append(r)
                # current: (vec . d0) ; we solve for delta so that vec.(d0 + dpb - dpa) = target
                rhs.append(target - vec @ d0)
                wts.append(1.0 / sg)
        for k in range(n):
            for ax in range(2):
                r = np.zeros(2 * n)
                r[2 * k + ax] = 1.0
                rows.append(r)
                rhs.append(0.0)
                wts.append(1.0 / (1e-3 if k == anchor else sigma_prior))
        A = np.array(rows) * np.array(wts)[:, None]
        b = np.array(rhs) * np.array(wts)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        for k in range(n):
            place[k].p = place[k].p + sol[2 * k:2 * k + 2]
        # residual check, drop gross outliers once
        bad = []
        for l in active:
            Na, tau, d0 = _link_rows(rooms, place, l, t)
            l.residual = float(np.hypot(tau @ d0, Na @ d0 - t))
            if l.residual > 0.25:
                bad.append(l)
        if not bad or len(bad) == len(active):
            break
        for l in bad:
            l.consistent = False
        active = [l for l in active if l.consistent]


def bfs_place(rooms: Sequence[RoomModel], links: List[Link], t: float, anchor: int = 0):
    """Initial placement for tiers without a shared frame: rotate/translate rooms outward from the anchor
    so that linked door walls face each other.  Returns (placements, reached set)."""
    n = len(rooms)
    place = [Placement() for _ in range(n)]
    reached = {anchor}
    frontier = [anchor]
    adj: Dict[int, List[Tuple[Link, bool]]] = {i: [] for i in range(n)}
    for l in links:
        adj[l.a].append((l, True))
        adj[l.b].append((l, False))
    while frontier:
        a = frontier.pop(0)
        for l, fwd in adj[a]:
            b = l.b if fwd else l.a
            if b in reached:
                continue
            oa = l.oa if fwd else l.ob
            ob = l.ob if fwd else l.oa
            ra, rb = rooms[a], rooms[b]
            ca, na = door_geom(ra, ra.openings[oa])
            cb, nb = door_geom(rb, rb.openings[ob])
            Na = place[a].R @ na
            ang = np.arctan2(-Na[1], -Na[0]) - np.arctan2(nb[1], nb[0])
            ang = np.round(ang / (np.pi / 2)) * (np.pi / 2)
            Rb = rot(ang)
            Pa = place[a].R @ ca + place[a].p
            pb = Pa + t * Na - Rb @ cb
            place[b] = Placement(Rb, pb)
            reached.add(b)
            frontier.append(b)
    return place, reached


def polygons_plan(rooms: Sequence[RoomModel], place: Sequence[Placement]):
    from shapely.geometry import Polygon
    out = []
    for r, pl in zip(rooms, place):
        pts = (pl.R @ r.polygon.T).T + pl.p
        out.append(Polygon(pts))
    return out
