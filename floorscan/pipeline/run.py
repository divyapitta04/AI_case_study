"""One command per capture: tier detection -> reconstruction -> plan JSON."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from ..config import Config
from ..geometry import openings as op_mod
from ..geometry import stitch as st
from ..ingest import detect_tier
from .. import damage as damage_mod
from .. import reasoning
from .assemble import assemble
from .globalfit import Reconstruction, _lookup_label, reconstruct_global


def _detect_openings_global(rec: Reconstruction):
    cap = rec.cap
    for rm in rec.rooms:
        own = rm.label

        def beyond_ok(p_aligned, own=own):
            lab = _lookup_label(rec.lab_full, rec.grid, np.asarray(p_aligned)[None, :])[0]
            return lab > 0 and lab != own

        rm.openings = op_mod.detect_openings(cap, rm, beyond_ok=beyond_ok,
                                             photo_mode=(cap.meta.get("tier") != "lidar"),
                                             cos_min=(0.30 if cap.meta.get("tier") == "lidar" else 0.35),
                                             min_area=(0.12 if cap.meta.get("tier") == "lidar" else 0.10))


def _stitch_global(rec: Reconstruction, cfg: Config):
    rooms = rec.rooms
    places = [st.Placement() for _ in rooms]
    links = st.match_links(rooms, places)
    # wall thickness from jamb evidence where available
    ts = []
    for l in links:
        for rm, oi in ((rooms[l.a], l.oa), (rooms[l.b], l.ob)):
            ev = rm.openings[oi].evidence
            if ev.get("thickness", -1) > 0.05 and ev.get("jamb", 0) >= 50:
                ts.append(ev["thickness"])
    t = float(np.median(ts)) if ts else cfg.default_wall_thickness
    t = float(np.clip(t, 0.06, 0.30))
    res0 = []
    for l in links:
        Na, tau, d0 = st._link_rows(rooms, places, l, t)
        res0.append(np.hypot(tau @ d0, Na @ d0 - t))
    if cfg.door_graph and links:
        anchor = 0
        st.solve_translations(rooms, places, links, t, anchor=anchor)
    else:
        for l, r0 in zip(links, res0):
            l.residual = float(r0)
    res1 = [l.residual for l in links]
    info = dict(
        method="plane-anchored per-room pose refinement + door-anchored room pose graph (loop closure)",
        local_planar=bool(cfg.local_planar), door_graph=bool(cfg.door_graph),
        links=len(links), loops=max(0, len(links) - (len(rooms) - 1)),
        door_residual_rms_before=round(float(np.sqrt(np.mean(np.square(res0)))) if res0 else 0.0, 4),
        door_residual_rms_after=round(float(np.sqrt(np.mean(np.square(res1)))) if res1 else 0.0, 4),
        wall_thickness_source="jamb evidence" if ts else "default")
    return places, links, t, info


def run_global(cap, cfg: Config, tier: str, source: str, device: Optional[str] = None,
               extra: Optional[dict] = None, scale_rel: float = 0.0):
    T = {}
    rec = reconstruct_global(cap, cfg)
    t0 = time.time()
    _detect_openings_global(rec)
    T["openings"] = time.time() - t0
    t0 = time.time()
    places, links, wall_t, drift = _stitch_global(rec, cfg)
    T["stitch"] = time.time() - t0
    T.update(rec.timings)
    meta = dict(frames=cap.n_frames)
    meta.update(extra or {})
    if cfg.damage:
        t0 = time.time()
        for rm in rec.rooms:
            regs = damage_mod.analyse_room(cap, rm, tier, extra_rel=scale_rel)
            rm.regs = regs
            rm.concealed_json = reasoning.concealed_flags(rm.id, regs, tier, extra_rel=scale_rel, wall_height=rm.height)
        T["damage"] = time.time() - t0
    out = assemble(tier, source, rec.rooms, places, links, wall_t, meta, T, rec.notes, drift, device,
                   extra_rel={rm.id: scale_rel for rm in rec.rooms})
    if cfg.damage:
        for rm, rj in zip(rec.rooms, out["rooms"]):
            surf = {s_["id"]: s_ for s_ in rj["surfaces"]}
            rj["scope"] = reasoning.scope_items(rm.id, rm.regs, rm.concealed_json, surf, tier, extra_rel=scale_rel)
    return out, rec, places, links


def run_capture(path: str, cfg: Optional[Config] = None, tier: Optional[str] = None):
    cfg = cfg or Config()
    tier = tier or detect_tier(path)
    t0 = time.time()
    if tier == "lidar":
        from ..ingest.lidar import load_native, load_polycam
        if os.path.isdir(os.path.join(path, "keyframes")):
            cap = load_polycam(path)
        else:
            cap = load_native(path)
        out, rec, places, links = run_global(cap, cfg, "lidar", cap.meta.get("source", "lidar"))
    elif tier == "video":
        from .video_tier import run_video
        out, rec, places, links = run_video(path, cfg)
    elif tier == "photos":
        from .photo_tier import run_photos
        out, rec, places, links = run_photos(path, cfg)
    else:
        raise ValueError(tier)
    out["timing"]["total"] = round(time.time() - t0, 3)
    return out, rec, places, links
