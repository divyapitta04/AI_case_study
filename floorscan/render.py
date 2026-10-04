"""Render the stitched plan (PNG + SVG) from the output JSON alone."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Polygon as MPoly  # noqa: E402

DMG_COL = {"water_stain": "#b5651d", "mold": "#2f4f2f", "hole": "#222222", "peeling": "#8a6fb0", "crack": "#c0392b"}



def render_plan(out: Dict, png_path: Optional[str] = None, svg_path: Optional[str] = None, title: Optional[str] = None,
                show_ci: bool = True):
    rooms = out["rooms"]
    allp = np.concatenate([np.array(r["polygon"]) for r in rooms]) if rooms else np.zeros((1, 2))
    lo, hi = allp.min(0) - 0.8, allp.max(0) + 0.8
    w, h = hi - lo
    fig, ax = plt.subplots(figsize=(max(6, min(14, w * 1.1)), max(5, min(12, h * 1.1))))
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    pal = ["#f6efe6", "#e8f0f6", "#eef5e8", "#f6e8ee", "#f3f0dc", "#e9e3f4"]
    wall_by_id = {}
    for ri, r in enumerate(rooms):
        P = np.array(r["polygon"])
        ax.add_patch(MPoly(P, closed=True, facecolor=pal[ri % len(pal)], edgecolor="none", zorder=1))
    for ri, r in enumerate(rooms):
        for wl in r["walls"]:
            p0, p1 = np.array(wl["p0"]), np.array(wl["p1"])
            wall_by_id[wl["id"]] = (p0, p1)
            ls = "-" if wl["observed"] else (0, (3, 3))
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#222", lw=3.2, ls=ls, solid_capstyle="butt", zorder=3)
    # openings
    for r in rooms:
        for o in r["openings"]:
            p0, p1 = wall_by_id[o["wall_id"]]
            d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            c = p0 + d * o["center_offset"]["value"]
            wd = o["width"]["value"]
            a, b = c - d * wd / 2, c + d * wd / 2
            if o["type"] == "window":
                ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=4.2, zorder=4, solid_capstyle="butt")
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#3a7bd5", lw=1.4, zorder=5, solid_capstyle="butt")
            else:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=4.4, zorder=4, solid_capstyle="butt")
                n = np.array([-d[1], d[0]])
                # swing into the room (towards the polygon centroid)
                cen = np.array(r["polygon"]).mean(0)
                if (cen - c) @ n < 0:
                    n = -n
                an = np.degrees(np.arctan2(n[1], n[0]))
                a0 = np.degrees(np.arctan2(d[1], d[0]))
                diff = ((an - a0 + 180) % 360) - 180
                t1, t2 = (a0, a0 + diff) if diff > 0 else (a0 + diff, a0)
                ax.plot([a[0], a[0] + n[0] * wd], [a[1], a[1] + n[1] * wd], color="#555", lw=0.8, zorder=5)
                ax.add_patch(Arc(a, 2 * wd, 2 * wd, theta1=t1, theta2=t2, color="#888", lw=0.7, zorder=5))
    # dimensions + labels
    for r in rooms:
        P = np.array(r["polygon"])
        cen = P.mean(0)
        for wl in r["walls"]:
            p0, p1 = np.array(wl["p0"]), np.array(wl["p1"])
            mid = 0.5 * (p0 + p1)
            d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            n = np.array([-d[1], d[0]])
            if (cen - mid) @ n < 0:
                n = -n
            txt = f"{wl['length']['value']:.2f} m"
            if show_ci:
                lo_, hi_ = wl["length"]["ci95"]
                txt += f"\n±{(hi_ - lo_) / 2:.2f}"
            ang = np.degrees(np.arctan2(d[1], d[0]))
            ang = ang if -90 <= ang <= 90 else ang + 180
            ax.text(*(mid + n * 0.28), txt, ha="center", va="center", fontsize=6.0, rotation=ang, color="#111", zorder=6)
        name = r.get("name") or r["id"]
        a_ = r["floor_area"]
        h_ = r["ceiling_height"]
        ax.text(*cen, f"{name}\n{a_['value']:.1f} m²  (±{(a_['ci95'][1] - a_['ci95'][0]) / 2:.1f})\nceiling {h_['value']:.2f} m "
                f"(±{(h_['ci95'][1] - h_['ci95'][0]) / 2:.2f})", ha="center", va="center", fontsize=7.5, weight="bold",
                color="#222", zorder=6)
    # damage
    for r in rooms:
        for dmg in r["damage"]:
            loc = dmg.get("location")
            if not loc:
                continue
            x, y = loc["plan_xy"]
            col = DMG_COL.get(dmg["class"], "k")
            ax.scatter([x], [y], s=60 + 120 * dmg["extent"]["area"]["value"], c=col, alpha=0.55, zorder=7,
                       edgecolors="white", linewidths=0.8,
                       marker="s" if dmg["surface_id"].endswith((".floor", ".ceiling")) else "o")
        for cf in r["concealed_damage"]:
            sid = cf["surface_id"]
            if sid in wall_by_id:
                p0, p1 = wall_by_id[sid]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#d62728", lw=1.6, ls=(0, (1, 1.5)), zorder=7)
    foot = out["plan"]["footprint_area"]
    ax.set_title(title or f"{out['capture']['tier']} tier — footprint {foot['value']:.1f} m² "
                 f"(95 % CI {foot['ci95'][0]:.1f}–{foot['ci95'][1]:.1f}) — {len(rooms)} rooms", fontsize=9)
    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=DMG_COL[k], markersize=7, label=k)
           for k in DMG_COL] + [Line2D([0], [0], color="#d62728", ls=":", label="concealed-damage flag (wall)"),
                                Line2D([0], [0], color="#3a7bd5", label="window"),
                                Line2D([0], [0], color="#222", ls=(0, (3, 3)), label="wall not observed")]
    ax.legend(handles=leg, loc="lower right", fontsize=6, frameon=True)
    # scale bar
    x0, y0 = lo[0] + 0.3, lo[1] + 0.3
    ax.plot([x0, x0 + 1], [y0, y0], color="k", lw=2)
    ax.text(x0 + 0.5, y0 + 0.12, "1 m", ha="center", fontsize=7)
    ax.axis("off")
    fig.tight_layout()
    if png_path:
        fig.savefig(png_path, dpi=150)
    if svg_path:
        fig.savefig(svg_path)
    plt.close(fig)
