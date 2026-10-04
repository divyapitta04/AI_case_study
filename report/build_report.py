"""Build results/TECHNICAL_REPORT.pdf (<= 6 pages) + .md from results/benchmark and results/fixloop (real ARKitScenes data)."""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Table, TableStyle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "benchmark")
OUT = os.path.join(ROOT, "results")


def arch_figure(path):
    fig, ax = plt.subplots(figsize=(10, 3.1))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)

    def box(x, y, w, h, t, fc="#eef3f8", fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", fc=fc, ec="#345", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#345", lw=1.0))

    box(1, 21, 17, 8, "LiDAR tier\nARKit depth 256x192\n+ poses + intrinsics", "#e3f1e3")
    box(1, 11, 17, 8, "Video tier\nmp4 -> COLMAP SfM\n+ metric depth per key-frame", "#fff1d6")
    box(1, 1, 17, 8, "Photo tier\nper-room folders (2-8)\nEXIF + metric depth", "#fde2e2")
    box(25, 11, 17, 8, "RaySet + Frames\n(gravity frame, metric,\nrange + sigma per ray)", "#dfe8f6")
    for y in (25, 15, 5):
        arrow(18.5, y, 24.5, 15)
    box(48, 21, 21, 8, "Rooms: free-space carving +\npersistence watershed;\nrectilinear polygon + planes", "#f3e8fa")
    box(48, 11, 21, 8, "Plane-anchored pose correction\n(per chunk x,y,yaw,z; smoothed)", "#f3e8fa")
    box(48, 1, 21, 8, "Openings from ray evidence\n(W / beyond / dropout / colour)", "#f3e8fa")
    arrow(42.5, 15, 47.5, 25)
    arrow(42.5, 15, 47.5, 15)
    arrow(42.5, 15, 47.5, 5)
    box(76, 21, 22, 8, "Door pose graph over rooms\n(loop closure) -> stitched plan", "#e8f0e8")
    box(76, 11, 22, 8, "Damage regions -> concealed rules\n-> scope line items per surface", "#e8f0e8")
    box(76, 1, 22, 8, "CI on every measurement\n-> JSON + plan.png / svg", "#e8f0e8")
    arrow(69.5, 25, 75.5, 25)
    arrow(69.5, 15, 75.5, 15)
    arrow(69.5, 5, 75.5, 5)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)



def load():
    m = json.load(open(os.path.join(RES, "metrics.json")))
    fl = os.path.join(ROOT, "results", "fixloop", "summary.json")
    fix = json.load(open(fl)) if os.path.exists(fl) else None
    gt = json.load(open(os.path.join(ROOT, "data", "gt", "421337.json")))["rooms"][0]
    return m, fix, gt


def tbl(data, widths=None, fs=6.8, head=colors.HexColor("#dde6f0")):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), head),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


def mm(x):
    return "-" if x is None else f"{1000 * x:.0f}"


def build():
    m, fix, gt = load()
    sc = m["scores"]
    arch_figure(os.path.join(OUT, "architecture.png"))
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.2, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=3)
    h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=2)
    warn = ParagraphStyle("w", parent=body, backColor=colors.HexColor("#fff3cd"), borderColor=colors.HexColor("#c9a227"),
                          borderWidth=0.6, borderPadding=3, spaceAfter=5)
    doc = SimpleDocTemplate(os.path.join(OUT, "TECHNICAL_REPORT.pdf"), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.3 * cm, bottomMargin=1.2 * cm, title="floorscan technical report")
    S = []
    md = []

    def P(t, st=body):
        S.append(Paragraph(t, st))
        md.append(t.replace("<b>", "**").replace("</b>", "**").replace("<i>", "_").replace("</i>", "_").replace("<br/>", "\n") + "\n")

    def H(t):
        S.append(Paragraph(t, h1))
        md.append("## " + t + "\n")

    def T(rows, widths, fs=6.8):
        S.append(tbl(rows, widths, fs=fs))
        md.append("| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n" +
                  "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows[1:]))

    S.append(Paragraph("floorscan - floor plan, damage and scope from iPhone photos, video or LiDAR", ParagraphStyle(
        "t", parent=ss["Title"], fontSize=14, spaceAfter=2)))
    md.append("# floorscan - technical report\n")
    split = json.load(open(os.path.join(ROOT, "data", "SPLIT.json")))
    n_dev, n_test = len(split["dev"]), len(split["test"])
    P("<b>Why the data looks like this.</b> No iPhone was available, so the benchmark uses Apple's public <b>ARKitScenes</b> "
      f"dataset: {n_dev + n_test} real rooms captured with an iPad Pro (same LiDAR + ARKit stack as Pro iPhones) - 2-3 LiDAR "
      "sessions, the device video and 8 frames of it as a photo set per room - scored against <b>FARO laser scans</b> of the same "
      f"rooms. Rooms were drawn at random before any was run; {n_dev} DEV rooms were used for fixes and fits, {n_test} TEST rooms "
      "only for the final numbers. Consequences: no scale card in any video or photo set (scale comes from priors and the depth "
      "network), no staged damage (damage is evaluated on real photos of damaged walls, BD3), single rooms only (multi-room "
      "stitching untested on real data), no consumer-app export (head-to-head NOT RUN), no git history. "
      "Details: docs/WHY_THESE_RESULTS.md.", warn)

    H("1. Architecture")
    S.append(Image(os.path.join(OUT, "architecture.png"), width=18 * cm, height=18 * cm * 3.1 / 10))
    P("Every tier is reduced to the same object: calibrated camera frames plus a bag of <i>rays</i> (direction, range, sigma, "
      "dropout) in a gravity-aligned metric frame. LiDAR rays come from the ARKit depth map; video rays from COLMAP poses (exhaustive "
      "matching) plus Depth-Anything-V2 metric depth aligned to the SfM points; photo rays from the same depth network, with gravity "
      "from depth normals and yaw from homographies. Shared geometry: free-space carving, persistence watershed and a merge of regions "
      "joined wider than any door; rectilinear walls snapped to the innermost full-height plane covering most of the wall (recesses "
      "and furniture excluded); floor = lowest real plane, ceiling accepted only if it covers the room and is >= 2 m up (else "
      "'not observed' + prior); plane-anchored drift correction; openings from rays crossing wall planes; door-anchored pose graph "
      "for multi-room plans. Damage: candidates from 2 cm/px ortho-textures, verified and classified by DINOv2 features + a linear "
      "head trained on real photos; concealed-damage rules and scope lines keyed to surfaces. Every number is {value, ci95, sigma}. "
      "Code map: docs/CODE_WALKTHROUGH.md.")

    H("2. Benchmark data and ground truth")
    P("dataprep/fetch_sample.py downloads the pre-registered sample (data/SPLIT.json) and converts each session to our three input "
      "layouts. Ground truth (dataprep/room_gt.py, openings_gt.py, no shared code with the pipeline): each session is registered to "
      "the venue's laser scans; the laser stations inside the room give floor, ceiling and the four main wall faces (a written "
      "convention: innermost full-height face covering >= 40 % of the side; recesses and furniture excluded) and the openings the laser "
      "sees as openings; closed flush doors and recesses are ignore regions. Exclusions by rules fixed in advance: one room whose laser "
      "does not cover the captured room (replaced by the next random room) and one session that captured a different space. "
      "bench/metrics.py aligns each plan to the ground truth (k x 90 deg + translation) and scores it.")

    H("3. Results: share of measurements inside each case-study gate")
    rows = [["split", "tier", "captures", "walls in gate", "ceilings in gate", "openings within tol.", "footprint in gate",
             "median worst wall err"]]
    for part in ("test", "dev"):
        for tier in ("lidar", "video", "photos"):
            g = [v["gates"] for n, v in sc.items() if v.get("split") == part and n.endswith(tier)]
            if not g:
                continue
            wn = sum(x["walls_n"] + x.get("walls_missing", 0) for x in g)
            wo = sum(x["walls_ok"] for x in g)
            cn, co = sum(x["ceilings_n"] for x in g), sum(x["ceilings_ok"] for x in g)
            on, oo = sum(x["openings_gt"] + x["openings_phantom"] for x in g), sum(x["openings_ok"] for x in g)
            werr = [x["walls_max_abs_err"] for x in g if x["walls_max_abs_err"] is not None]
            rows.append([part.upper(), tier, str(len(g)), f"{wo}/{wn} ({100 * wo / max(wn, 1):.0f} %)",
                         f"{co}/{cn} ({100 * co / max(cn, 1):.0f} %)", f"{oo}/{on} ({100 * oo / max(on, 1):.0f} %)" if on else "-",
                         f"{sum(bool(x['footprint_pass']) for x in g)}/{len(g)}",
                         f"{1000 * float(np.median(werr)):.0f} mm" if werr else "-"])
    T(rows, [1.3 * cm, 1.3 * cm, 1.5 * cm, 2.6 * cm, 2.4 * cm, 2.6 * cm, 2.3 * cm, 2.4 * cm], fs=6.5)
    P("Gates (case study): LiDAR walls within max(2 cm, 0.5 %), ceiling within 1.5 cm, openings within 2 cm (missed and phantom "
      "openings count as misses); video within +-3 %; photos within +-8 %. TEST rows are the held-out result. Per-capture tables, "
      "per-wall errors and timing: results/benchmark/BENCHMARK_REPORT.md.")

    H("4. Repeatability and drift handling")
    rp = list(m.get("repeatability", {}).values())
    for part in ("test", "dev"):
        x = [r for r in rp if r.get("split") == part]
        if x:
            mx = [max(abs(w["diff"]) for w in r["walls"]) for r in x if r["walls"]] or [float("nan")]
            P(f"<b>{part.upper()}</b>: {len(x)} pairs of LiDAR sessions of the same room - walls within max(1 cm, 0.5 %) in "
              f"{sum(r['walls_ok'] for r in x)}/{len(x)} pairs, ceiling spread <= 1 cm in {sum(r['height_ok'] for r in x)}/{len(x)}, "
              f"median largest wall difference {1000 * float(np.median(mx)):.0f} mm.")
    ab = m.get("ablation", {})
    if ab:
        rows = [["capture (TEST)", "drift handling", "walls in gate", "max wall err", "ceiling err", "footprint err"]]
        for n, d in ab.items():
            for k in ("off", "on"):
                x = d[k]
                rows.append([n, "OFF (poses as delivered)" if k == "off" else "ON", f"{x['walls_ok']}/{x['walls_n']}",
                             f"{mm(x['walls_max_abs_err'])} mm", f"{mm(x['ceilings_max_abs_err'])} mm", f"{x['footprint_err_pct']:+.1f} %"])
        T(rows, [3.4 * cm, 3.6 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm, 2.4 * cm])
    P("ARKit poses are never used as-is: frames are chunked and each chunk gets an (x, y, yaw, z) correction putting its points on the "
      "fitted planes (joint over chunks, smoothness prior, 3 iterations); multi-room captures add a door-anchored pose graph.")

    H("5. The fix loop (on real data, DEV room)")
    if fix:
        P(fix.get("story", ""))
        rows = [["", "before", "predicted", "after"],
                ["worst wall error (both captures)", f"{1000 * fix['before']:.0f} mm", f"{1000 * fix['predicted']:.0f} mm",
                 f"{1000 * fix['after']:.0f} mm"],
                ["walls in the 2 cm gate", f"{fix['walls_ok_before']}/{fix['walls_n']}", "0-2/8", f"{fix['walls_ok_after']}/{fix['walls_n']}"]]
        T(rows, [6.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
        P("Declaration, laser-registered evidence and post-mortem: fixloop/FIX_DECLARATION.md; diff: fixloop/fix.diff. Further fixes "
          "found on the other DEV rooms (room merge, floor/ceiling rules, height and depth-scale fits) are listed in docs/DISCLOSURES.md.")

    H("6. Calibration of the intervals")
    rows = [["split", "tier", "n", "inside 95 % CI", "error > 3 half-widths"]]
    for part in ("test", "dev"):
        for tier in ("lidar", "video", "photos"):
            c = [e for n, v in sc.items() if v.get("split") == part and n.endswith(tier) for e in v.get("calib", [])]
            if c:
                rows.append([part.upper(), tier, str(len(c)), f"{100 * np.mean([e['inside'] for e in c]):.0f} %",
                             f"{100 * np.mean([abs(e['err']) > 3 * max(e['half'], 1e-6) for e in c]):.0f} %"])
    T(rows, [1.6 * cm, 1.6 * cm, 1.2 * cm, 3.0 * cm, 3.4 * cm])
    P("value +- 1.96 k sqrt(sigma_fit^2 + floor^2 + (rel x value)^2); k fitted per tier and kind on the DEV captures "
      "(bench/calibrate.py --from-results; DEV rows above were produced with k = 1). The TEST rows show whether that calibration "
      "transfers to unseen rooms.")

    H("7. Damage")
    bd = os.path.join(ROOT, "results", "damage", "bd3_metrics.json")
    if os.path.exists(bd):
        b = json.load(open(bd))
        P(f"Classifier on <b>{b['n_test']} real photos</b> it never saw (BD3 test split): class accuracy <b>{100 * b['accuracy']:.1f} %</b> "
          f"over {', '.join(b['classes'])}; damage-vs-clean precision {100 * b['detection_precision']:.1f} %, recall "
          f"{100 * b['detection_recall']:.1f} % (results/damage/BD3_EVAL.md).")
    fpl = [v["gates"]["damage_fp"] for n, v in sc.items() if v.get("split") == "test" and n.endswith("lidar")]
    if fpl:
        P(f"On the undamaged benchmark rooms every detection is a false positive: TEST LiDAR captures report "
          f"{np.mean(fpl):.1f} detections on average (median {np.median(fpl):.0f}). Metric extent of real damage cannot be "
          "scored without staged damage and measured ground truth, which no public dataset provides.")

    H("8. Known limits")
    P("Video and photos have no scale card in this dataset, so their scale rests on the depth network (15 % spread across DEV "
      "captures) and the ceiling/door priors - the main reason those tiers miss their gates; with the protocol's card the scale error "
      "is 1-2 %. Sessions that never look at the ceiling report a prior with a wide interval. Multi-room stitching, mirrors, wet "
      "floors and low light are not exercised by this sample. Full list: docs/KNOWN_FAILURES.md; what is missing: "
      "docs/REAL_CAPTURE_TODO.md.")
    doc.build(S)
    open(os.path.join(OUT, "TECHNICAL_REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
    from pypdf import PdfReader
    n = len(PdfReader(os.path.join(OUT, "TECHNICAL_REPORT.pdf")).pages)
    print("pages:", n)
    assert n <= 6, "technical report exceeds 6 pages"


if __name__ == "__main__":
    build()
