"""Command line: one command per capture.

    python -m floorscan run <capture_dir> --out <out_dir> [--tier auto|lidar|video|photos]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema", "floorscan.schema.json")


def canonical_hash(out: dict) -> str:
    d = {k: v for k, v in out.items() if k != "timing"}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def validate(out: dict) -> list:
    import jsonschema
    with open(SCHEMA) as fh:
        schema = json.load(fh)
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(out)]


def run(path: str, out_dir: str, tier: str = None, damage: bool = True, quiet: bool = False) -> dict:
    from .config import Config
    from .pipeline.run import run_capture
    from .render import render_plan

    cfg = Config()
    cfg.damage = damage
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    out, rec, places, links = run_capture(path, cfg, tier=None if tier in (None, "auto") else tier)
    out["timing"]["total"] = round(time.time() - t0, 3)
    errs = validate(out)
    if errs:
        out["quality"]["warnings"].append(f"schema validation: {len(errs)} issue(s), first: {errs[0]}")
    with open(os.path.join(out_dir, "plan.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    render_plan(out, os.path.join(out_dir, "plan.png"), os.path.join(out_dir, "plan.svg"))
    if not quiet:
        f = out["plan"]["footprint_area"]
        print(f"tier={out['capture']['tier']} rooms={len(out['rooms'])} footprint={f['value']:.2f} m2 "
              f"[{f['ci95'][0]:.2f}, {f['ci95'][1]:.2f}] time={out['timing']['total']:.1f}s hash={canonical_hash(out)}")
        for r in out["rooms"]:
            h = r["ceiling_height"]
            print(f"  {r.get('name') or r['id']}: area {r['floor_area']['value']:.2f} m2, ceiling {h['value']:.3f} m "
                  f"[{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}], openings {len(r['openings'])}, damage {len(r['damage'])}, "
                  f"flags {len(r['concealed_damage'])}, scope items {len(r['scope'])}")
        for w in out["quality"]["warnings"][:6]:
            print("  note:", w)
        print(f"  wrote {out_dir}/plan.json, plan.png, plan.svg")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floorscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline on one capture folder")
    r.add_argument("capture")
    r.add_argument("--out", required=True)
    r.add_argument("--tier", default="auto", choices=["auto", "lidar", "video", "photos"])
    r.add_argument("--no-damage", action="store_true")
    v = sub.add_parser("validate", help="validate a plan.json against the published schema")
    v.add_argument("json")
    rd = sub.add_parser("render", help="re-render plan.png/svg from a plan.json")
    rd.add_argument("json")
    rd.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.capture, a.out, a.tier, damage=not a.no_damage)
    elif a.cmd == "validate":
        errs = validate(json.load(open(a.json)))
        print("OK" if not errs else "\n".join(errs))
        sys.exit(1 if errs else 0)
    elif a.cmd == "render":
        from .render import render_plan
        os.makedirs(a.out, exist_ok=True)
        render_plan(json.load(open(a.json)), os.path.join(a.out, "plan.png"), os.path.join(a.out, "plan.svg"))


if __name__ == "__main__":
    main()
