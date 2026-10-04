# Functionality walkthrough

What the system does, from the user's side: what you give it, what you get back, how to read it, and how it is checked.
For the code behind each step see `docs/CODE_WALKTHROUGH.md`.

## 1. The job

Turn a phone capture of a property into a **dimensioned floor plan** with **damage, concealed-damage flags and a scope of
work**, and put a **95 % confidence interval on every number**. One command, whatever the capture:

```
python -m floorscan run <capture folder> --out <output folder>
```

## 2. Three ways to capture (tiers)

| tier | what the capture folder holds | hardware | scale comes from |
|---|---|---|---|
| **LiDAR** | depth maps + camera poses + intrinsics + RGB (Polycam "raw data" export, or our `floorscan-lidar-v1` layout) | Pro iPhones / iPad Pro | the depth sensor (metric) |
| **Video** | one walkthrough clip (`.mp4` / `.mov`) | any iPhone 15+ | printed scale card filmed first > door height > ceiling-height prior |
| **Photos** | one sub-folder per room, 2-8 JPEGs each; doorway shots named `*_to_<room>.jpg` | any iPhone 15+ | scale card (optional) > door height > depth network |

The tier is detected from the folder contents (`--tier` overrides it). How to capture: `capture/PROTOCOL.md`; which device gives
what accuracy: `capture/DEVICE_MATRIX.md`.

## 3. A real run, step by step

The repository ships one real room from Apple's ARKitScenes dataset in `data/` (two LiDAR captures, a video and a photo set of
the same room) plus its laser-scan ground truth in `data/gt/`.

```
> python -m floorscan run data/42444949_lidar --out out/49
tier=lidar rooms=1 footprint=16.57 m2 [15.99, 17.15] time=25.3s hash=ffbb18249435f13d
  room_1: area 15.61 m2, ceiling 3.040 m [3.018, 3.063], openings 8, damage 11, flags 7, scope items 17
  wrote out/49/plan.json, plan.png, plan.svg
```

What happened inside, in plain words:

1. **Read the capture** – every depth pixel becomes a ray from the camera (direction + measured distance + its uncertainty);
   noisy pixels at depth edges are dropped.
2. **Find floor, ceiling and the room axes** – histograms of point heights give floor and ceiling; the dominant wall direction
   gives the room's orientation.
3. **Find the rooms** – the space the rays flew through is "free"; a watershed on that free space separates rooms at doorways.
4. **Fit the walls** – each wall snaps to a plane of the point cloud. Furniture (does not reach the ceiling) is ignored, and
   recesses such as bay windows and alcoves do not pull the wall outwards (the innermost full-height plane covering most of the
   wall is taken).
5. **Correct drift** – phone tracking drifts by centimetres; the frames are cut into short chunks and each chunk is nudged so its
   points sit on the fitted planes. Steps 4-5 repeat 3 times.
6. **Openings** – where rays pass through a wall plane (or vanish in glass), there is a door, passage or window; its width,
   height, sill and position are measured.
7. **Stitch** – in multi-room captures, doors seen from both sides tie the rooms together (a pose graph); here there is one room.
8. **Damage** – each wall, floor and ceiling is unrolled into a 2 cm/pixel image; stains, mould and holes are found by colour,
   texture and depth rules.
9. **Reasoning** – rules turn visible damage into concealed-damage flags ("water stain on the ceiling → check the cavity above")
   and scope line items per surface ("patch hole in drywall, 1 ea").
10. **Intervals and output** – every number gets a 95 % interval; the plan is validated against the published schema and drawn.

## 4. What comes out

`plan.png` / `plan.svg`: the drawn plan (walls with length ± interval, openings, damage markers, flagged surfaces).
`plan.json` (schema `schema/floorscan.schema.json`), abridged from the run above:

```json
{
 "schema_version": "floorscan-1.0",
 "capture": {"tier": "lidar", "source": "floorscan-lidar-v1", "frames": 279},
 "rooms": [{
   "id": "room_1",
   "walls": [{"id": "room_1.w0", "p0": [0, 0], "p1": [4.1993, 0],
              "length": {"value": 4.1993, "ci95": [4.1703, 4.2283], "sigma": 0.0148}, "observed": true}, "..."],
   "ceiling_height": {"value": 3.0404, "ci95": [3.0181, 3.0627], "sigma": 0.0114},
   "floor_area":     {"value": 15.6061, "ci95": [15.4168, 15.7954], "sigma": 0.0966},
   "openings": [{"id": "room_1.o0", "wall_id": "room_1.w0", "type": "door",
                 "width": {"value": 0.6832, "ci95": [0.6627, 0.7037]}, "height": {"value": 1.9023, "...": "..."},
                 "sill": {"value": 0.0403, "...": "..."}, "center_offset": {"value": 3.8215, "...": "..."}}, "..."],
   "damage": [{"id": "room_1.dmg0", "surface_id": "room_1.w0", "class": "water_stain",
               "extent": {"width": {"value": 0.20, "...": "..."}, "height": {"value": 0.32, "...": "..."}}}, "..."],
   "concealed_damage": [{"id": "room_1.cd0", "rule": "R-HOLE-04", "surface_id": "room_1.w1",
                         "triggered_by": ["room_1.dmg1"], "description": "Opening in the wall surface: cavity ... is exposed; ..."}],
   "scope": [{"id": "room_1.sc0", "surface_id": "room_1.w1", "code": "HOLE-PATCH", "description": "Patch hole in drywall",
              "quantity": {"value": 1.0, "ci95": [1.0, 1.0]}, "evidence": ["room_1.dmg1"]}]
 }],
 "plan": {"adjacency": [], "footprint_area": {"value": 16.5704, "ci95": [15.9894, 17.1514]},
          "drift": {"method": "plane-anchored per-room pose refinement + door-anchored room pose graph (loop closure)", "...": "..."}},
 "quality": {"warnings": ["..."], "hazards": {"...": "..."}},
 "timing": {"total": 25.3}
}
```

How to read it:
* Every measurement is `{value, ci95, sigma}` in metres / m². The interval is wider when the tier or the evidence is weaker
  (a photo-tier wall interval is more than 5x a LiDAR one; a wall no photo saw is `observed: false` with a wide prior).
* `quality.warnings` says what the system could not do: partial reconstruction, which scale source was used, unobserved walls.
* Damage, flags and scope are linked by ids: scope item → `evidence` damage id → `surface_id` wall.
* **Caution:** this real room has no damage; the 11 damage regions in this run are false positives (furniture, pictures,
  shadows) and so are the flags and scope lines built on them. The damage detector is the weakest part on real data.

## 5. How it is checked against reality

`python bench/run_bench.py` runs every capture in `data/` (9 real ARKitScenes rooms, 4 DEV + 5 held-out TEST), compares each
with the laser ground truth (`data/gt/`) and writes `results/benchmark/BENCHMARK_REPORT.md`. Held-out TEST result:

| tier | walls in gate | ceilings in gate | median worst wall error |
|---|---|---|---|
| LiDAR (15 sessions) | 10 % | 57 % | 29 cm |
| video (3 clips, no scale card) | 0 % | 0 % | 1.4 m |
| photos (5 sets, no scale card) | 30 % | 60 % | 1.2 m |

It also reports per-wall errors, repeatability between sessions of the same room, the drift ablation (correction ON vs poses
as delivered), interval coverage and timing. Why the numbers are what they are: `docs/WHY_THESE_RESULTS.md`.

## 6. The fix loop

The worst gate on the real room was LiDAR wall length (worst error 224 mm). `fixloop/diagnose.py` registered our points onto the
laser scan and showed that the error came from walls snapping into a bay window and a chimney alcove. The fix (recess-aware plane
choice) was declared with a predicted result before it was written (`fixloop/FIX_DECLARATION.md`), and
`python fixloop/run_fixloop.py` reruns before and after: 224 → 37 mm worst error, 0/8 → 4/8 walls in the gate
(`results/fixloop/FIXLOOP_REPORT.md`).

## 7. Other commands

| command | does |
|---|---|
| `python -m floorscan validate out/plan.json` | checks a plan against the schema (`OK` or the errors) |
| `python -m floorscan render out/plan.json --out out/` | redraws `plan.png` / `plan.svg` from the JSON |
| `python -m floorscan run <dir> --out <out> --no-damage` | geometry only |
| `python bench/headtohead.py --export <app export> --app "<name version>"` | our LiDAR plan vs a consumer app's export (table NOT RUN without one) |
| `python dataprep/fetch_sample.py` | re-downloads the ARKitScenes sample + laser scans and rebuilds `data/` |
| `python report/build_report.py` | rebuilds `results/TECHNICAL_REPORT.pdf` |
| `python -m pytest -q tests` | 15 tests on real captures, incl. real iPhone 15 Pro photos (about 3 minutes) |
