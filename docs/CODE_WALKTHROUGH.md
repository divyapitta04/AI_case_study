# Code walkthrough

How the code is organised and what each part does, in the order data flows through it. Read
`docs/FUNCTIONALITY_WALKTHROUGH.md` first if you want the *what* before the *how*.

```
capture folder ──► ingest ──► Capture (frames + rays) ──► geometry ──► RoomModels ──► assemble ──► plan.json ──► render
 (lidar/video/photos)  floorscan/ingest   floorscan/frames.py   floorscan/geometry   + damage + reasoning   schema/   plan.png/svg
                       + pipeline/*_tier                         + pipeline/globalfit
```

## 1. Entry point — `floorscan/cli.py`, `floorscan/__main__.py`

* `python -m floorscan run <capture> --out <dir> [--tier auto|lidar|video|photos] [--no-damage]`
  calls `cli.run()`: builds a `Config`, calls `pipeline.run.run_capture()`, validates the result against
  `schema/floorscan.schema.json` (`cli.validate`), writes `plan.json`, then renders `plan.png` + `plan.svg`.
* `python -m floorscan validate <plan.json>` — schema check only, prints `OK` or the errors.
* `python -m floorscan render <plan.json> --out <dir>` — re-draw a plan from its JSON alone.
* `cli.canonical_hash()` hashes the plan without the `timing` block; the tests use it to prove determinism.

## 2. Settings — `floorscan/config.py`

One dataclass, `Config`, holds every tunable parameter (grid resolution, watershed persistence, wall search window,
chunk size for drift correction, the drift-handling switches used by the ablation, the scale-card size). Nothing in the
geometry code is hard-coded behind your back; the benchmark switches drift handling off by setting `local_planar=False,
door_graph=False`.

## 3. Tier detection and dispatch — `floorscan/ingest/__init__.py`, `floorscan/pipeline/run.py`

* `detect_tier(path)`: `capture.json` `"tier"` if present; else `frames/*.json` or `keyframes/depth/` → lidar; a
  `.mp4/.mov` → video; sub-folders with JPEGs → photos.
* `run_capture(path, cfg)` sends the capture to one of three paths:
  * **lidar** → `ingest/lidar.py` (`load_native` or `load_polycam`) → `run_global()`
  * **video** → `pipeline/video_tier.py: run_video()` → builds a Capture → `run_global()`
  * **photos** → `pipeline/photo_tier.py: run_photos()` (own room fitting, then the shared assembler)
* `run_global(cap, cfg, tier, ...)` is the shared back half for tiers whose frames live in one world frame
  (LiDAR, video): reconstruct rooms → detect openings → stitch → damage → reasoning → assemble.

## 4. The common data model — `floorscan/frames.py`

Every tier is reduced to the same object, so all geometry downstream is shared:

* `Frame`: camera rotation `R` (OpenCV camera → world), centre `org`, intrinsics `K`, image size, lazy RGB / depth loaders,
  photo-tier extras (room folder, doorway-shot target).
* `RaySet`: for every sampled pixel, a unit direction in the camera frame, the measured range (0 = dropout), a kind
  (`KIND_HIT`, `KIND_DROP`, `KIND_IGNORE`), the frame index, the pixel, and a 1-σ range error.
* `Capture`: frames + rays + metadata, with `world(fidx)` (rays → world points) and `apply_correction()` (rigid yaw +
  translation per frame chunk — this is how drift correction edits poses in place).

World frame everywhere: metres, **Z up**, gravity aligned, yaw arbitrary.

## 5. Ingest per tier

### LiDAR — `floorscan/ingest/lidar.py`
* `load_native`: our `floorscan-lidar-v1` layout (`capture.json`, `frames/*.json` with ARKit camera-to-world + intrinsics,
  `depth/*.png` 16-bit millimetres, `rgb/*.jpg`). `dataprep/arkitscenes.py` writes this layout from ARKitScenes.
* `load_polycam`: a Polycam "raw data" export (`keyframes/{cameras,depth,images}`), written to Polycam's public docs.
* `arkit_pose_to_cv`: ARKit (Y up, camera −Z forward) → our frame (Z up, OpenCV camera).
* `_rays_from_depth`: back-projects every 2nd depth pixel to a ray, drops ranges > 5 m, removes *flying pixels* at depth
  edges (`flying_pixel_mask`, large second difference), and gives each ray a range-dependent σ.

### Video — `floorscan/pipeline/video_tier.py`
1. `extract_keyframes`: decode the clip, keep sharp, textured frames (`_informative`: ORB features ≥ 40, Laplacian
   variance > 20), so blank walls and motion blur do not poison SfM.
2. `run_sfm`: COLMAP via `pycolmap` (sequential matching, self-calibrating focal length seeded from the lens; relaxed mapper),
   `_load_best_model` keeps the largest model. With `FLOORSCAN_CACHE=<dir>` the key-frames + models are cached so repeated
   runs replay identical poses.
3. `estimate_up`: gravity = the direction in which SfM point heights cluster most sharply (floor + ceiling).
4. `_build_capture`: per key-frame **metric monocular depth** (`ingest/photos.py: DepthAnythingV2`), aligned frame by frame
   to the SfM points (median ratio → removes the network's per-image scale error), turned into rays.
5. Scale: printed ArUco card (`detect_marker`, 100 mm) if filmed; otherwise the metric depth network (median over
   key-frames of network depth / SfM depth) combined with the ceiling-height prior (2.5 m ± 10 %) by inverse variance,
   refined by the door-height prior (2.04 m) when a door is detected (`_door_scale`, `scale_capture`).
   COLMAP matches every pair of key-frames when there are <= 300 (loop closures keep one model: 96 % of a real clip
   registered vs 60 % with sequential matching).
6. `run_global(...)`, plus warnings for partial reconstruction and the scale source used.

### Photos — `floorscan/ingest/photos.py`, `floorscan/pipeline/photo_tier.py`
* `list_photos`: one sub-folder per room; EXIF → 35 mm-equivalent focal length → intrinsics; doorway shots named
  `*_to_<room>.jpg` become links between rooms.
* `DepthAnythingV2`: Depth-Anything-V2-Metric-Indoor-Small (HuggingFace transformers, CPU) → metric depth per photo,
  multiplied by (focal / longer image side) / f0 (`net_calibration`, f0 fitted on DEV rooms) so lenses with other fields of
  view are handled.
* `process_photo` / `_estimate_up`: gravity from the depth-map normals (Manhattan structure).
* `register_yaws`: relative yaw between photos of a room from pure-rotation homographies (SIFT, `_pair_rotation`); without
  texture, a geometric fallback (`_geo_edge`: Manhattan offsets + horizontal range profiles + the clockwise-walk prior).
* `fit_room_photos`: rays of all photos of a room → walls (histogram peaks per Manhattan direction), ceiling/floor, and
  `observed: false` + a wide prior for walls no photo saw.
* Scale: card > door height (`_door_scale`) > the depth network.
* `_links_from_doorway_shots` + `_place_rooms`: rooms placed relative to each other via doorway shots, then the shared
  assembler.

## 6. Geometry (shared) — `floorscan/geometry/`

### `primitives.py`
* `mode_peak`: robust 1-D histogram mode (the workhorse for floor, ceiling and wall planes).
* `estimate_yaw`: dominant Manhattan direction from wall-band points.
* `carve`: 2-D occupancy + free space by ray carving, in the Manhattan-aligned frame.
* `segment_rooms` / `_watershed_persistence`: free space → rooms by a persistence watershed on the distance transform
  (a doorway is a saddle, a room a peak); `merge_wide_contacts` then merges regions that touch along more than 1.3 m of
  free space (wider than any door: a table or bed in the middle of a room had split real rooms in two).

### `room.py` (per-room model)
* `RoomModel`, `Wall`, `Opening` data classes; `polygon_from_walls`.
* `runs_from_mask` → `walls_from_runs`: room mask → rectilinear wall loop.
* `refine_walls` → `pick_wall_plane`: snap every wall to a plane within ±45 cm of its current position. All histogram
  peaks of the wall-band points along the wall normal are candidates (`_plane_candidates`); among those that reach the
  ceiling, cover ≥ 60 % of the wall length (`_length_coverage`) and have ≥ 25 % of the best support, the **innermost** wins,
  so the back of a bay window or an alcove no longer beats the main wall face (the fix loop). Fallback `pick_wall_peak`:
  highest peak, or a plane ≥ 18 cm further out with ≥ 50 % of its support (wall behind furniture).
* `drop_objects`: wall candidates that do not reach the ceiling are furniture → removed.
* `simplify_walls`: removes notches / spurs.
* `planar_correct`: **drift correction** — frames are cut into chunks; each chunk gets an (x, y, yaw, z) correction so its
  points land on the fitted wall / floor / ceiling planes, solved jointly over chunks with a smoothness prior.
* `levels`: floor = lowest plane with real support (beds hide most floors), ceiling = highest plane, accepted only if it
  is >= 2.0 m above the floor and covers the room (`plane_coverage`); otherwise "ceiling not observed" and a prior with a
  wide interval (`pipeline/globalfit.py`). LiDAR heights get the DEV-fitted factor `_lidar_height` of `calibration.json`.

### `openings.py`
`detect_openings`: per wall, rays that cross the wall plane and end beyond it (or drop out — glass) are rasterised into
evidence images (wall / open / dropout); open regions become doors, passages or windows, with sub-cell edge interpolation
(`_profile_cross`), a mixed-pixel footprint correction, colour checks (`_rgb_samples`: sky-like glass vs mirror) and
quality-dependent σ.

### `stitch.py`
`match_links` (door pairs between rooms), `solve_translations` (door-anchored pose graph, weighted least squares with
outlier rejection — loop closure in multi-room captures), `bfs_place` (initial placement for the photo tier),
`polygons_plan`.

### `pipeline/globalfit.py`
`reconstruct_global`: floor/ceiling levels → yaw → carve → segment → per-room `_fit_room` (refine walls, drop
furniture, simplify, levels, `planar_correct`; 3 iterations).

## 7. Damage, flags and scope

* `floorscan/damage.py`: `build_texture` renders a 2 cm/px ortho-texture of every wall, floor and ceiling from the
  frames that see it; `_fit_lighting` removes smooth illumination; `detect_regions` proposes candidates (chroma, darkness,
  speckle, depth inset); `source_crop` cuts each candidate from the most frontal source image; `verify` classifies the
  crops with the learned head and drops "none" (furniture, pictures, shadows); `analyse_room` outputs class, metric extent
  (± CI) and polygon per region.
* `floorscan/damage_model.py`: frozen DINOv2-small features + a softmax head (`floorscan/models/damage_head.npz`, trained by
  `dataprep/train_damage_head.py` on real BD3 photos) over water_stain, mold, hole, peeling, crack, none.
* `floorscan/reasoning.py`: `RULES` (R-CEIL-01 … R-FLOOR-06) turn regions into **concealed-damage flags** that name the
  rule and the triggering regions; `CATALOG` + `scope_items` produce **scope line items keyed to surface ids**.

## 8. Uncertainty — `floorscan/uncertainty.py`

Every number is `{value, ci95, sigma}` with σ = k · √(σ_fit² + floor² + (rel · value)²): σ_fit from the fit itself
(peak scatter, chunk-to-chunk spread), `floor` / `rel` per tier and measurement kind (`DEFAULTS`), k from
`floorscan/calibration.json` (currently k = 1: no held-out real calibration captures; see `bench/calibrate.py`).

## 9. Output — `floorscan/pipeline/assemble.py`, `schema/`, `floorscan/render.py`

`assemble()` builds the published contract (`schema/floorscan.schema.json`): per room polygon, walls with length CI,
ceiling height, floor area, openings, surfaces, damage, concealed flags, scope; per plan placement, adjacency, footprint,
drift report, warnings, timing. `render_plan()` draws the PNG/SVG from the JSON alone.

## 10. Real data preparation — `dataprep/`

* `fetch_sample.py`: downloads the pre-registered sample of `data/SPLIT.json` from ARKitScenes (LiDAR sessions, the device
  video and hi-res frames of each room's first session, a 1/8 strided sample of every laser scan via HTTP range requests)
  and converts it.
* `arkitscenes.py`: ARKitScenes session → our three tier layouts (`<id>_lidar`, `<id>_video`, `<id>_photos`; photos =
  8 frames from one standing spot in clockwise order, as the protocol asks).
* `register.py`: registers a LiDAR session to the venue's laser scans (raw depth only).
* `room_gt.py`, `openings_gt.py`: laser ground truth for walls, ceiling and openings (`data/gt/README.md`).
* `fit_lidar_scale.py`, `fit_depth_scale.py`: the two DEV-fitted scale corrections.
* `train_damage_head.py`: the damage classifier head on real photos (BD3).

## 11. Benchmark, calibration, head-to-head — `bench/`

* `metrics.py`: aligns a plan to the ground truth (k·90° + translation) and scores walls, ceilings, areas, openings,
  damage, footprint, adjacency, CI coverage. Never imported by the pipeline.
* `gates.py`: the case-study gates per tier, repeatability and its verdict (repeatable-but-biased vs unrepeatable).
* `run_bench.py`: runs every capture in `data/`, drift ablation (ON vs OFF), repeatability, calibration coverage, timing →
  `results/benchmark/`.
* `calibrate.py`: fits k on *held-out* real captures (`--data`), refuses to run on `data/`.
* `headtohead.py`: our LiDAR plan vs a consumer app's export (RoomPlan JSON or a simple JSON) → one table; writes
  NOT RUN without an export.

## 12. Fix loop — `fixloop/`

`before_src/` (frozen source when the fix was declared), `after_src/` (= before + `fix.diff` only), `diagnose.py`
(root-cause evidence on the real captures), `FIX_DECLARATION.md`, `run_fixloop.py` (runs both trees on the same captures
via `FLOORSCAN_SRC`) → `results/fixloop/`.

## 13. Report, scripts, tests

* `report/build_report.py` → `results/TECHNICAL_REPORT.pdf` (≤ 6 pages, asserted) + `.md`.
* `scripts/quickstart.*`, `scripts/reproduce.*`, `scripts/fetch_models.py` (torch + transformers + model weights),
  `scripts/make_scale_card.py`.
* `tests/`: end-to-end on the real captures (accuracy regression bounds, determinism, CI on every measurement),
  photo tier, CLI, tier detection, Polycam adapter, head-to-head harness, geometry and reasoning unit tests.
