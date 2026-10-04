# data/ — every input the reported results use

Each folder below is read by `bench/run_bench.py` (or a test) and has a result in `../results/`. Nothing here is unused.

## Benchmark captures (Apple ARKitScenes, real iPad Pro sessions) — 9 rooms

Folder name = `<ARKitScenes session id>_<tier>`. Every room has 2-3 LiDAR sessions (repeatability) and, from its first
session, one video and one photo set (8 frames). `SPLIT.json` says which rooms are DEV (used for fixes / fits) and which are
TEST (held out).

| room (visit) | split | LiDAR sessions | video | photos |
|---|---|---|---|---|
| 421337 | DEV | 42444946, 42444949, 42444950 | 42444946 | 42444946 |
| 467293 | DEV | 45261615, 45261619, 45261620 | 45261615 | 45261615 |
| 434650 | DEV | 42898811, 42898816, 42898817 | 42898811 | 42898811 |
| 421380 | DEV | 42445022, 42445026 (42445021 excluded: different space) | 42445022 | 42445022 |
| 483311 | TEST | 48018382, 48018386, 48018387 | 48018382 | 48018382 |
| 469249 | TEST | 47332915, 47332916, 47332918 | 47332915 | 47332915 |
| 469782 | TEST | 47334107, 47334109, 47334110 | 47334107 | 47334107 |
| 422851 | TEST | 42897501, 42897504, 42897508 | not processed (CPU time) | 42897501 |
| 471952 | TEST | 47204552, 47204554, 47204556 | not processed (CPU time) | 47204552 |

Layouts: `*_lidar` = `floorscan-lidar-v1` (depth PNG + poses + RGB), `*_video` = one mp4, `*_photos` = `room/IMG_*.jpg` with EXIF.
Rebuilt from the raw dataset by `python dataprep/fetch_sample.py`.

## Real iPhone 15 Pro example
`iphone15pro_photos/` — 4 real iPhone 15 Pro photos of one room (Wikimedia Commons, CC BY-SA 4.0, `ATTRIBUTION.md`); used by
`tests/test_iphone15pro.py` and the quickstart. No measured dimensions exist for this room.

## Ground truth
`gt/` — laser-derived dimensions and openings per room (`gt/README.md`).
