# Third-party models, datasets and APIs (disclosure)

Nothing calls the author's infrastructure or any web API at run time; everything runs locally on CPU. The only downloads
are the datasets (once, `dataprep/fetch_sample.py`, `dataprep/train_damage_head.py`) and model weights (once,
`scripts/fetch_models.py`).

| Component | Used for | Licence | Fetched by |
|---|---|---|---|
| **Apple ARKitScenes** - 9 rooms (visits) of the Validation fold: 26 iPad Pro LiDAR sessions, 9 device videos, FARO laser scans | the benchmark: real LiDAR depth, RGB, ARKit poses, video; laser scans as ground truth | **non-commercial research licence** (github.com/apple/ARKitScenes) - fine for this evaluation, not for a commercial product | `python dataprep/fetch_sample.py` (~25 GB raw; converted captures + ground truth in `data/`) |
| **BD3 - Building Defect Dataset** (Kottari & Arjunan 2024; HuggingFace mirror `chandrabhuma/building_defect_vqa`) - 3 965 real photos of damaged / plain building surfaces | training (train split) and evaluation (test split) of the damage classifier head | CC-BY-4.0 | `python dataprep/train_damage_head.py` |
| **DINOv2-small** (`facebook/dinov2-small`) | frozen image features for the damage classifier | Apache-2.0 | `scripts/fetch_models.py` |
| **Depth-Anything-V2-Metric-Indoor-Small** | metric monocular depth for the photo and video tiers | Apache-2.0 (Small model) | `scripts/fetch_models.py` |
| **COLMAP** via `pycolmap` | structure-from-motion for the video tier | BSD-3-Clause | `pip` |
| PyTorch, transformers, OpenCV, NumPy, SciPy, Shapely, Matplotlib, jsonschema, ReportLab, pypdf, pyarrow | runtime / tooling | BSD / MIT / Apache | `pip` |

## What was fitted, and on what (DEV rooms only)

The sample (`data/SPLIT.json`) was drawn at random (fixed seed) from the 115 Validation-fold visits with laser scans and
>= 3 sessions **before** any of them was run: 4 DEV rooms (421337, 467293, 434650, 421380) may be used for design and
fitting; 5 TEST rooms (483311, 469249, 469782, 422851, 471952) were used for nothing until the final evaluation.

Fitted on DEV only:
* the recess-aware wall-plane rule (fix loop, room 421337) and the room-merge / floor / ceiling robustness rules found on
  the other DEV rooms;
* the LiDAR floor-to-ceiling factor 1.0107 (`dataprep/fit_lidar_scale.py`, 9 DEV sessions: the iPad LiDAR reads heights
  ~1 % short; a range rescale was tried and rejected because it made DEV walls worse);
* the depth-network focal correction f0 = 0.817, spread 15 % (`dataprep/fit_depth_scale.py`, 6 DEV photo/video captures);
* the interval calibration factors k (`bench/calibrate.py --from-results`, DEV entries of the benchmark run);
Not trained or fine-tuned: any network. The video tier's ceiling-prior width (10 %) was set after seeing room 421337.

**TEST numbers are the honest ones**; DEV numbers are in-sample.

## Ground truth
Laser-derived, not tape-measured (`data/gt/README.md`): plane fits and edge snapping are automatic; the wall convention and
the rough opening marks are human judgement on the laser figures.

## Reference material
Apple RoomPlan `CapturedRoom`; ARKit `sceneDepth` (256x192, ~5 m range); Polycam raw-data export layout; ARKitScenes data
format (`.traj`, `.pincam`, laser scans + `_pose.txt`). Consumer-app accuracy claims were read but no incumbent result is
reproduced or invented.

A repository found while researching (`keerthika61/room-scan-floorplan`) was not used, copied from, or consulted for code.
