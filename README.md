# floorscan — floor plan, damage and scope from iPhone photos, video or LiDAR

One command per capture, three input tiers (photos, video, LiDAR), one output contract: `plan.json` to a published schema plus a
rendered plan, with a 95 % interval on every measurement.

```
python -m floorscan run <your capture folder> --out <output folder>
```

**All generated outputs are in [`results/`](results/README.md)**: the technical report, the benchmark report, the fix loop,
the damage-classifier evaluation and the real iPhone 15 Pro example.

## Use it on your own iPhone 15 Pro photos (main scenario)

1. Follow `capture/PROTOCOL.md`: stand in the middle of each room, turn clockwise, take 6-8 photos (0.5x lens), print and
   photograph the scale card if you can.
2. Put each room's photos in its own folder: `myhouse/living/IMG_*.jpg`, `myhouse/kitchen/...` (JPEG, EXIF kept - just copy
   the files off the phone).
3. `python -m floorscan run myhouse --out result` → `result/plan.json`, `plan.png`, `plan.svg`.

This exact path is tested on **real iPhone 15 Pro photos** (`data/iphone15pro_photos/`, 4 photos of one room, CC BY-SA 4.0;
`tests/test_iphone15pro.py`; output in `results/iphone15pro_photos/`). Video (`.mp4/.mov`) and LiDAR (Polycam raw export)
folders work the same way; the tier is detected automatically.

## How to run

```powershell
.\scripts\quickstart.ps1                       # Windows: venv, dependencies, model weights, then the iPhone 15 Pro photos + a real LiDAR capture
.\scripts\quickstart.ps1 -Capture D:\myhouse   # your own capture
```
```bash
./scripts/quickstart.sh [capture_dir] [out]    # macOS / Linux
```
By hand: `python -m venv .venv`, `.venv/Scripts/pip install -r requirements.txt`, `python scripts/fetch_models.py`, then
`python -m floorscan run ...`. Python 3.10+, CPU only, no network after setup. Tests: `python -m pytest -q tests` (~3 min).
Regenerate every reported number: `scripts/reproduce.ps1` (or `.sh`).

## Why the evaluation uses a public dataset — and what it shows

No iPhone, laser measurer or scanning-app licence was available, so accuracy is measured on **Apple's ARKitScenes**: 9 real
rooms captured with an iPad Pro (same LiDAR / ARKit stack as Pro iPhones), each scored against a **FARO laser scan** of the
room. The 9 rooms were drawn at random before any was run: **4 DEV rooms** were used for every fix and fit, **5 TEST rooms**
only for the final numbers. Full reasoning: [`docs/WHY_THESE_RESULTS.md`](docs/WHY_THESE_RESULTS.md).

Held-out TEST rooms (share of measurements inside the case-study gate; `results/benchmark/BENCHMARK_REPORT.md`):

| tier | captures | walls within gate | ceilings within 1.5 cm / gate | openings within 2 cm | median worst wall error |
|---|---|---|---|---|---|
| LiDAR | 15 | 10 % | **57 %** | 0 % | 29 cm |
| video (no scale card in dataset) | 3 | 0 % | 0 % | 0 % | 1.4 m |
| photos (frames, no scale card) | 5 | 30 % | 60 % | 0 % | 1.2 m |

DEV rooms (in-sample): LiDAR walls 41 % in gate, ceilings 70 %, median worst wall error 36 mm.

What this means, honestly:
* **LiDAR** measures ceilings and simple rooms well (several TEST captures within 4–7 cm on every wall, ceilings within
  2–20 mm), but cluttered or irregular rooms still produce metre-level failures (rooms mis-segmented, walls never seen), so the
  2 cm wall gate is met for only a minority of walls.
* **Video and photos** fail mainly on **scale**: this dataset has no scale card, so scale comes from a depth network that is
  ±15 % between rooms. The protocol's printed card (1–2 %) is the fix and could not be tested.
* **Openings** are detected but not to 2 cm; many reported openings are phantoms (furniture, recesses).
* **Damage**: the learned classifier is **94.7 %** accurate on 793 unseen real photos of damaged walls (BD3); the benchmark
  rooms have no damage, so detections there are false positives.
* **Fix loop** (DEV room): worst LiDAR wall error **217 → 38 mm**, walls in gate 0/8 → 4/8 (`fixloop/FIX_DECLARATION.md`).
* **Not done**: own iPhone captures with measured ground truth, multi-room real capture, staged damage, consumer-app
  head-to-head (NOT RUN, nothing invented), git history. See `COMPLIANCE_MATRIX.md`, `docs/REAL_CAPTURE_TODO.md`.

## What comes out (`plan.json`)
Per room: polygon, walls (length ± 95 % CI, `observed` flag), ceiling height, floor area, openings (type, width, height, sill,
offset), damage regions (water_stain / mold / hole / peeling / crack, metric extent, polygon), concealed-damage flags (rule
id + trigger), scope line items keyed to surfaces. Per plan: placement, adjacency, footprint, drift report, warnings (scale
source, unobserved walls / ceiling, partial reconstruction). Walkthrough with a real example: `docs/FUNCTIONALITY_WALKTHROUGH.md`.

## Repository map
| path | what |
|---|---|
| `floorscan/` | the pipeline (`ingest/`, `pipeline/`, `geometry/`, `damage.py`, `damage_model.py`, `reasoning.py`, `uncertainty.py`, `render.py`, `cli.py`, `calibration.json`, `models/`) |
| `schema/` | published output contract |
| `capture/` | protocol, printable scale card, device matrix |
| `data/` | benchmark captures (ARKitScenes, converted), `iphone15pro_photos/` (real iPhone example), `gt/` (laser ground truth), `SPLIT.json` |
| `dataprep/` | fetch + convert ARKitScenes, laser ground truth, DEV-only fits, damage-head training |
| `bench/` | benchmark runner, metrics, gates, calibration, head-to-head harness; `cache/` = COLMAP replay cache |
| `fixloop/` | fix declaration + post-mortem, diagnosis, before/after source snapshots, diff |
| `report/` | technical-report builder |
| `results/` | **all generated outputs** |
| `docs/` | why the results are what they are, code and functionality walkthroughs, disclosures, known failures, to-do |
| `scripts/`, `tests/` | quickstart / reproduce / model fetch / scale card; 15 tests on real data |

## Documentation
`docs/WHY_THESE_RESULTS.md` · `docs/FUNCTIONALITY_WALKTHROUGH.md` · `docs/CODE_WALKTHROUGH.md` · `docs/DISCLOSURES.md` ·
`docs/KNOWN_FAILURES.md` · `docs/REAL_CAPTURE_TODO.md` · `data/gt/README.md` · `capture/PROTOCOL.md` · `capture/DEVICE_MATRIX.md` ·
`COMPLIANCE_MATRIX.md` · `results/TECHNICAL_REPORT.pdf`
