# Compliance matrix

Status: **DONE** = implemented and exercised on real data · **PARTIAL** = exercised, requirement only partly met or measured ·
**BUILT, NOT MEASURED** = runs, but no real data tests it · **NOT DONE**.
Benchmark = 9 real ARKitScenes rooms (iPad Pro), FARO laser ground truth, 4 DEV / 5 held-out TEST rooms (`data/SPLIT.json`);
numbers below are TEST unless marked. Why a public dataset: `docs/WHY_THESE_RESULTS.md`. All outputs: `results/`.

## Part 1 – Capture route and tiers
| Requirement | Where | Status |
|---|---|---|
| Capture route; Route 2 = stock protocol a non-engineer follows | `capture/PROTOCOL.md`, `capture/scale_card_*.pdf` | **PARTIAL** — written; not field-tested with a non-engineer |
| Photos tier: 2-8 stills per room in folders, any iPhone 15+, same stitched plan | `floorscan/ingest/photos.py`, `floorscan/pipeline/photo_tier.py` | **PARTIAL** — runs on real iPhone 15 Pro photos (`tests/test_iphone15pro.py`); on TEST rooms 30 % of walls in the ±8 % gate (no scale card in dataset); multi-room stitch not measured |
| Video tier: walkthrough clip, iPhone 15+ | `floorscan/pipeline/video_tier.py` | **PARTIAL** — runs on real clips; fails ±3 % on scale (no card); 2 of 5 TEST clips not processed (CPU time) |
| LiDAR tier on Pro devices | `floorscan/ingest/lidar.py` (native + Polycam adapter) | **PARTIAL** — real iPad Pro LiDAR: ceilings 57 % within 1.5 cm, walls 10 % within 2 cm (41 % on DEV); Polycam adapter tested only by repackaging |
| Same output contract from each tier, intervals widen as data thins | `schema/`, `floorscan/uncertainty.py` | **DONE** |
| Device matrix with honest accuracy | `capture/DEVICE_MATRIX.md` | **DONE** (measured numbers) |

## Part 2 – Output contract and gates
| Requirement | Where | Status |
|---|---|---|
| Dimensioned per-room plan | `floorscan/geometry/`, `floorscan/pipeline/assemble.py` | **DONE** |
| Stitched multi-room plan with adjacency | `floorscan/geometry/stitch.py` | **BUILT, NOT MEASURED** (benchmark rooms are single) |
| Damage regions, class + metric extent | `floorscan/damage.py`, `floorscan/damage_model.py` | **PARTIAL** — class accuracy 94.7 % on 793 unseen real photos (`results/damage/BD3_EVAL.md`); metric extent not scorable (no staged damage) |
| Concealed-damage flags with the rule that fired | `floorscan/reasoning.py` | **DONE** |
| Scope line items keyed to surfaces | `floorscan/reasoning.py` | **DONE** (own catalogue) |
| CI on every measurement | `floorscan/uncertainty.py`, `tests/test_end_to_end.py` | **DONE** |
| One command per capture, schema JSON, rendered plan | `floorscan/cli.py` | **DONE** |
| Benchmark composition (multi-room, staged damage, 3 tiers, repeats, laser GT, raw data) | `dataprep/`, `data/` | **PARTIAL** — 9 real rooms, 3 tiers, 2-3 repeat sessions, laser GT; no multi-room, no staged damage, not iPhone/protocol captures |
| Gate: opening widths <= 2 cm on >= 85 % incl. missed/phantom | `bench/gates.py`, `data/gt/README.md` | **DONE (FAILS)** — 0 % TEST, 3 % DEV |
| Gate: ceiling <= 1.5 cm; repeat spread <= 1 cm; biased vs unrepeatable | `bench/gates.py` | **DONE (PARTLY FAILS)** — 57 % TEST ceilings in gate; spread <= 1 cm in 6/15 pairs |
| Gate: repeatability (1 cm or 0.5 % per wall) | `bench/gates.py` | **DONE (FAILS)** — 0/15 TEST pairs |
| Gate: drift accountability, ON/OFF ablation | `floorscan/geometry/room.py`, `bench/run_bench.py` | **DONE** — ablation on every TEST room (`results/benchmark/drift_ablation_*.png`) |
| Gate: photo-tier whole-property stitch | `floorscan/pipeline/photo_tier.py` | **PARTIAL** — single rooms only; footprint ±8 % in 1/5 TEST sets |
| Calibration scored at every tier | `bench/calibrate.py`, `floorscan/calibration.json` | **DONE** — fitted on DEV (k capped at 3), coverage reported on TEST |

## Part 3 – Head-to-head vs a consumer app
| Requirement | Where | Status |
|---|---|---|
| LiDAR output vs one app's export on 2 rooms, beat/tie >= 70 % | `bench/headtohead.py`, `tests/test_headtohead.py` | **NOT DONE** — harness built and tested; no app export exists; table marked NOT RUN |

## Part 4 – Fix loop
| Requirement | Where | Status |
|---|---|---|
| Declaration: worst gate, number, root cause + evidence, fix, prediction | `fixloop/FIX_DECLARATION.md`, `fixloop/diagnose.py` | **DONE** (real data, laser-registered evidence) |
| Shipped fix, before/after runs, diff, post-mortem | `fixloop/`, `results/fixloop/` | **DONE** — 217 → 38 mm worst wall error, 0/8 → 4/8 walls (predicted 70 mm); gate still fails |

## Part 5 – Process evidence
| Commit history | — | **NOT DONE** |

## Deliverables
| # | Deliverable | Path | Status |
|---|---|---|---|
| 1 | Compliance matrix | `COMPLIANCE_MATRIX.md` | **DONE** |
| 2 | Capture route + device matrix | `capture/` | **DONE** |
| 3 | Repo, README, < 15 min to a result, one command per capture | `README.md`, `scripts/quickstart.*`, `docs/` | **DONE** |
| 4 | Reproduction bundle | `scripts/reproduce.*`, `dataprep/`, `data/`, `bench/cache/` | **DONE** |
| 5 | Benchmark report (gates, repeatability, head-to-head, timing) | `results/benchmark/` | **DONE**; head-to-head NOT RUN |
| 6 | Fix loop bundle | `fixloop/`, `results/fixloop/` | **DONE** |
| 7 | Technical report <= 6 pages | `results/TECHNICAL_REPORT.pdf` | **DONE** |
| 8 | Raw data: sensor logs, ground truth, app exports | `data/`, `data/gt/` | **PARTIAL** — real sensor data + laser GT; no app exports |

## Constraints
| Requirement | Status |
|---|---|
| Runs cold on an unseen capture, all tiers | **DONE** (auto-detected; real iPhone 15 Pro photos tested) |
| Models / datasets disclosed; no own infrastructure | **DONE** (`docs/DISCLOSURES.md`) |
| Weights / large data fetched by script | **DONE** (`scripts/fetch_models.py`, `dataprep/fetch_sample.py`) |
| Mirrors, glass, wet floors, low light | **PARTIAL** — glass / bay windows / a mirror in the real rooms; wet floors and low light not in the sample (`docs/KNOWN_FAILURES.md`) |
