# What is still missing (and how to add it)

The benchmark today is **9 real rooms** from Apple's ARKitScenes dataset (iPad Pro LiDAR sessions, device videos, frames as
photo sets) scored against FARO laser scans, 5 of them held out (`results/benchmark/`), plus a run on real iPhone 15 Pro photos
without ground truth. That is real data, but not the case study's full benchmark. Still *not* done:

1. **More real rooms, captured with our protocol on an iPhone 15+** (case study Part 2 composition): a multi-room capture (>= 3 rooms +
   connector), a furnished room with two staged damage classes, the same rooms at all three tiers (photos as per-room folders, scale card
   filmed), one room captured twice per tier, laser / tape ground truth on everything (including opening widths). Add each capture as
   `data/<id>_<tier>/` and its ground truth as `data/gt/<file>.json` + an entry in `data/gt/index.json`, then
   `python bench/run_bench.py`. ARKitScenes has more rooms with laser scans: add visits to `data/SPLIT.json` and run
   `dataprep/fetch_sample.py`, `dataprep/room_gt.py`, `dataprep/openings_gt.py`.
2. **More held-out rooms**: 5 TEST rooms are too few to tune the room-type robustness further without overfitting.
3. **Validate the stock-app adapter** (`floorscan/ingest/lidar.py: load_polycam`) on a real Polycam raw-data export - it was written
   from Polycam's public documentation; the test only repackages the ARKitScenes capture in that layout.
4. **Head-to-head** (Part 3): export the same two LiDAR-tier rooms from Polycam or magicplan (name + version), then
   `python bench/headtohead.py --capture <name> --app "<app version>" --export <file> --gt <gt json> --room <room>`.
5. **Scale for video / photos**: the ARKitScenes video has no scale card, so both tiers fall back to priors and the depth network (±15 %).
   Re-test with the card filmed first (`capture/PROTOCOL.md`).
6. **Damage ground truth**: the benchmark room has no damage; detections there are all false positives. Stage water stains / mould /
   holes and record their extents.
7. **Print and check the scale card** (`capture/scale_card_*.pdf`): the printed bar must measure 10.0 cm.
8. **Commit history** (case-study Part 5): this working tree has no git history. Create the repository and commit as you work from here.
9. **Walk-in dry run**: capture a space you have not seen with a friend's iPhone, following `capture/PROTOCOL.md` literally, and time
   install-to-result.
