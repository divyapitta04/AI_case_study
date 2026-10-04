# Why the results look the way they do

## The constraint
The case study asks for captures made with an **iPhone 15+** following our protocol, a laser / tape ground truth, staged
damage and an export from a consumer scanning app. None of that was available: **no iPhone, no laser measurer, no app
licence**. Making numbers up, or evaluating on a simulator, would not say anything about real rooms. So the evaluation uses
the closest real, public substitute:

| case study wants | what this repository uses instead | effect on the numbers |
|---|---|---|
| iPhone 15+ / Pro captures with our protocol | **Apple ARKitScenes**: real rooms captured with an **iPad Pro** (same LiDAR sensor family and ARKit stack as Pro iPhones), 2-3 sessions per room | LiDAR tier is representative; capture style is ARKitScenes', not ours (some sessions never look at the ceiling, some are held low) |
| laser / tape ground truth | the **FARO laser scans** ARKitScenes provides for the same rooms; walls, ceiling and openings fitted from them (`data/gt/README.md`) | mm-level reference; the wall convention and the rough opening marks involve human judgement; repeated fits differ by <= 7 mm (walls) and 1.5 cm (ceiling) |
| video filmed with a **scale card** first | the device video of the session - **no scale card** | video scale comes from the depth network + ceiling / door priors: tens of % worse than with a card (1-2 %) |
| 2-8 photos per room from a standing spot | **8 frames** picked from the walkthrough at one standing spot (clockwise) - 1x lens, **no scale card** | photo scale from the depth network only; walls seen at grazing angles |
| furnished room with staged damage | no damage in the benchmark rooms; the damage classifier is measured on **BD3**, a public set of real photos of damaged walls | damage *class* accuracy is real; metric extent of real damage cannot be scored |
| multi-room property | ARKitScenes sessions cover one room each | multi-room stitching is not exercised on real data |
| consumer-app export (head-to-head) | none exists for these rooms | head-to-head **NOT RUN** (harness built, numbers never invented) |
| git history while building | none | NOT DONE |

## How the numbers were kept honest
* **Pre-registered random sample** (`data/SPLIT.json`): 9 rooms drawn with a fixed seed from the 115 qualifying ARKitScenes
  rooms before any of them was run. **4 DEV rooms** were used for every fix and fit; **5 TEST rooms** were used only for the
  final numbers. Report the TEST rows; DEV rows are in-sample.
* **Exclusions only by rules fixed in advance** (laser does not cover the room; session shows a different space) - one room
  and one session, both documented.
* **Ground truth independent of the pipeline** (`dataprep/room_gt.py`, `openings_gt.py`).
* Everything that was fitted is listed with the data it was fitted on (`docs/DISCLOSURES.md`).

## What the real data showed, and what was done about it
1. **Wall planes snapped into recesses** (bay window, alcove): 224 mm worst error -> recess-aware plane choice (fix loop,
   `fixloop/FIX_DECLARATION.md`).
2. **Furniture split one room into two or three** (a table or bed makes a false "doorway") -> regions touching along more
   than 1.3 m of free space are merged (DEV room 467293: worst wall error 2.2 m -> 3-4 cm).
3. **Beds and wardrobe tops taken as floor / ceiling** -> floor = lowest real plane, ceiling must be >= 2 m up and cover the
   room; otherwise "ceiling not observed" with a prior and a wide interval.
4. **iPad LiDAR heights read ~1 % short** (8 of 9 DEV sessions, median 1.07 %) -> a DEV-fitted height factor (1.0107). Rescaling all ranges
   was tried and rejected: it made DEV walls worse.
5. **Depth network scale**: about 15 % spread between captures -> a DEV-fitted focal correction (+2.5 % here) and an honest
   interval; the real fix is the scale card in the protocol.
6. **COLMAP lost track in real walkthroughs** (60 % of key-frames) -> exhaustive matching for clips up to 300 key-frames (96 %).
7. **Damage detector fired on furniture and pictures** -> a learned verifier (DINOv2 features + head trained on real photos,
   94.7 % class accuracy on unseen real photos).

## Reading the final table
Held-out TEST rooms (`results/benchmark/BENCHMARK_REPORT.md` section 0):

| tier | walls in gate | ceilings in gate | openings within 2 cm | median worst wall error |
|---|---|---|---|---|
| LiDAR (15 sessions) | 10 % | 57 % | 0 % | 29 cm |
| video (3 clips) | 0 % | 0 % | 0 % | 1.4 m |
| photos (5 sets) | 30 % | 60 % | 0 % | 1.2 m |

* **LiDAR ceilings** pass in more than half of the unseen sessions (2-20 mm errors); misses are sessions that never looked at
  the ceiling (reported as "not observed" with a prior) or cluttered rooms.
* **LiDAR walls**: simple rooms come out within 4-7 cm on every wall, which is close to but outside the 2 cm gate; cluttered or
  irregular rooms (kitchen with partitions, bathroom, a narrow furnished room) fail by tens of cm to metres because the room
  is mis-segmented or walls are hidden. The DEV rooms - where those failure modes were fixed - reach 41 % / 70 %, so the
  remaining gap is robustness to room types not seen during development, not sensor precision.
* **Video / photos** miss mainly on **scale** (no scale card in this dataset; the depth network varies by ~15 % between
  rooms). Their geometry is otherwise consistent; with the protocol's card the scale error is 1-2 %.
* **Openings**: detected, but widths are not within 2 cm and phantoms (furniture, recesses) count as misses.
* **Repeatability** (two sessions of the same room) fails at the 1 cm level for the same reasons as the walls.
* **Damage**: 94.7 % class accuracy on 793 unseen real photos (BD3); no damage exists in the benchmark rooms.
* **Real iPhone 15 Pro photos** (`results/iphone15pro_photos/`): the photo tier runs end to end on genuine iPhone input and
  reports wide intervals and unobserved walls instead of false precision; accuracy cannot be scored (no measured dimensions).

Two TEST videos were not processed (COLMAP takes ~25 min per clip on CPU); they are listed as not run, not as passes.

## What would close the gap
Capture 3-5 rooms with an iPhone 15 Pro following `capture/PROTOCOL.md` (scale card filmed first, ceiling and every wall
looked at), tape-measure walls and openings, stage two damage types, export the same rooms from Polycam, and run
`python bench/run_bench.py` on them (`docs/REAL_CAPTURE_TODO.md`).
