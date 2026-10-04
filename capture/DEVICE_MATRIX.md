# Device matrix – which tier runs on which hardware, and what accuracy it honestly delivers

Measured on 9 real ARKitScenes rooms (iPad Pro: same LiDAR + ARKit stack as Pro iPhones) against FARO laser scans; numbers are
the 5 held-out TEST rooms (DEV in brackets). Video and photo inputs come from the iPad walkthroughs **without a scale card**,
which is what limits those tiers here (`docs/WHY_THESE_RESULTS.md`).

| Tier | Hardware | What the user captures | Scale source | Measured on real rooms | Where it fails |
|---|---|---|---|---|---|
| **LiDAR** | iPhone 12 Pro ... 17 Pro (Pro / Pro Max), iPad Pro | Polycam raw export (depth, poses, intrinsics, RGB) or native layout | metric (time of flight) | ceilings within 1.5 cm: 57 % (70 %); walls within 2 cm: 10 % (41 %); simple rooms within 4-7 cm on every wall | cluttered / irregular rooms (mis-segmentation, hidden walls); sessions that never look at the ceiling |
| **Video** | any iPhone 15+ (0.5x lens, H.264) | one walkthrough clip, scale card filmed first | card (1-2 %) > door height > depth network + ceiling prior | without a card: 0 % of walls within ±3 %, median worst wall error 1.4 m | scale without the card (depth network ±15 %); COLMAP track loss |
| **Photos** | any iPhone 15+ (0.5x lens, JPEG) | 2-8 stills per room from one spot, doorway shots `*_to_<room>` | card > door height > depth network | without a card: 30 % of walls within ±8 %; runs on real iPhone 15 Pro photos (`results/iphone15pro_photos/`) | scale without the card; walls seen in no photo (reported `observed: false`) |

Tier choice: Pro iPhone → LiDAR; otherwise video if the property is open and well lit; otherwise photos. Always film / photograph
the scale card. Every measurement carries a 95 % interval that widens with the tier, the scale source and what was observed.
