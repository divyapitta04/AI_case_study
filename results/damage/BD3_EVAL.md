# Damage classifier on real photos (BD3 test split, never used for training)

Images: 793 real photos of building surfaces (BD3, CC-BY-4.0). Features: DINOv2-small (frozen); head: logistic regression trained on BD3 train (3172 photos) + 0 non-damage candidates from DEV rooms.

* **class accuracy: 94.7 %** (6 classes: water_stain, mold, hole, peeling, crack, none)
* damage vs no damage: precision 100.0 %, recall 99.9 %

| true \ predicted | water_stain | mold | hole | peeling | crack | none | recall |
|---|---|---|---|---|---|---|---|
| water_stain | 105 | 2 | 2 | 2 | 1 | 0 | 94 % |
| mold | 1 | 121 | 0 | 1 | 0 | 0 | 98 % |
| hole | 2 | 0 | 90 | 4 | 4 | 0 | 90 % |
| peeling | 5 | 3 | 6 | 89 | 2 | 0 | 85 % |
| crack | 1 | 0 | 2 | 3 | 223 | 1 | 97 % |
| none | 0 | 0 | 0 | 0 | 0 | 123 | 100 % |
| precision | 92 % | 96 % | 90 % | 90 % | 97 % | 99 % | |

This measures recognition of the damage *class* on real close-up photos. Metric extent on a real wall needs staged damage with measured ground truth, which no public dataset provides (docs/REAL_CAPTURE_TODO.md).
