"""Format test of the head-to-head harness. The 'app export' below is SYNTHETIC (ground truth + noise) and exists only to
exercise the code path on our real LiDAR plan; the real comparison needs a real Polycam / magicplan / RoomPlan export."""
import json

import numpy as np


def test_roomplan_adapter_and_table(lidar_plan, gt, tmp_path):
    from bench import headtohead as H
    out = lidar_plan[0]
    g = gt["rooms"][0]
    # fabricate a RoomPlan-like CapturedRoom JSON with 4 walls, 1 door, 2 windows (transform = column-major 4x4)
    def wall(length, cx, cz, ang):
        c, s = np.cos(ang), np.sin(ang)
        T = [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, cx, 1.2, cz, 1]
        return dict(dimensions=[length, g["ceiling_height"] + 0.02, 0.1], transform=T)
    room = dict(walls=[wall(g["w"] + 0.03, g["w"] / 2, 0, 0), wall(g["d"] - 0.02, g["w"], g["d"] / 2, np.pi / 2),
                       wall(g["w"] + 0.03, g["w"] / 2, g["d"], 0), wall(g["d"] - 0.02, 0, g["d"] / 2, np.pi / 2)],
                doors=[dict(dimensions=[0.9, 2.0, 0.1], transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])],
                windows=[dict(dimensions=[1.28, 1.2, 0.1], transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
                         dict(dimensions=[0.88, 1.2, 0.1], transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])],
                openings=[])
    p = tmp_path / "roomplan.json"
    json.dump(room, open(p, "w"))
    app = H.load_roomplan(str(p))
    assert len(app["rooms"][0]["wall_lengths"]) == 4 and app["rooms"][0]["floor_area"] > 10
    rows = H.compare(out, gt, app, "room")
    assert any(r[0].startswith("ceiling") for r in rows)
    H.write_table(rows, "SYNTHETIC format test", "cap", str(tmp_path / "h2h.md"), ran=True)
    txt = open(tmp_path / "h2h.md", encoding="utf-8").read()
    assert "Beat or tie on" in txt


def test_not_run_when_no_export(tmp_path):
    from bench import headtohead as H
    H.write_table([], "app", "cap", str(tmp_path / "h.md"), ran=False)
    assert "NOT RUN" in open(tmp_path / "h.md", encoding="utf-8").read()
