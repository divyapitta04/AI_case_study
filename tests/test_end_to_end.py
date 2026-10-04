"""End-to-end tests on the REAL ARKitScenes captures (about 2-3 minutes on CPU).

Accuracy thresholds are regression bounds around today's real results (see results/benchmark/BENCHMARK_REPORT.md),
not the case-study gates: they catch a change that makes the real numbers worse.
"""
import os


def test_lidar_end_to_end(real_lidar, lidar_plan, gt, tmp_path):
    from bench.metrics import evaluate
    from floorscan.cli import canonical_hash, run, validate
    out, out_dir = lidar_plan
    assert out["capture"]["tier"] == "lidar"
    assert validate(out) == []
    for f in ("plan.json", "plan.png", "plan.svg"):
        assert os.path.getsize(os.path.join(out_dir, f)) > 1000
    assert len(out["rooms"]) == 1
    m = evaluate(out, gt)
    assert max(abs(e) for e in m["rooms"][0]["wall_err"].values()) < 0.06   # today: 37 mm vs the laser
    assert abs(m["rooms"][0]["height_err"]) < 0.06                         # today: -41 mm
    assert abs(m["footprint_err_pct"]) < 2.0                                 # today: -0.2 %
    # determinism: same capture in -> same plan out (timing excluded)
    out2 = run(real_lidar, str(tmp_path / "again"), quiet=True)
    assert canonical_hash(out) == canonical_hash(out2)


def test_every_measurement_has_ci(lidar_plan):
    out, _ = lidar_plan

    def walk(x, path=""):
        if isinstance(x, dict):
            if "value" in x and "ci95" in x:
                lo, hi = x["ci95"]
                assert lo <= x["value"] <= hi, path
                return 1
            return sum(walk(v, path + "/" + k) for k, v in x.items())
        if isinstance(x, list):
            return sum(walk(v, path + "/#") for v in x)
        return 0

    assert walk(out["rooms"]) > 20


def test_photo_tier_runs_and_reports_uncertainty(real_photos, lidar_plan, tmp_path):
    from floorscan.cli import run, validate
    out = run(real_photos, str(tmp_path / "p"), quiet=True)
    assert out["capture"]["tier"] == "photos"
    assert validate(out) == []
    # photo intervals must be visibly wider than LiDAR intervals on the same room (honest widening)
    width = lambda o: o["rooms"][0]["walls"][0]["length"]["ci95"][1] - o["rooms"][0]["walls"][0]["length"]["ci95"][0]  # noqa: E731
    assert width(out) > 5 * width(lidar_plan[0])
