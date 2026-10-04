#!/usr/bin/env bash
# Regenerate every reported number from the raw real captures; everything is written to results/.
#   ./scripts/reproduce.sh [--fetch-data]
set -euo pipefail
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install -q -r requirements.txt
$PY scripts/fetch_models.py
if [ "${1:-}" = "--fetch-data" ]; then
  $PY dataprep/fetch_sample.py
  for v in $($PY -c "import json; s=json.load(open('data/SPLIT.json')); print(' '.join(list(s['dev']) + list(s['test'])))"); do
    $PY dataprep/room_gt.py --visit "$v"
    $PY dataprep/openings_gt.py --visit "$v"
  done
  $PY dataprep/fit_lidar_scale.py
  $PY dataprep/fit_depth_scale.py
  $PY dataprep/train_damage_head.py
fi
FLOORSCAN_NO_CAL=1 $PY bench/run_bench.py --split dev --force --no-ablation
$PY bench/calibrate.py --from-results results/benchmark/metrics.json
$PY bench/run_bench.py --split test --force
$PY bench/run_bench.py
$PY bench/headtohead.py
$PY fixloop/run_fixloop.py
$PY report/build_report.py
$PY -m pytest -q tests
