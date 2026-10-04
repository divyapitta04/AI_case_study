#!/usr/bin/env bash
# Clean-machine quickstart (macOS / Linux).
#   ./scripts/quickstart.sh                      # installs deps + model, runs real iPhone 15 Pro photos + a real LiDAR capture from data/
#   ./scripts/quickstart.sh <capture_dir> [out]  # runs your capture (tier is auto-detected)
set -euo pipefail
start=$(date +%s)
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python scripts/fetch_models.py
echo "deps + model ready after $(( $(date +%s) - start )) s"
if [ "${1:-}" != "" ]; then
  .venv/bin/python -m floorscan run "$1" --out "${2:-out}"
else
  for s in iphone15pro_photos 42444946_lidar; do .venv/bin/python -m floorscan run "data/$s" --out "out/$s"; done
fi
echo "total $(( $(date +%s) - start )) s"
