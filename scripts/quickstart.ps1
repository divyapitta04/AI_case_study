# Clean-machine quickstart (Windows PowerShell). Target: capture folder -> plan.json/png in < 15 minutes.
#   .\scripts\quickstart.ps1                       # installs deps + model, runs real iPhone 15 Pro photos + a real LiDAR capture from data\
#   .\scripts\quickstart.ps1 -All                  # ... and the real video capture (COLMAP SfM, adds ~10 min on CPU)
#   .\scripts\quickstart.ps1 -Capture D:\my\scan   # runs your capture (tier is auto-detected)
param([string]$Capture = "", [string]$Out = "out", [switch]$All)
$ErrorActionPreference = "Stop"
$sw = [Diagnostics.Stopwatch]::StartNew()
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install -q --upgrade pip
.\.venv\Scripts\python -m pip install -q -r requirements.txt
.\.venv\Scripts\python scripts\fetch_models.py
"deps + model ready after {0:n0} s" -f $sw.Elapsed.TotalSeconds
if ($Capture -ne "") {
  .\.venv\Scripts\python -m floorscan run $Capture --out $Out
} else {
  $caps = @("iphone15pro_photos", "42444946_lidar")
  if ($All) { $caps += "42444946_video" }
  foreach ($s in $caps) {
    .\.venv\Scripts\python -m floorscan run "data\$s" --out "$Out\$s"
  }
}
"total {0:n0} s" -f $sw.Elapsed.TotalSeconds
