# Regenerate every reported number from the raw real captures; everything is written to results\.
#   .\scripts\reproduce.ps1              # uses the converted captures + ground truth in data\ (several hours on CPU: video SfM)
#   .\scripts\reproduce.ps1 -FetchData   # first re-download the ARKitScenes sample (~25 GB) + BD3, rebuild data\ and the fits
param([switch]$FetchData)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python"
if (-not (Test-Path .venv)) { python -m venv .venv }
& $py -m pip install -q -r requirements.txt
& $py scripts\fetch_models.py
$split = Get-Content data\SPLIT.json | ConvertFrom-Json
$visits = @($split.dev.PSObject.Properties.Name) + @($split.test.PSObject.Properties.Name)
if ($FetchData) {
  & $py dataprep\fetch_sample.py                                      # captures (+ laser samples) -> data\
  foreach ($v in $visits) {                                           # laser ground truth with the recorded windows / marks
    & $py dataprep\room_gt.py --visit $v
    & $py dataprep\openings_gt.py --visit $v
  }
  & $py dataprep\fit_lidar_scale.py                                   # DEV-only fits
  & $py dataprep\fit_depth_scale.py
  & $py dataprep\train_damage_head.py                                 # damage head on BD3 (real photos) -> results\damage\
}
$env:FLOORSCAN_NO_CAL = "1"; & $py bench\run_bench.py --split dev --force --no-ablation; Remove-Item Env:FLOORSCAN_NO_CAL
& $py bench\calibrate.py --from-results results\benchmark\metrics.json # interval calibration from the DEV entries
& $py bench\run_bench.py --split test --force                         # TEST rooms, held out, + drift ablation
& $py bench\run_bench.py                                              # merged report over all plans (no re-run)
& $py bench\headtohead.py                                             # NOT RUN table unless an app export is supplied
& $py fixloop\run_fixloop.py                                          # before / after of the declared fix
& $py report\build_report.py                                          # technical report PDF
& $py -m pytest -q tests
