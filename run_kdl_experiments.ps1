# run_kdl_experiments.ps1
# Chay Scenario 2 (baseline_od) & Scenario 3 (fedkdl) voi du lieu anh URPC
# Sau khi train xong tu dong goi cac plot script.

$N_LIST    = @(50, 100, 150, 200)
$DATASETS  = @("URPC")
$ALPHAS    = @(0.1, 10000.0)
$SEEDS     = @(42, 123, 2024)
$BASELINES = @("baseline_od", "fedkdl")

Write-Host "============================================================"
Write-Host "[START RUN] Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

$ROUNDS   = 20
$OUT_DIR  = "results/logs_kdl"
$ENVS_DIR = "environments"

if (-Not (Test-Path -Path $OUT_DIR)) {
    New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
}

$total = $N_LIST.Length * $DATASETS.Length * $ALPHAS.Length * $SEEDS.Length * $BASELINES.Length
$count = 0

foreach ($n in $N_LIST) {
    foreach ($ds in $DATASETS) {
        foreach ($alpha in $ALPHAS) {
            foreach ($seed in $SEEDS) {
                $topo      = Join-Path $ENVS_DIR "topo\N_$($n)\topo_N$($n)_seed$($seed).pkl"
                $alpha_str = [string]$alpha -replace '\.', 'p'
                $data      = Join-Path $ENVS_DIR "data\$($ds)\N_$($n)\data_N$($n)_$($ds)_a$($alpha_str)_seed$($seed).pkl"

                if (!(Test-Path $topo) -or !(Test-Path $data)) {
                    Write-Host "[Warning] Missing env: N=$n, DS=$ds, alpha=$alpha, seed=$seed. Run generate_all_envs.py first." -ForegroundColor Yellow
                    $count += $BASELINES.Length
                    continue
                }

                foreach ($baseline in $BASELINES) {
                    $count++
                    $log_file = Join-Path $OUT_DIR "log_N$($n)_$($ds)_a$($alpha_str)_$($baseline)_seed$($seed).json"

                    if (Test-Path $log_file) {
                        Write-Host "[$count/$total] Skip (exists): $log_file" -ForegroundColor DarkGray
                        continue
                    }

                    Write-Host "[$count/$total] OD | N=$n | DS=$ds | alpha=$alpha | seed=$seed | baseline=$baseline" -ForegroundColor Cyan

                    $logDir = "results/train_logs/kdl"

                    $env:PYTHONIOENCODING = "utf-8"
                    if (Test-Path ".venv/Scripts/python.exe") { $py = ".venv/Scripts/python.exe" } else { $py = "python" }
                    & $py main_trainer_od.py --topo $topo --data $data --baseline $baseline --rounds $ROUNDS --out-dir $OUT_DIR --log-dir $logDir

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "[Error] Run failed." -ForegroundColor Red
                    }
                }
            }
        }
    }
}

Write-Host "`n[KDL] All training runs done. Generating plots..." -ForegroundColor Green

$env:PYTHONIOENCODING = "utf-8"
if (Test-Path ".venv/Scripts/python.exe") { $py = ".venv/Scripts/python.exe" } else { $py = "python" }
& $py scripts/fedkdl/plot_od_comparison.py
& $py scripts/fedkdl/plot_od_scalability.py
& $py scripts/fedkdl/plot_heterogeneity.py
& $py scripts/fedkdl/eval_baselines.py --results-dir $OUT_DIR

Write-Host "`n[KDL] All done! Plots saved in results/." -ForegroundColor Green
