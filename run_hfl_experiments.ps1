# run_hfl_experiments.ps1
# Chay tat ca kich ban Scenario 1 (HFL baseline) voi cau hinh Tach biet (Phase 5)
# Sau khi train xong tu dong goi cac plot script.

$N_LIST    = @(50, 100, 150, 200)
$DATASETS  = @("SMD", "SMAP", "MSL")
$ALPHAS    = @(0.1, 10000.0)
$SEEDS     = @(42, 123, 2024)
$BASELINES = @("hfl_selective", "hfl_nearest", "hfl_nocoop", "fedprox", "fedavg")

Write-Host "============================================================"
Write-Host "[START RUN] Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

$ROUNDS   = 20
$RHO_S    = 0.05
$OUT_DIR  = "results/logs"
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
                # --- New path scheme: environments/topo/N_{n}/ and environments/data/{ds}/N_{n}/ ---
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
                    $rho_str  = [string]$RHO_S -replace '\.', 'p'
                    $log_file = Join-Path $OUT_DIR "log_N$($n)_$($ds)_a$($alpha_str)_$($baseline)_rho$($rho_str)_seed$($seed).json"

                    if (Test-Path $log_file) {
                        Write-Host "[$count/$total] Skip (exists): $log_file" -ForegroundColor DarkGray
                        continue
                    }

                    Write-Host "[$count/$total] N=$n | DS=$ds | alpha=$alpha | seed=$seed | baseline=$baseline" -ForegroundColor Cyan

                    $logDir = "results/train_logs/hfl"

                    $env:PYTHONIOENCODING = "utf-8"
                    if (Test-Path ".venv/Scripts/python.exe") { $py = ".venv/Scripts/python.exe" } else { $py = "python" }
                    & $py main_trainer.py --topo $topo --data $data --baseline $baseline --rho-s $RHO_S --rounds $ROUNDS --out-dir $OUT_DIR --log-dir $logDir

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "[Error] Run failed." -ForegroundColor Red
                    }
                }
            }
        }
    }
}

Write-Host "`n[HFL] All training runs done. Generating plots..." -ForegroundColor Green

$env:PYTHONIOENCODING = "utf-8"
if (Test-Path ".venv/Scripts/python.exe") { $py = ".venv/Scripts/python.exe" } else { $py = "python" }
& $py scripts/hfl/plot_convergence.py
& $py scripts/hfl/plot_scalability.py
& $py scripts/hfl/plot_heterogeneity.py
& $py scripts/hfl/plot_real_benchmark.py

Write-Host "`n[HFL] All done! Plots saved in results/." -ForegroundColor Green
