$env:PYTHONIOENCODING="utf-8"

Write-Host "============================================="
Write-Host "Running Real Benchmarks (Scenario 1 & 2)..."
Write-Host "============================================="
python scripts_baseline/run_real_benchmarks.py

Write-Host "============================================="
Write-Host "Running Scenario 3 (FedKDL with YOLO)..."
Write-Host "============================================="
python scripts_fedkdl/run_scenario3_fedkdl.py

Write-Host "============================================="
Write-Host "Running Scalability Analysis..."
Write-Host "============================================="
python scripts_baseline/run_scalability.py

Write-Host "============================================="
Write-Host "Running Heterogeneity Analysis..."
Write-Host "============================================="
python scripts_baseline/run_heterogeneity.py

Write-Host "============================================="
Write-Host "ALL EXPERIMENTS COMPLETED!"
Write-Host "============================================="
