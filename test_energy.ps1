$TOPO = Get-ChildItem -Path "environments\2d\topo" -Filter "*.pkl" -Recurse | Select-Object -First 1
$DATA = Get-ChildItem -Path "environments\2d\data" -Filter "*.pkl" -Recurse | Select-Object -First 1

if (-not $TOPO -or -not $DATA) {
    Write-Host "[!] Không tìm thấy file môi trường (.pkl). Hãy tạo môi trường trước!" -ForegroundColor Red
    exit
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "KIỂM CHỨNG TOÁN HỌC: NĂNG LƯỢNG TÍNH TOÁN" -ForegroundColor Cyan
Write-Host "Topo: $($TOPO.FullName)" -ForegroundColor Cyan
Write-Host "Data: $($DATA.FullName)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$BASELINES = @("fedkdl", "fedkdl_nolora", "topk_grad")

foreach ($b in $BASELINES) {
    Write-Host "`n>>> Đang chạy Baseline: $b (1 Vòng FL) ..." -ForegroundColor Yellow
    
    # Chạy lệnh và hiển thị trực tiếp để thấy thanh tiến trình (nếu có)
    python main_trainer_od.py --topo "$($TOPO.FullName)" --data "$($DATA.FullName)" --baseline $b --rounds 1
    
    Write-Host ">>> Xong $b ! Hãy cuộn lên xem mục 'e_comp' trong log Metrics nhé." -ForegroundColor Green
    Write-Host "--------------------------------------------------------"
}
