# EPW-A 4位量化运行脚本 (PowerShell)
# 适用于Windows系统

Write-Host "=== EPW-A 4位量化运行脚本 ===" -ForegroundColor Green
Write-Host ""

# 设置环境变量
$env:EPW_LOAD_IN_4BIT = "true"
Write-Host "已设置环境变量: EPW_LOAD_IN_4BIT=true" -ForegroundColor Yellow
Write-Host "这将使用4位量化，提供最快的加载速度" -ForegroundColor Cyan
Write-Host "注意：4位量化会降低模型精度，但显著减少内存使用" -ForegroundColor Red
Write-Host ""

# 检查Python是否可用
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "错误：未找到Python，请确保Python已安装并添加到PATH" -ForegroundColor Red
    exit 1
}

# 检查脚本文件是否存在
if (-not (Test-Path "epw-enhance-1.py")) {
    Write-Host "错误：未找到 epw-enhance-1.py 文件" -ForegroundColor Red
    Write-Host "请确保在正确的目录中运行此脚本" -ForegroundColor Yellow
    exit 1
}

Write-Host "开始运行EPW-A（4位量化模式）..." -ForegroundColor Green
Write-Host ""

# 运行EPW脚本
try {
    python epw-enhance-1.py
    Write-Host ""
    Write-Host "✅ EPW-A运行完成！" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "❌ EPW-A运行失败：" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 其他量化选项 ===" -ForegroundColor Cyan
Write-Host "8位量化: `$env:EPW_LOAD_IN_8BIT='true'; python epw-enhance-1.py" -ForegroundColor Yellow
Write-Host "快速加载: `$env:EPW_FAST_LOADING='true'; python epw-enhance-1.py" -ForegroundColor Yellow
Write-Host "CPU模式: `$env:EPW_USE_CPU='true'; python epw-enhance-1.py" -ForegroundColor Yellow
Write-Host "组合使用: `$env:EPW_LOAD_IN_4BIT='true'; `$env:EPW_USE_CPU='true'; python epw-enhance-1.py" -ForegroundColor Yellow 