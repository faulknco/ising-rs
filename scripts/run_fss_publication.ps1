# Publication-Quality FSS with Wolff Algorithm
# =============================================
# Runs finite-size scaling on 3D cubic Ising model using Wolff cluster algorithm.
# Designed for overnight runs — expect 4-8 hours for full parameter set.
#
# Usage:
#   .\scripts\run_fss_publication.ps1                    # full run (overnight)
#   .\scripts\run_fss_publication.ps1 -Quick              # quick test (30 min)
#   .\scripts\run_fss_publication.ps1 -Sizes "8,12,16"   # custom sizes
#
# Output: analysis/data/fss_N<size>.csv for each size

param(
    [switch]$Quick,
    [string]$Sizes,
    [int]$Warmup,
    [int]$Samples,
    [int]$Steps,
    [double]$Tmin = 3.8,
    [double]$Tmax = 5.2,
    [string]$Outdir = "analysis/data"
)

$ErrorActionPreference = "Stop"

# Defaults based on mode
if ($Quick) {
    if (-not $Sizes)   { $Sizes   = "8,12,16,20,24" }
    if (-not $Warmup)  { $Warmup  = 2000 }
    if (-not $Samples) { $Samples = 5000 }
    if (-not $Steps)   { $Steps   = 41 }
    Write-Host "=== Quick mode: ~30 min ===" -ForegroundColor Yellow
} else {
    if (-not $Sizes)   { $Sizes   = "8,12,16,20,24,32,40,48" }
    if (-not $Warmup)  { $Warmup  = 5000 }
    if (-not $Samples) { $Samples = 20000 }
    if (-not $Steps)   { $Steps   = 61 }
    Write-Host "=== Publication mode: expect 4-8 hours ===" -ForegroundColor Cyan
}

# Build
Write-Host "`nBuilding fss binary..." -ForegroundColor Green
cargo build --release --bin fss 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Run
$exe = "target\release\fss.exe"
Write-Host "`nRunning Wolff FSS:" -ForegroundColor Green
Write-Host "  Sizes:   $Sizes"
Write-Host "  T range: $Tmin - $Tmax ($Steps steps)"
Write-Host "  Warmup:  $Warmup Wolff steps"
Write-Host "  Samples: $Samples Wolff steps"
Write-Host "  Output:  $Outdir\fss_N*.csv"
Write-Host ""

$start = Get-Date
& $exe --wolff --sizes $Sizes --tmin $Tmin --tmax $Tmax --steps $Steps --warmup $Warmup --samples $Samples --outdir $Outdir
$elapsed = (Get-Date) - $start

Write-Host "`n=== Done in $($elapsed.ToString('hh\:mm\:ss')) ===" -ForegroundColor Green
Write-Host "Output files:"
Get-ChildItem "$Outdir\fss_N*.csv" | ForEach-Object {
    Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)"
}

Write-Host "`nNext steps:"
Write-Host "  1. Open analysis/fss.ipynb in Jupyter"
Write-Host "  2. Run all cells to generate plots and exponent fits"
Write-Host "  3. Check Binder crossing for Tc, peak scaling for exponents"
