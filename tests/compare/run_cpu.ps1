param(
    [ValidateSet("saxpy", "naive_gemm", "unknown_op", "tiled_transpose", "all")]
    [string]$Fixture = "all"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$fixtures = if ($Fixture -eq "all") {
    @("saxpy", "naive_gemm", "unknown_op", "tiled_transpose")
} else {
    @($Fixture)
}

$gcc = Get-Command gcc -ErrorAction SilentlyContinue
foreach ($name in $fixtures) {
    $base = Join-Path $PSScriptRoot $name
    $stem = if ($name -eq "naive_gemm") { "naive_gemm" } else { $name }

    Write-Output "== ${name}: Wukong recognition-off/native =="
    $env:WUKONG_NO_RECOGNIZE = "1"
    & cargo run -q -p wukongc --manifest-path (Join-Path $repo "Cargo.toml") -- `
        --run --backend=native -O2 (Join-Path $base "$stem.wk")
    if ($LASTEXITCODE -ne 0) { throw "Wukong run failed for $name" }

    if (-not $gcc) {
        Write-Output "== ${name}: C skipped (gcc not found) =="
        continue
    }

    Write-Output "== ${name}: C (-O3 -march=native) =="
    $exe = Join-Path ([System.IO.Path]::GetTempPath()) "wukong-compare-$name.exe"
    & $gcc.Source -O3 -march=native (Join-Path $base "$stem.c") -o $exe
    if ($LASTEXITCODE -ne 0) { throw "C build failed for $name" }
    try {
        & $exe
        if ($LASTEXITCODE -ne 0) { throw "C run failed for $name" }
    } finally {
        Remove-Item $exe -ErrorAction SilentlyContinue
    }
}
