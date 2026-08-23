param(
    [ValidateSet("saxpy", "naive_gemm", "unknown_op", "tiled_transpose", "all")]
    [string]$Fixture = "all"
)

# Runs each comparison fixture's Wukong program and its C peer and prints what they print. It does
# not time anything and does not publish a ratio -- `wukong_xbench compare` is the timed harness.
#
# It sets NO environment variable. It used to export WUKONG_NO_RECOGNIZE=1 inside the loop and never
# clear it, which leaked into the caller's shell: an auditor who ran this script and then re-derived
# the `--emit=mir -O2 | grep wukong_` census in the same terminal got the committed answer whether or
# not `@generic` was present -- a circular oracle sitting on the corpus's only instrument. The
# variable was also redundant, and redundant in a way that mattered: every fixture already carries
# `@generic`, which is the per-fixture opt-out the corpus exists to demonstrate, and forcing the
# global switch on top meant the script never actually exercised it. Measured before removing it --
# all four fixtures produce an identical census either way.
#
# The premise is now checked instead of assumed. After each run the MIR is re-derived and any
# recognized-kernel dispatch is a failure, so a fixture that silently loses its `@generic` cannot
# quietly turn this into a measurement of the dispatcher. `wukong_parallel_for` is exempt: it is the
# threading runtime entry point, not a recognizer, and `tiled_transpose` is `@parallel`.

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manifest = Join-Path $repo "Cargo.toml"
$fixtures = if ($Fixture -eq "all") {
    @("saxpy", "naive_gemm", "unknown_op", "tiled_transpose")
} else {
    @($Fixture)
}

$gcc = Get-Command gcc -ErrorAction SilentlyContinue
foreach ($name in $fixtures) {
    $wk = Join-Path (Join-Path $PSScriptRoot $name) "$name.wk"

    Write-Output "== ${name}: Wukong recognition-off/native =="
    & cargo run -q -p wukongc --manifest-path $manifest -- --run --backend=native -O2 $wk
    if ($LASTEXITCODE -ne 0) { throw "Wukong run failed for $name" }

    $mir = & cargo run -q -p wukongc --manifest-path $manifest -- --emit=mir -O2 $wk
    if ($LASTEXITCODE -ne 0) { throw "Wukong --emit=mir failed for $name" }
    $dispatch = $mir |
        Select-String -Pattern "call wukong_" -AllMatches |
        Where-Object { $_.Line -notmatch "wukong_parallel_for" }
    if ($dispatch) {
        throw ("$name dispatches a recognized kernel, so this is not a recognition-off run: " +
               ($dispatch -join "; ") + " -- check the fixture still carries @generic")
    }
    Write-Output "== ${name}: census clean (no recognized kernel) =="

    if (-not $gcc) {
        Write-Output "== ${name}: C skipped (gcc not found) =="
        continue
    }

    Write-Output "== ${name}: C (-O3 -march=native) =="
    $exe = Join-Path ([System.IO.Path]::GetTempPath()) "wukong-compare-$name.exe"
    & $gcc.Source -O3 -march=native (Join-Path (Join-Path $PSScriptRoot $name) "$name.c") -o $exe
    if ($LASTEXITCODE -ne 0) { throw "C build failed for $name" }
    try {
        & $exe
        if ($LASTEXITCODE -ne 0) { throw "C run failed for $name" }
    } finally {
        Remove-Item $exe -ErrorAction SilentlyContinue
    }
}
