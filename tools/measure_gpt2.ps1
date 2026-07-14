<#
  measure_gpt2.ps1 — the honest, repeatable real-GPT-2 measurement sweep.

  Runs, adjacently (same power state, same run), every regime we compare:
    * Wukong serial vehicle,   1 core   (examples/gpt2_forward_bench.wk,     RAYON_NUM_THREADS=1)
    * Wukong @parallel vehicle, all core (examples/gpt2_forward_bench_par.wk, default pool = physical cores)
    * Wukong @parallel vehicle, 1 core   (RAYON_NUM_THREADS=1 → wuk_pool_width()==1 serial fast-path)
    * PyTorch HF GPT2Model: eager 1-thread, eager all-thread, compiled all-thread (tools/bench_gpt2_torch.py)

  POWER LAW (see memory power-state-is-part-of-the-instrument): this is the whole point of the gate.
    * battery (Offline)        -> NON-REPORTABLE (single-core ±58%, DVFS noise). Numbers printed but flagged.
    * AC + charging (<~99%)    -> all-core CAPPED ~25% (charging draws the power budget); 1c is fine.
    * AC + full (~100%, ~0 W)  -> REPORTABLE. This is the only state whose all-core number is a real claim.
  The banner tells you which you got. Only publish REPORTABLE all-core numbers.

  Usage:  pwsh tools/measure_gpt2.ps1            # full sweep
          pwsh tools/measure_gpt2.ps1 -SkipTorch # Wukong only (no python env needed)
          pwsh tools/measure_gpt2.ps1 -Outer 5   # best-of-5 outer reps per Wukong regime (default 3)
#>
[CmdletBinding()]
param(
    [int]$Outer = 3,
    [switch]$SkipTorch,
    [switch]$SkipBuild,
    # Python with torch(cpu)+transformers for the HF peer. Default: the coalescence conda env
    # (torch 2.11.0+cpu, transformers 4.41.2 — a CPU build, the honest fp32-CPU peer). Falls back to
    # `python` on PATH if that interpreter is missing.
    [string]$TorchPython = "$env:USERPROFILE\Anaconda3\envs\coalescence\python.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$exe  = Join-Path $repo "target\release\wukongc.exe"
$serial = Join-Path $repo "examples\gpt2_forward_bench.wk"
$par    = Join-Path $repo "examples\gpt2_forward_bench_par.wk"

function Get-PowerClass {
    $ps  = [System.Windows.Forms.SystemInformation]::PowerStatus
    $online = $ps.PowerLineStatus -eq 'Online'
    $pct = [math]::Round($ps.BatteryLifePercent * 100)
    $charge = 0; $discharge = 0
    try {
        $w = Get-CimInstance -ClassName BatteryStatus -Namespace root\wmi -ErrorAction Stop
        $charge = $w.ChargeRate; $discharge = $w.DischargeRate
    } catch {}
    if (-not $online) {
        return [pscustomobject]@{ Class='NON-REPORTABLE'; Detail="battery (Offline, ${pct}%, -${discharge} mW): single-core noisy, all-core meaningless"; Reportable=$false }
    }
    if ($pct -ge 99 -and $charge -le 100) {
        return [pscustomobject]@{ Class='REPORTABLE'; Detail="AC + full (${pct}%, ${charge} mW): all-core is a real claim"; Reportable=$true }
    }
    return [pscustomobject]@{ Class='ALL-CORE CAPPED'; Detail="AC + charging (${pct}%, +${charge} mW): 1c fine, all-core throttled ~25%"; Reportable=$false }
}

# Run one Wukong vehicle, return @{ Argmax; Ms } taking the min forward-ms over $Outer outer reps.
# The vehicle itself prints "<argmax> <min_microseconds> <nreps>" (min-of-8 in-program).
function Measure-Vehicle {
    param([string]$Wk, [hashtable]$EnvVars, [int]$Reps)
    $bestMs = [double]::PositiveInfinity
    $argmax = $null
    for ($i = 0; $i -lt $Reps; $i++) {
        foreach ($k in $EnvVars.Keys) { Set-Item -Path "Env:$k" -Value $EnvVars[$k] }
        $out = & $exe --run --backend=native $Wk 2>$null
        foreach ($k in $EnvVars.Keys) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
        $lines = @($out | Where-Object { $_ -match '^\s*-?\d+\s*$' } | ForEach-Object { [long]($_.Trim()) })
        if ($lines.Count -ge 2) {
            $argmax = $lines[0]
            $ms = $lines[1] / 1000.0
            if ($ms -lt $bestMs) { $bestMs = $ms }
        }
    }
    return [pscustomobject]@{ Argmax = $argmax; Ms = $bestMs }
}

Add-Type -AssemblyName System.Windows.Forms
$power = Get-PowerClass

Write-Host ""
Write-Host "=============================================================================="
Write-Host " real GPT-2 124M forward, S=512  —  power: $($power.Class)"
Write-Host "   $($power.Detail)"
Write-Host "=============================================================================="

if (-not $SkipBuild) {
    Write-Host "`n[build] cargo build --release -p wukongc ..."
    Push-Location $repo
    cargo build --release -p wukongc 2>&1 | Select-Object -Last 1
    Pop-Location
}
if (-not (Test-Path $exe)) { throw "release wukongc not found at $exe (drop -SkipBuild)" }

Write-Host "`n[wukong] serial vehicle, 1 core ..."
$s1 = Measure-Vehicle -Wk $serial -EnvVars @{ RAYON_NUM_THREADS = '1' } -Reps $Outer
Write-Host "[wukong] @parallel vehicle, all core ..."
$pAll = Measure-Vehicle -Wk $par -EnvVars @{} -Reps $Outer
Write-Host "[wukong] @parallel vehicle, 1 core ..."
$p1 = Measure-Vehicle -Wk $par -EnvVars @{ RAYON_NUM_THREADS = '1' } -Reps $Outer

# Correctness gate: every regime must agree on argmax 338 or the timing is meaningless.
$argmaxes = @($s1.Argmax, $pAll.Argmax, $p1.Argmax) | Sort-Object -Unique
$argmaxOk = ($argmaxes.Count -eq 1 -and $argmaxes[0] -eq 338)

Write-Host ""
Write-Host ("{0,-34} {1,10}" -f "regime", "ms/fwd")
Write-Host ("{0,-34} {1,10:N1}" -f "wukong serial          1c", $s1.Ms)
Write-Host ("{0,-34} {1,10:N1}" -f "wukong @parallel  all-core", $pAll.Ms)
Write-Host ("{0,-34} {1,10:N1}" -f "wukong @parallel       1c", $p1.Ms)
if ($p1.Ms -gt 0 -and [double]::IsFinite($p1.Ms)) {
    Write-Host ("{0,-34} {1,10:N2}x" -f "  -> @parallel scaling (1c/allcore)", ($p1.Ms / $pAll.Ms))
}

if (-not $SkipTorch) {
    $py = if (Test-Path $TorchPython) { $TorchPython } else { "python" }
    Write-Host "`n[torch] HF GPT2Model peer (eager 1t / all-t / compiled) via $py ..."
    Push-Location $repo
    try { & $py tools/bench_gpt2_torch.py 512 } catch { Write-Host "  torch peer failed: $_" }
    Pop-Location
}

Write-Host ""
if ($argmaxOk) { Write-Host "correctness: all Wukong regimes argmax=338  OK" }
else { Write-Host "correctness: ARGMAX MISMATCH ($($argmaxes -join ',')) — TIMING INVALID until fixed" }
if ($power.Reportable) { Write-Host "-> REPORTABLE run: these numbers may be published (cross-check torch argmax + rel in the peer output)." }
else { Write-Host "-> $($power.Class): numbers are DIRECTIONAL ONLY. Re-run at AC + full charge before publishing." }
Write-Host ""
