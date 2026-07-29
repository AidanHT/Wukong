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
  The law binds BOTH sides of a comparison: the power class is sampled around the torch peer too, and
  a vs-torch verdict is REPORTABLE only if the Wukong regimes AND the peer were clean. A peer timed on
  battery is 2-4x slow, which shows up as a Wukong "win" that the instrument, not the compiler, made.

  Usage:  pwsh tools/measure_gpt2.ps1            # full sweep
          pwsh tools/measure_gpt2.ps1 -SkipTorch # Wukong only (no python env needed)
          pwsh tools/measure_gpt2.ps1 -Outer 5   # best-of-5 outer reps per Wukong regime (default 3)
#>
[CmdletBinding()]
param(
    [int]$Outer = 3,
    [switch]$SkipTorch,
    [switch]$SkipBuild,
    # Light/fast mode: measure only the single-core regimes (both Wukong 1c + torch), skipping the heavy
    # Wukong all-core run. Single-core load is ~20 W, which an adapter covers, so this can complete inside
    # a brief stable-AC window (e.g. a battery-care trickle) without the all-core draw knocking off AC.
    [switch]$SingleCoreOnly,
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
    if ($pct -ge 99 -and $charge -le 2000) {
        return [pscustomobject]@{ Class='REPORTABLE'; Detail="AC + full (${pct}%, ${charge} mW trickle): all-core is a real claim"; Reportable=$true }
    }
    return [pscustomobject]@{ Class='ALL-CORE CAPPED'; Detail="AC + charging (${pct}%, +${charge} mW): 1c fine, all-core throttled ~25%"; Reportable=$false }
}

# Rank a power class for "worse of two samples" (lower = worse): battery < capped < reportable.
function Power-Rank([string]$class) { switch ($class) { 'NON-REPORTABLE' { 0 } 'ALL-CORE CAPPED' { 1 } 'REPORTABLE' { 2 } default { 0 } } }

# Run one Wukong vehicle, return @{ Argmax; Ms; Power } — the min forward-ms over $Outer outer reps and the
# WORST power class sampled immediately before and after the run. Per-regime sampling matters because a
# heavy all-core run can knock the machine onto battery mid-sweep (adapter can't cover the draw / battery-
# care), which would silently corrupt a run the top-of-script banner had labelled reportable. The vehicle
# prints "<argmax> <min_microseconds> <nreps>" (min-of-8 in-program).
function Measure-Vehicle {
    param([string]$Wk, [hashtable]$EnvVars, [int]$Reps)
    $bestMs = [double]::PositiveInfinity
    $argmax = $null
    $before = Get-PowerClass
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
    $after = Get-PowerClass
    $worst = if ((Power-Rank $before.Class) -le (Power-Rank $after.Class)) { $before } else { $after }
    return [pscustomobject]@{ Argmax = $argmax; Ms = $bestMs; Power = $worst }
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

# Order matters: run the LIGHT single-core regimes first (an adapter easily covers ~20 W, so they stay
# on stable AC and are reportable), and the HEAVY all-core run LAST — it can knock the machine onto
# battery, but by then the single-core numbers are already captured under their own (clean) power window.
Write-Host "`n[wukong] serial vehicle, 1 core ..."
$s1 = Measure-Vehicle -Wk $serial -EnvVars @{ RAYON_NUM_THREADS = '1' } -Reps $Outer
Write-Host "[wukong] @parallel vehicle, 1 core ..."
$p1 = Measure-Vehicle -Wk $par -EnvVars @{ RAYON_NUM_THREADS = '1' } -Reps $Outer
if (-not $SingleCoreOnly) {
    Write-Host "[wukong] @parallel vehicle, all core (heavy — may drop AC) ..."
    $pAll = Measure-Vehicle -Wk $par -EnvVars @{} -Reps $Outer
} else {
    Write-Host "[wukong] @parallel all-core SKIPPED (-SingleCoreOnly: staying light to hold AC)"
    $pAll = [pscustomobject]@{ Argmax = 338; Ms = [double]::PositiveInfinity; Power = (Get-PowerClass) }
}

# Correctness gate: every measured regime must agree on argmax 338 or the timing is meaningless.
$measuredArgmax = if ($SingleCoreOnly) { @($s1.Argmax, $p1.Argmax) } else { @($s1.Argmax, $pAll.Argmax, $p1.Argmax) }
$argmaxes = $measuredArgmax | Sort-Object -Unique
$argmaxOk = ($argmaxes.Count -eq 1 -and $argmaxes[0] -eq 338)

# Single-core self-consistency: serial-1c and @parallel-1c are the IDENTICAL computation, so if the
# machine wasn't throttling they agree tightly. A big gap is a direct throttle detector, independent of
# the power labels — the honest gate for the single-core numbers.
$sc1cOk = ($s1.Ms -gt 0 -and $p1.Ms -gt 0 -and [double]::IsFinite($s1.Ms) -and [double]::IsFinite($p1.Ms))
$sc1cSkew = if ($sc1cOk) { [math]::Abs($s1.Ms - $p1.Ms) / [math]::Min($s1.Ms, $p1.Ms) } else { 1.0 }

# Per-regime reportability: single-core needs only stable AC (light load, charging-immune) => any non-
# battery class; all-core needs AC+full trickle => REPORTABLE class. Tag each row with what it earned.
function Tag-1c($p) { if ($p.Class -eq 'NON-REPORTABLE') { 'battery/NR' } else { 'reportable' } }
function Tag-all($p) { if ($p.Class -eq 'REPORTABLE') { 'reportable' } elseif ($p.Class -eq 'ALL-CORE CAPPED') { 'CAPPED' } else { 'battery/NR' } }

Write-Host ""
Write-Host ("{0,-30} {1,10} {2,-12}" -f "regime", "ms/fwd", "power")
Write-Host ("{0,-30} {1,10:N1} {2,-12}" -f "wukong serial       1c", $s1.Ms, (Tag-1c $s1.Power))
Write-Host ("{0,-30} {1,10:N1} {2,-12}" -f "wukong @parallel    1c", $p1.Ms, (Tag-1c $p1.Power))
if (-not $SingleCoreOnly) {
    Write-Host ("{0,-30} {1,10:N1} {2,-12}" -f "wukong @parallel all-core", $pAll.Ms, (Tag-all $pAll.Power))
    if ($p1.Ms -gt 0 -and [double]::IsFinite($p1.Ms) -and $pAll.Ms -gt 0 -and [double]::IsFinite($pAll.Ms)) {
        Write-Host ("{0,-30} {1,10:N2}x" -f "  -> @parallel scaling", ($p1.Ms / $pAll.Ms))
    }
}
Write-Host ("  single-core self-consistency (serial-1c vs par-1c): {0:P1} skew  [{1}]" -f $sc1cSkew, $(if ($sc1cSkew -le 0.05) { 'STABLE' } else { 'THROTTLED — 1c numbers suspect' }))

# The PEER is measured under the same instrument law as the Wukong regimes. bench_gpt2_torch.py has
# no power gate of its own (min-of-5 + 2 warmups, nothing else), so if AC drops between the Wukong
# sweep and the peer run — battery-care disengaging, a bumped adapter — the peer eats the documented
# 2-4x battery DVFS penalty and a fabricated Wukong win gets published under this script's own
# REPORTABLE stamp. Sample around the peer too, keep the worse class, and fold it into the verdict.
$tPower = $null
if (-not $SkipTorch) {
    $py = if (Test-Path $TorchPython) { $TorchPython } else { "python" }
    Write-Host "`n[torch] HF GPT2Model peer (eager 1t / all-t / compiled) via $py ..."
    $tBefore = Get-PowerClass
    Push-Location $repo
    try { & $py tools/bench_gpt2_torch.py 512 } catch { Write-Host "  torch peer failed: $_" }
    Pop-Location
    $tAfter = Get-PowerClass
    $tPower = if ((Power-Rank $tBefore.Class) -le (Power-Rank $tAfter.Class)) { $tBefore } else { $tAfter }
    Write-Host ("  torch peer power: {0} — {1}" -f $tPower.Class, $tPower.Detail)
}

Write-Host ""
if ($argmaxOk) { Write-Host "correctness: all Wukong regimes argmax=338  OK" }
else { Write-Host "correctness: ARGMAX MISMATCH ($($argmaxes -join ',')) — TIMING INVALID until fixed" }

# A vs-torch claim needs BOTH sides clean. An unmeasured peer (-SkipTorch) is not a clean peer:
# with no peer number at all there is no comparison to stamp, so it cannot be reportable either.
$peerOk1c  = ($null -ne $tPower) -and (Tag-1c  $tPower) -eq 'reportable'
$peerOkAll = ($null -ne $tPower) -and (Tag-all $tPower) -eq 'reportable'
$wukOk1c  = $argmaxOk -and $sc1cSkew -le 0.05 -and (Tag-1c $s1.Power) -eq 'reportable' -and (Tag-1c $p1.Power) -eq 'reportable'
$wukOkAll = $argmaxOk -and (Tag-all $pAll.Power) -eq 'reportable'
$scReportable = $wukOk1c -and $peerOk1c
$allReportable = $wukOkAll -and $peerOkAll
Write-Host ""
if ($scReportable) { Write-Host "-> SINGLE-CORE: REPORTABLE (stable AC across the Wukong regimes AND the peer, 1c self-consistent) — serial-1c vs torch eager-1t is a real claim." }
elseif ($SkipTorch) { Write-Host "-> SINGLE-CORE: no peer measured (-SkipTorch) — Wukong 1c timings only. NOT a vs-torch claim." }
elseif ($wukOk1c) { Write-Host "-> SINGLE-CORE: NOT reportable — the Wukong side was clean but the torch peer ran under $($tPower.Class). A peer timed under DVFS manufactures a Wukong win; re-run the peer on stable AC." }
else { Write-Host "-> SINGLE-CORE: NOT reportable (battery or >5% 1c skew). Re-run on stable AC." }
if ($SingleCoreOnly) { Write-Host "-> ALL-CORE: skipped (-SingleCoreOnly). Run the full sweep on AC+full for the all-core claim." }
elseif ($allReportable) { Write-Host "-> ALL-CORE: REPORTABLE (AC+full trickle held through the heavy run AND the peer) — vs torch eager all-thread is a real claim." }
elseif ($SkipTorch) { Write-Host "-> ALL-CORE: no peer measured (-SkipTorch) — Wukong all-core timing only. NOT a vs-torch claim." }
elseif ($wukOkAll) { Write-Host "-> ALL-CORE: NOT reportable — the Wukong all-core run held AC+full but the torch peer ran under $($tPower.Class). DIRECTIONAL ONLY until the peer is re-timed on AC+full." }
else { Write-Host "-> ALL-CORE: NOT reportable (needs AC+full that survives the all-core draw — disable battery-care, ensure the adapter covers peak). DIRECTIONAL ONLY." }
Write-Host ""
