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
    * AC + full (~100%, ~0 W)  -> REPORTABLE power class.
  REPORTABLE below covers this script's correctness/power checks only. Controlled order, preparation,
  thermal state, and host-idle eligibility still require the coordinator's measurement protocol.
  The law binds BOTH sides of a comparison: the power class is sampled around the torch peer too, and
  a vs-torch verdict is REPORTABLE only if the Wukong regimes AND the peer were clean. A peer timed on
  battery is 2-4x slow, which shows up as a Wukong "win" that the instrument, not the compiler, made.

  Usage:  pwsh tools/measure_gpt2.ps1            # full sweep
          pwsh tools/measure_gpt2.ps1 -SkipTorch # Wukong only (no python env needed)
          pwsh tools/measure_gpt2.ps1 -Outer 5   # best-of-5 outer reps per Wukong regime (default 3)
#>
[CmdletBinding()]
param(
    [ValidateRange(1, [int]::MaxValue)][int]$Outer = 3,
    [switch]$SkipTorch,
    [switch]$SkipBuild,
    # Light/fast mode: measure only the single-core regimes (both Wukong 1c + torch), skipping the heavy
    # Wukong all-core run. Single-core load is ~20 W, which an adapter covers, so this can complete inside
    # a brief stable-AC window. The existing Torch peer still includes heavy all-thread/compiled regimes.
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
        return [pscustomobject]@{ Class='REPORTABLE'; Detail="AC + full (${pct}%, ${charge} mW trickle): all-core power gate passed"; Reportable=$true }
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
    $records = @()
    $before = Get-PowerClass
    for ($i = 0; $i -lt $Reps; $i++) {
        $saved = @{}
        # All-core means the default pool, even if the caller set a thread cap.
        $keys = @(@('RAYON_NUM_THREADS') + @($EnvVars.Keys) | Select-Object -Unique)
        $out = @(); $code = $null; $failure = $null
        $reference = Join-Path $repo 'data/gpt2/gpt2_bench_lastrow.bin'
        $oldStamp = if (Test-Path -LiteralPath $reference) { (Get-Item -LiteralPath $reference).LastWriteTimeUtc.Ticks } else { $null }
        $started = [DateTime]::UtcNow.Ticks
        Push-Location $repo
        try {
            foreach ($k in $keys) {
                $saved[$k] = @{ Exists=(Test-Path "Env:$k"); Value=[Environment]::GetEnvironmentVariable($k, 'Process') }
                if ($EnvVars.ContainsKey($k) -and $null -ne $EnvVars[$k]) {
                    [Environment]::SetEnvironmentVariable($k, $EnvVars[$k], 'Process')
                } else { Remove-Item -LiteralPath "Env:$k" -ErrorAction SilentlyContinue }
            }
            $out = @(& $exe --run --backend=native $Wk)
            $code = $LASTEXITCODE
        } catch { $failure = $_.ToString() }
        finally {
            foreach ($k in $saved.Keys) {
                if ($saved[$k].Exists) { [Environment]::SetEnvironmentVariable($k, $saved[$k].Value, 'Process') }
                else { Remove-Item -LiteralPath "Env:$k" -ErrorAction SilentlyContinue }
            }
            Pop-Location
        }
        $record = Convert-VehicleResult -Output $out -ExitCode $code -Failure $failure
        if ($record.Valid) {
            $fresh = Get-Item -LiteralPath $reference -ErrorAction SilentlyContinue
            if ($null -eq $fresh -or $fresh.Length -ne (50257 * 4) -or
                $fresh.LastWriteTimeUtc.Ticks -lt $started -or $fresh.LastWriteTimeUtc.Ticks -eq $oldStamp) {
                $record.Valid = $false
                $record.Error = 'missing, stale, or malformed last-row reference'
            }
        }
        $records += $record
        if ($record.Valid) {
            if ($record.Ms -lt $bestMs) { $bestMs = $record.Ms }
        } else {
            Write-Host "  repetition $($i + 1) INVALID: $($record.Error)"
            foreach ($line in $out) { Write-Host "  stdout: $line" }
        }
    }
    $after = Get-PowerClass
    $worst = if ((Power-Rank $before.Class) -le (Power-Rank $after.Class)) { $before } else { $after }
    $valid = $records.Count -eq $Reps -and @($records | Where-Object { -not $_.Valid }).Count -eq 0
    return [pscustomobject]@{ Argmax = $(if ($valid) { 338 } else { $null });
        Ms = $(if ($valid) { $bestMs } else { [double]::PositiveInfinity }); Power = $worst;
        Valid = $valid; Repetitions = $records }
}

function Convert-VehicleResult {
    param([AllowEmptyCollection()][object[]]$Output, $ExitCode, [string]$Failure)
    $result = [pscustomobject]@{ Valid=$false; Argmax=$null; Ms=$null; ExitCode=$ExitCode; Error=$Failure }
    if ($Failure) { return $result }
    if ($null -eq $ExitCode -or $ExitCode -ne 0) { $result.Error = "vehicle exit code: $ExitCode"; return $result }
    $lines = @($Output | ForEach-Object { "$_".Trim() } | Where-Object { $_ -ne '' })
    if ($lines.Count -ne 3 -or @($lines | Where-Object { $_ -notmatch '^-?\d+$' }).Count -ne 0) {
        $result.Error = 'expected exactly argmax, microseconds, and nreps'; return $result
    }
    try { $argmax = [long]$lines[0]; $us = [long]$lines[1]; $nreps = [long]$lines[2] }
    catch { $result.Error = 'integer output overflow'; return $result }
    if ($argmax -ne 338 -or $us -le 0 -or $nreps -ne 8) {
        $result.Error = "invalid vehicle result: $($lines -join ', ')"; return $result
    }
    $result.Valid = $true; $result.Argmax = $argmax; $result.Ms = $us / 1000.0
    return $result
}

function Test-PeerResult {
    param([AllowEmptyCollection()][object[]]$Output, $ExitCode)
    if ($null -eq $ExitCode -or $ExitCode -ne 0) { return $false }
    $rows = @($Output | Where-Object { "$_".StartsWith('GPT2_PEER_RESULT ') })
    if ($rows.Count -ne 1) { return $false }
    try {
        $r = $rows[0].Substring('GPT2_PEER_RESULT '.Length) | ConvertFrom-Json -NoEnumerate -ErrorAction Stop
        # The [pscustomobject] accelerator also matches wrapped arrays; require
        # the concrete JSON object type so array property enumeration cannot pass.
        if ($r.GetType() -ne [System.Management.Automation.PSCustomObject] -or
            $r.results.GetType() -ne [System.Management.Automation.PSCustomObject] -or
            $r.status -isnot [string]) { return $false }
        foreach ($name in @('schema', 'sequence_length', 'vocab', 'argmax')) {
            if ($r.$name -isnot [int] -and $r.$name -isnot [long]) { return $false }
        }
        if ($r.schema -ne 1 -or $r.status -ne 'ok' -or $r.sequence_length -ne 512 -or
            $r.vocab -ne 50257 -or $r.argmax -ne 338 -or $null -eq $r.relative_error) { return $false }
        if (-not (Test-FiniteNumber $r.relative_error) -or $r.relative_error -lt 0 -or $r.relative_error -gt 1e-3) { return $false }
        foreach ($name in @('eager 1-thread', 'eager all-thread')) {
            $ms = $r.results.$name
            if (-not (Test-FiniteNumber $ms) -or $ms -le 0) { return $false }
        }
        if (@($r.results.PSObject.Properties.Name | Where-Object {
            $_ -notin @('eager 1-thread', 'eager all-thread', 'compiled all-thread')
        }).Count -ne 0) { return $false }
        if ('compiled all-thread' -notin $r.results.PSObject.Properties.Name) { return $false }
        $compiled = $r.results.'compiled all-thread'
        if ($null -ne $compiled -and (-not (Test-FiniteNumber $compiled) -or $compiled -le 0)) { return $false }
        return $true
    } catch { return $false }
}

function Test-FiniteNumber($Value) {
    return ($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]) -and
        [double]::IsFinite([double]$Value)
}

Add-Type -AssemblyName System.Windows.Forms
$power = Get-PowerClass

Write-Host ""
Write-Host "=============================================================================="
Write-Host 'REPORTABLE covers script validation/power only; order, preparation, thermal state, and host-idle eligibility remain unverified.'
Write-Host " real GPT-2 124M forward, S=512  —  power: $($power.Class)"
Write-Host "   $($power.Detail)"
Write-Host "=============================================================================="

if (-not $SkipBuild) {
    Write-Host "`n[build] cargo build --release -p wukongc ..."
    Push-Location $repo
    try {
        & cargo build --release --locked -j1 -p wukongc
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
    } finally { Pop-Location }
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
    $pAll = [pscustomobject]@{ Argmax = $null; Ms = [double]::PositiveInfinity; Power = (Get-PowerClass); Valid = $false }
}

# Correctness gate: every measured regime must agree on argmax 338 or the timing is meaningless.
$measuredArgmax = if ($SingleCoreOnly) { @($s1.Argmax, $p1.Argmax) } else { @($s1.Argmax, $pAll.Argmax, $p1.Argmax) }
$argmaxes = $measuredArgmax | Sort-Object -Unique
$argmaxOk = $s1.Valid -and $p1.Valid -and ($SingleCoreOnly -or $pAll.Valid) -and ($argmaxes.Count -eq 1 -and $argmaxes[0] -eq 338)

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
$peerValid = $false
if (-not $SkipTorch) {
    $py = if (Test-Path $TorchPython) { $TorchPython } else { "python" }
    Write-Host "`n[torch] HF GPT2Model peer (eager 1t / all-t / compiled) via $py ..."
    if ($SingleCoreOnly) { Write-Host '  -SingleCoreOnly skips Wukong all-core only; the peer still runs all-thread and compiled regimes.' }
    $tBefore = Get-PowerClass
    Push-Location $repo
    $peerOutput = @(); $peerExit = $null
    try {
        $peerOutput = @(& $py tools/bench_gpt2_torch.py 512)
        $peerExit = $LASTEXITCODE
    } catch { Write-Host "  torch peer failed: $_" }
    finally { Pop-Location }
    foreach ($line in $peerOutput) { Write-Host $line }
    $peerValid = Test-PeerResult -Output $peerOutput -ExitCode $peerExit
    if (-not $peerValid) { Write-Host "  torch peer INVALID: exit=$peerExit, missing/failed/malformed correctness summary" }
    $tAfter = Get-PowerClass
    $tPower = if ((Power-Rank $tBefore.Class) -le (Power-Rank $tAfter.Class)) { $tBefore } else { $tAfter }
    Write-Host ("  torch peer power: {0} — {1}" -f $tPower.Class, $tPower.Detail)
}

Write-Host ""
if ($argmaxOk) { Write-Host "correctness: every Wukong repetition has argmax=338, valid timing, and fresh last-row output  OK" }
else { Write-Host "correctness: vehicle execution/output validation FAILED ($($argmaxes -join ',')) — TIMING INVALID until fixed" }

# A vs-torch claim needs BOTH sides clean. An unmeasured peer (-SkipTorch) is not a clean peer:
# with no peer number at all there is no comparison to stamp, so it cannot be reportable either.
$peerOk1c  = $peerValid -and ($null -ne $tPower) -and (Tag-1c  $tPower) -eq 'reportable'
$peerOkAll = $peerValid -and ($null -ne $tPower) -and (Tag-all $tPower) -eq 'reportable'
$wukOk1c  = $argmaxOk -and $sc1cSkew -le 0.05 -and (Tag-1c $s1.Power) -eq 'reportable' -and (Tag-1c $p1.Power) -eq 'reportable'
$wukOkAll = -not $SingleCoreOnly -and $argmaxOk -and (Tag-all $pAll.Power) -eq 'reportable'
$scReportable = $wukOk1c -and $peerOk1c
$allReportable = $wukOkAll -and $peerOkAll
Write-Host ""
if ($scReportable) { Write-Host "-> SINGLE-CORE: REPORTABLE (script correctness/power checks passed, 1c self-consistent). Controlled measurement eligibility still requires coordinator review." }
elseif ($SkipTorch) { Write-Host "-> SINGLE-CORE: no peer measured (-SkipTorch) — Wukong 1c timings only. NOT a vs-torch claim." }
elseif (-not $peerValid) { Write-Host "-> SINGLE-CORE: NOT reportable — torch peer failed correctness or execution validation." }
elseif ($wukOk1c) { Write-Host "-> SINGLE-CORE: NOT reportable — the Wukong side was clean but the torch peer ran under $($tPower.Class). A peer timed under DVFS manufactures a Wukong win; re-run the peer on stable AC." }
else { Write-Host "-> SINGLE-CORE: NOT reportable (battery or >5% 1c skew). Re-run on stable AC." }
if ($SingleCoreOnly) { Write-Host "-> ALL-CORE: skipped (-SingleCoreOnly). Run the full sweep on AC+full for the all-core claim." }
elseif ($allReportable) { Write-Host "-> ALL-CORE: REPORTABLE (script correctness/power checks passed across Wukong and peer). Controlled measurement eligibility still requires coordinator review." }
elseif ($SkipTorch) { Write-Host "-> ALL-CORE: no peer measured (-SkipTorch) — Wukong all-core timing only. NOT a vs-torch claim." }
elseif (-not $peerValid) { Write-Host "-> ALL-CORE: NOT reportable — torch peer failed correctness or execution validation." }
elseif ($wukOkAll) { Write-Host "-> ALL-CORE: NOT reportable — the Wukong all-core run held AC+full but the torch peer ran under $($tPower.Class). DIRECTIONAL ONLY until the peer is re-timed on AC+full." }
else { Write-Host "-> ALL-CORE: NOT reportable (needs AC+full that survives the all-core draw — disable battery-care, ensure the adapter covers peak). DIRECTIONAL ONLY." }
Write-Host ""
Write-Host 'Full-logit cross-check covers the final vehicle output only; every vehicle repetition is checked for argmax and output freshness.'
if (-not $argmaxOk -or (-not $SkipTorch -and -not $peerValid)) { exit 1 }
