<#
.SYNOPSIS
  Same-machine A/B of two wukong_codegen_gpu test binaries, with a self-control that measures the
  run's own noise floor.

.DESCRIPTION
  The GPU perf-identity leg of GPU_RETARGET_PLAN.md Phase 2 ("on the 4050, before/after this phase
  must be behaviorally identical ... same-run perf A/B ties", plan line 614). Two arms cannot live
  in one process, so "same-run adjacent A/B" becomes a tight per-bench alternation between separate
  binaries, with three protections that the C(twin) episode proved are load-bearing:

    1. A CONTROL ARM. Arm `C` is the *same binary as arm `A`*, invoked a second time. Its expected
       ratio against A is exactly 1.00 and it contains no code difference, so C/A measures what this
       particular run can resolve. A B/A ratio inside the C/A spread is a TIE, not a win or a loss.
       Do not replace this with a hardcoded threshold -- the floor is a property of the run.
       (An unpinned round once read a byte-identical binary as "1.31x faster"; see the
       pin-the-timing-thread note.)
    2. ROTATING ARM ORDER. Round r starts at arm r mod 3, so no arm sits permanently in the
       first slot. GPU clocks ramp on first launch, and a fixed order silently pays that tax to
       whichever arm goes first.
    3. A DISCARD ROUND. Round 0 runs everything and is thrown away, so the cubin and autotune
       caches are warm for BOTH arms before anything is recorded. The caches are device-keyed as of
       Phase 2, so the two arms write different keys and the cold JIT would otherwise land entirely
       on whichever arm ran first.

  Power state is checked before AND after and recorded in the log. On battery this script REFUSES
  to run without -Force: battery is a 2-4x slower machine and its numbers are not comparable to
  AC ones. That rule is encoded here rather than left in a comment because it has been violated.

  Nothing here parses a bench's numbers. Each invocation's full stderr+stdout is written to its own
  file for reading afterwards; the benches print their own labelled columns and regimes.

.EXAMPLE
  pwsh tools/perf/gpu_ab.ps1 -BaseExe C:\v2bt\release\deps\wukong_codegen_gpu-<hash>.exe `
                             -NewExe  target\release\deps\wukong_codegen_gpu-<hash>.exe `
                             -Rounds 5 -Out bench/gpu/4050-identity
#>
[CmdletBinding()]
param(
    # Arm A / arm C: the BASELINE binary (both arms are this same file, on purpose).
    [Parameter(Mandatory = $true)][string]$BaseExe,
    # Arm B: the binary under test.
    [Parameter(Mandatory = $true)][string]$NewExe,
    [int]$Rounds = 5,
    [Parameter(Mandatory = $true)][string]$Out,
    # Bench names as libtest reports them (suffix match is enough; see -List).
    [string[]]$Benches = @(
        'gemm_throughput',          # ptx_gemm / ptx_wmma f32+f16 GEMM
        'tensorcore_throughput',    # ptx_wmma tensor cores
        'tensorcore_roofline_pct',  # same family, roofline framing
        'flash_throughput',         # ptx_flash attention
        'int8_swz_vs_handplaced',   # ptx_int8 swizzle
        'hbm_bandwidth',            # ptx.rs streaming; doubles as a machine-health canary
        'transformer_layer_throughput',
        'resident_model_throughput',
        'mega_vs_single_gemm',      # megakernel
        'mega_vs_single_vmath',     # megakernel + vmath
        'cubin_cache_compile_latency' # the device-keyed cubin cache itself
    ),
    # Skip the battery refusal. Only for a deliberately non-publishable smoke run.
    [switch]$Force,
    # Print the ignored-test names the binary actually exposes, then exit.
    [switch]$List
)

$ErrorActionPreference = 'Stop'

function Get-PowerState {
    $b = Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $b) { return [pscustomobject]@{ Status = 'no-battery(desktop)'; OnBattery = $false; Pct = 100 } }
    # Win32_Battery.BatteryStatus: 1 = discharging, 2 = on AC, 3 = fully charged, 6/7/8/9 = charging.
    $onBat = ($b.BatteryStatus -eq 1)
    $name = switch ($b.BatteryStatus) {
        1 { 'DISCHARGING (on battery)' } 2 { 'AC, not charging' } 3 { 'AC, fully charged' }
        6 { 'AC, charging' } 7 { 'AC, charging (high)' } 8 { 'AC, charging (low)' }
        9 { 'AC, charging (critical)' } default { "code $($b.BatteryStatus)" }
    }
    [pscustomobject]@{ Status = $name; OnBattery = $onBat; Pct = $b.EstimatedChargeRemaining }
}

function Get-GpuState {
    try {
        (nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader) -join '; '
    } catch { 'nvidia-smi unavailable' }
}

foreach ($p in @($BaseExe, $NewExe)) {
    if (-not (Test-Path $p)) { throw "arm binary not found: $p" }
}

# The arms MUST be different files, and this is easy to get wrong: cargo's metadata hash keys on
# package/version/features/profile -- NOT on source -- so the baseline and the new tree produce test
# binaries with the SAME filename (`wukong_codegen_gpu-<same hash>.exe`), differing only by target
# directory. Hand the same path to both arms and all three arms become identical: every row ties
# perfectly and the run reads as a clean identity pass while having compared nothing. Hash them.
$hBase = (Get-FileHash -Algorithm SHA256 $BaseExe).Hash
$hNew = (Get-FileHash -Algorithm SHA256 $NewExe).Hash
if ($hBase -eq $hNew) {
    throw @"
REFUSING TO RUN: -BaseExe and -NewExe are the SAME BINARY (sha256 $($hBase.Substring(0,16))...).
Every arm would be identical, so every row would tie and the run would prove nothing. Note that the
two arms legitimately share a cargo hash in their FILENAME -- check the target directory, not the
name. Base: $BaseExe
New : $NewExe
"@
}

if ($List) {
    & $BaseExe --list --ignored 2>&1 | Where-Object { $_ -match ': test$' }
    exit 0
}

# Resolve short bench names to the full libtest paths, from the binary itself rather than from a
# hardcoded module path -- the two arms must agree, and a typo would otherwise run zero tests and
# still exit 0.
$listing = & $BaseExe --list --ignored 2>&1 |
    Where-Object { $_ -match ': test$' } |          # drop the blank + "N tests, M benchmarks" summary
    ForEach-Object { ($_ -replace ': test$', '').Trim() }
$resolved = [ordered]@{}
foreach ($b in $Benches) {
    $hit = @($listing | Where-Object { $_ -eq $b -or $_.EndsWith("::$b") })
    if ($hit.Count -ne 1) { throw "bench '$b' resolved to $($hit.Count) tests in $BaseExe (want exactly 1)" }
    $resolved[$b] = $hit[0]
}

$power0 = Get-PowerState
if ($power0.OnBattery -and -not $Force) {
    throw @"
REFUSING TO RUN: the machine is on battery ($($power0.Pct)%).
Battery is a 2-4x slower machine than AC and its timings are not comparable to AC timings, so this
round could not be published. Plug in, let the charge settle, and re-run. Use -Force only for a
smoke run you will not report.
"@
}

New-Item -ItemType Directory -Force -Path $Out | Out-Null
$log = Join-Path $Out 'round.log'

# Arm C is deliberately the SAME path as arm A. That is the control.
$arms = [ordered]@{ A = $BaseExe; C = $BaseExe; B = $NewExe }
$armNames = @($arms.Keys)

function Log($m) { $m | Tee-Object -FilePath $log -Append }

Log "# GPU perf-identity A/B"
Log "started      : $(Get-Date -Format o)"
Log "arm A (base) : $BaseExe"
Log "             : sha256 $hBase"
Log "arm C (ctrl) : $BaseExe   <- same binary as A; C/A is this run's noise floor"
Log "arm B (new)  : $NewExe"
Log "             : sha256 $hNew"
Log "rounds       : $Rounds (plus a discarded warm-up round 0)"
Log "benches      : $($Benches -join ', ')"
Log "power BEFORE : $($power0.Status) $($power0.Pct)%"
Log "gpu   BEFORE : $(Get-GpuState)"
Log ""

for ($r = 0; $r -le $Rounds; $r++) {
    $discard = ($r -eq 0)
    $tag = if ($discard) { 'warm-up (DISCARDED)' } else { "round $r" }
    Log "== $tag  $(Get-Date -Format HH:mm:ss)"
    foreach ($b in $Benches) {
        # Rotate which arm leads, so the clock-ramp tax does not always fall on the same arm.
        $order = @(0, 1, 2) | ForEach-Object { $armNames[($_ + $r) % 3] }
        foreach ($arm in $order) {
            $dest = Join-Path $Out "$b.$arm.r$r.txt"
            if ($discard) { $dest = Join-Path $Out "warmup.$b.$arm.txt" }
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & $arms[$arm] $resolved[$b] --exact --ignored --nocapture *> $dest
            $ok = $LASTEXITCODE
            $sw.Stop()
            "wall_ms=$([int]$sw.Elapsed.TotalMilliseconds) exit=$ok arm=$arm round=$r" |
                Out-File -FilePath $dest -Append -Encoding utf8
            $flag = if ($ok -eq 0) { '' } else { "  *** EXIT $ok" }
            Log ("   {0,-32} {1}  {2,7} ms{3}" -f $b, $arm, [int]$sw.Elapsed.TotalMilliseconds, $flag)
        }
    }
}

$power1 = Get-PowerState
Log ""
Log "power AFTER  : $($power1.Status) $($power1.Pct)%"
Log "gpu   AFTER  : $(Get-GpuState)"
Log "finished     : $(Get-Date -Format o)"
if ($power0.Status -ne $power1.Status) {
    Log "*** POWER STATE CHANGED MID-ROUND ($($power0.Status) -> $($power1.Status)). This round is NOT publishable."
}
Log ""
Log "Read the per-invocation files in $Out. Compare B/A against the C/A control: a B/A ratio inside"
Log "the C/A spread is a TIE. Publish no size whose range does not clear that floor."
