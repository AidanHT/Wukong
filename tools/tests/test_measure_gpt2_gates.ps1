# Isolated gate tests: AST-load functions, synthetic files, fake vehicle, fake peer.
$ErrorActionPreference = 'Stop'
$sourcePath = Join-Path (Split-Path $PSScriptRoot -Parent) 'measure_gpt2.ps1'
$source = Get-Content -Raw -LiteralPath $sourcePath
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors -join "`n") }
foreach ($function in $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    . ([scriptblock]::Create($function.Extent.Text))
}
$script:assertions = 0
function Assert($Condition, [string]$Message) {
    $script:assertions++
    if (-not $Condition) { throw "FAIL: $Message" }
}
function Get-PowerClass { [pscustomobject]@{ Class='REPORTABLE'; Detail='synthetic test power'; Reportable=$true } }

$good = @('338', '1000', '8')
Assert (Convert-VehicleResult $good 0).Valid 'valid vehicle'
foreach ($bad in @(@('337','1000','8'), @('338','0','8'), @('338','-1','8'), @('338','NaN','8'),
    @('338','1000','7'), @('338','1000'), @('338','1000','8','9'), @('noise','338','1000','8'),
    @('338','999999999999999999999999','8'))) {
    Assert (-not (Convert-VehicleResult $bad 0).Valid) "invalid output $bad"
}
Assert (-not (Convert-VehicleResult $good 9).Valid) 'nonzero exit with valid stdout'
Assert (-not (Convert-VehicleResult $good $null).Valid) 'missing exit'

$summary = @{ schema=1; status='ok'; sequence_length=512; vocab=50257; argmax=338; relative_error=0.001;
    results=@{ 'eager 1-thread'=1.0; 'eager all-thread'=2.0; 'compiled all-thread'=$null } }
function Peer-Line($Value) { 'GPT2_PEER_RESULT ' + ($Value | ConvertTo-Json -Compress -Depth 5) }
$line = Peer-Line $summary
Assert (Test-PeerResult @($line) 0) 'valid peer, existing tolerance boundary'
Assert (-not (Test-PeerResult @($line) 3)) 'failed peer process'
Assert (-not (Test-PeerResult @($line, $line) 0)) 'duplicate peer summary'
Assert (-not (Test-PeerResult @('GPT2_PEER_RESULT {') 0)) 'malformed peer summary'
Assert (-not (Test-PeerResult @('normal output only') 0)) 'missing peer summary'
Assert (-not (Test-PeerResult @('GPT2_PEER_RESULT [' + $line.Substring('GPT2_PEER_RESULT '.Length) + ']') 0)) 'array summary rejected'
$originalResults = $summary.results
$summary.results = @($originalResults)
Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) 'array results rejected'
$summary.results = $originalResults
$summary.status = @('ok')
Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) 'array status rejected'
$summary.status = 'ok'
foreach ($name in @('schema', 'sequence_length', 'vocab', 'argmax')) {
    $original = $summary[$name]
    foreach ($bad in @("$original", $true)) {
        $summary[$name] = $bad
        Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) "identity $name cannot be coerced"
    }
    $summary[$name] = $original
}
$summary.results.'unexpected variant' = 1
Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) 'unknown peer variants rejected'
$summary.results.Remove('unexpected variant')
foreach ($bad in @($null, 0, -1, 'NaN', 'Infinity', '1', $true)) {
    $summary.results.'eager 1-thread' = $bad
    Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) "bad peer timing: $bad"
}
$summary.results.'eager 1-thread' = 1
$summary.relative_error = 0.001001
Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) 'diverging peer'
$summary.relative_error = 0
$summary.argmax = 337
Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) 'wrong peer argmax'
$summary.argmax = 338
$summary.results.Remove('compiled all-thread')
Assert (-not (Test-PeerResult @((Peer-Line $summary)) 0)) 'incomplete peer regimes'

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('gpt2-gate-test-' + [guid]::NewGuid().ToString('N'))
$savedThreads = [Environment]::GetEnvironmentVariable('RAYON_NUM_THREADS', 'Process')
$savedMode = [Environment]::GetEnvironmentVariable('GPT2_TEST_MODE', 'Process')
$startLocation = Get-Location
try {
    New-Item -ItemType Directory -Path (Join-Path $tempRoot 'data/gpt2') -Force | Out-Null
    $repo = $tempRoot
    $exe = Join-Path $tempRoot 'fake-vehicle.ps1'
    @'
$global:Gpt2GateFakeCalls++
if ($env:GPT2_TEST_MODE -eq 'throw') { throw 'synthetic invocation failure' }
if ($env:GPT2_TEST_MODE -ne 'stale') {
    [IO.File]::WriteAllBytes((Join-Path (Get-Location) 'data/gpt2/gpt2_bench_lastrow.bin'), [byte[]]::new(50257 * 4))
}
if ($env:GPT2_TEST_MODE -eq 'allcore' -and $null -ne $env:RAYON_NUM_THREADS) { throw 'all-core inherited thread cap' }
if ($env:GPT2_TEST_MODE -eq 'malformed-first' -and $global:Gpt2GateFakeCalls -eq 1) { 'unexpected stdout' }
if ($env:GPT2_TEST_MODE -eq 'bad-first' -and $global:Gpt2GateFakeCalls -eq 1) { '337' } else { '338' }
'1000'
if ($env:GPT2_TEST_MODE -ne 'partial-first' -or $global:Gpt2GateFakeCalls -ne 1) { '8' }
if ($env:GPT2_TEST_MODE -eq 'exit-first' -and $global:Gpt2GateFakeCalls -eq 1) {
    [Console]::Error.WriteLine('synthetic vehicle stderr retained')
    $global:LASTEXITCODE = 7
} else { $global:LASTEXITCODE = 0 }
'@ | Set-Content -LiteralPath $exe
    $env:RAYON_NUM_THREADS = 'caller-value'
    foreach ($mode in @('good', 'bad-first', 'exit-first', 'malformed-first', 'partial-first', 'stale', 'throw', 'allcore')) {
        $env:GPT2_TEST_MODE = $mode
        $global:Gpt2GateFakeCalls = 0
        $vars = if ($mode -eq 'allcore') { @{} } else { @{ RAYON_NUM_THREADS='1' } }
        $result = Measure-Vehicle 'unused.wk' $vars 2
        $expected = $mode -in @('good', 'allcore')
        Assert ($result.Valid -eq $expected) "vehicle mode $mode"
        Assert ($result.Repetitions.Count -eq 2) "retained repetitions $mode"
        Assert ($env:RAYON_NUM_THREADS -eq 'caller-value') "caller environment restored $mode"
        Assert ((Get-Location).Path -eq $startLocation.Path) "caller location restored $mode"
        if (-not $expected) { Assert ([double]::IsPositiveInfinity($result.Ms)) "invalid run cannot retain best timing $mode" }
    }
    Remove-Item -LiteralPath Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue
    $env:GPT2_TEST_MODE = 'good'
    $result = Measure-Vehicle 'unused.wk' @{ RAYON_NUM_THREADS='1' } 1
    Assert ($result.Valid -and $null -eq $env:RAYON_NUM_THREADS) 'absent caller environment restored'
    [Environment]::SetEnvironmentVariable('RAYON_NUM_THREADS', '', 'Process')
    $result = Measure-Vehicle 'unused.wk' @{ RAYON_NUM_THREADS='1' } 1
    Assert ($result.Valid -and (Test-Path Env:RAYON_NUM_THREADS) -and $env:RAYON_NUM_THREADS -eq '') 'empty caller environment restored'
    # Execute the real top-level gates in a child shell with only external effects
    # replaced. No real compiler, model, power query, or build is invoked.
    $toolsDir = Join-Path $tempRoot 'tools'
    New-Item -ItemType Directory -Path $toolsDir | Out-Null
    $driver = Join-Path $toolsDir 'measure.ps1'
    $peerPath = Join-Path $toolsDir 'fake-peer.ps1'
    @'
if ($env:GPT2_TEST_MODE -eq 'peer-fail') {
    [Console]::Error.WriteLine('synthetic peer stderr retained')
    'GPT2_PEER_RESULT {"schema":1,"status":"invalid"}'
    $global:LASTEXITCODE = 19
} else {
    'GPT2_PEER_RESULT {"schema":1,"status":"ok","sequence_length":512,"vocab":50257,"argmax":338,"relative_error":0,"results":{"eager 1-thread":1,"eager all-thread":1,"compiled all-thread":null}}'
    $global:LASTEXITCODE = 0
}
'@ | Set-Content -LiteralPath $peerPath
    $substitute = @'
function Get-PowerClass { [pscustomobject]@{ Class='REPORTABLE'; Detail='synthetic'; Reportable=$true } }
function cargo { 'synthetic build stdout'; [Console]::Error.WriteLine('synthetic build stderr'); $global:LASTEXITCODE=23 }
$exe = Join-Path $repo 'fake-vehicle.ps1'
'@
    $source.Replace('Add-Type -AssemblyName System.Windows.Forms', $substitute) | Set-Content -LiteralPath $driver
    foreach ($mode in @('good', 'peer-fail', 'build-fail')) {
        $env:GPT2_TEST_MODE = $mode
        $argsList = @('-NoProfile', '-File', $driver, '-Outer', '1', '-TorchPython', $peerPath)
        if ($mode -ne 'build-fail') { $argsList += '-SkipBuild' }
        $output = @(& (Join-Path $PSHOME 'pwsh.exe') @argsList)
        $code = $LASTEXITCODE
        foreach ($captured in $output) { Write-Host "[$mode] $captured" }
        Write-Host "[$mode] exit=$code"
        $reportable = @($output | Where-Object { $_ -match '^-> .*: REPORTABLE' }).Count
        if ($mode -eq 'good') {
            Assert ($code -eq 0 -and $reportable -eq 2) 'valid full script is reportable'
        } else {
            Assert ($code -ne 0 -and $reportable -eq 0) "$mode cannot be reportable"
        }
        if ($mode -eq 'build-fail') {
            Assert (@($output | Where-Object { $_ -match '^\[wukong\]' }).Count -eq 0) 'failed build stops before vehicle'
        }
    }
} finally {
    if ($null -eq $savedThreads) { Remove-Item -LiteralPath Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue }
    else { [Environment]::SetEnvironmentVariable('RAYON_NUM_THREADS', $savedThreads, 'Process') }
    if ($null -eq $savedMode) { Remove-Item -LiteralPath Env:GPT2_TEST_MODE -ErrorAction SilentlyContinue }
    else { [Environment]::SetEnvironmentVariable('GPT2_TEST_MODE', $savedMode, 'Process') }
    Set-Location $startLocation
    $resolved = [IO.Path]::GetFullPath($tempRoot)
    $parent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if (-not $resolved.StartsWith($parent, [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path $resolved -Leaf) -notlike 'gpt2-gate-test-*') { throw 'unsafe test cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
Write-Host "PASS: $script:assertions GPT-2 PowerShell gate assertions (synthetic only)"
