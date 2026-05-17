param(
    [string]$InputPath = "model_lab/data/qwen_skill_seed.generated.jsonl",
    [string]$OutputPath = "model_lab/data/qwen_skill_train.generated.jsonl",
    [string]$ShardDir = "model_lab/data/teacher_shards",
    [string]$Model = "gpt-5.2",
    [int]$Concurrency = 6,
    [int]$MaxSkills = 15
)

$ErrorActionPreference = "Stop"

if ($Concurrency -lt 1) {
    throw "Concurrency must be >= 1."
}

$items = New-Object System.Collections.Generic.List[object]
foreach ($line in Get-Content $InputPath) {
    if (-not $line.Trim()) {
        continue
    }
    $row = $line | ConvertFrom-Json
    foreach ($posting in @($row.postings)) {
        $items.Add($posting)
    }
}

$total = $items.Count
$existing = 0
if (Test-Path $OutputPath) {
    $existing = (Get-Content $OutputPath | Where-Object { $_.Trim() }).Count
}
if ($existing -ge $total) {
    Write-Output "Nothing to do: existing=$existing total=$total"
    exit 0
}

New-Item -ItemType Directory -Force -Path $ShardDir | Out-Null
$script = (Resolve-Path "model_lab/scripts/generate_qwen_training_data.ps1").Path
$remaining = $total - $existing
$chunkSize = [Math]::Ceiling($remaining / $Concurrency)
$started = @()

for ($worker = 0; $worker -lt $Concurrency; $worker++) {
    $start = $existing + ($worker * $chunkSize)
    if ($start -ge $total) {
        break
    }
    $end = [Math]::Min($total, $start + $chunkSize)
    $shardOutput = Join-Path $ShardDir ("qwen_skill_train.generated.{0:D2}.jsonl" -f $worker)
    $shardProgress = Join-Path $ShardDir ("qwen_skill_train.generated.{0:D2}.progress.txt" -f $worker)
    $shardLog = Join-Path $ShardDir ("qwen_skill_train.generated.{0:D2}.runner.log" -f $worker)
    $shardErr = "$shardLog.err"

    if (Test-Path $shardOutput) {
        Remove-Item -LiteralPath $shardOutput -Force
    }
    if (Test-Path $shardProgress) {
        Remove-Item -LiteralPath $shardProgress -Force
    }

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $script,
        "-InputPath",
        $InputPath,
        "-OutputPath",
        $shardOutput,
        "-ProgressPath",
        $shardProgress,
        "-Model",
        $Model,
        "-MaxSkills",
        [string]$MaxSkills,
        "-StartIndex",
        [string]$start,
        "-EndIndex",
        [string]$end
    )

    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $args `
        -WorkingDirectory (Get-Location).Path `
        -WindowStyle Hidden `
        -RedirectStandardOutput $shardLog `
        -RedirectStandardError $shardErr `
        -PassThru
    $started += [pscustomobject]@{
        Worker = $worker
        Pid = $process.Id
        StartIndex = $start
        EndIndexExclusive = $end
        Output = $shardOutput
        Progress = $shardProgress
    }
}

$started | Format-Table -AutoSize
Write-Output "Started $($started.Count) worker(s). Existing main rows: $existing. Total target rows: $total."
Write-Output "When all workers finish, merge in worker order: Get-Content '$ShardDir\\*.jsonl' | Add-Content '$OutputPath'"
