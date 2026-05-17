param(
    [string]$InputPath = "model_lab/data/qwen_skill_seed.generated.jsonl",
    [string]$OutputPath = "model_lab/data/qwen_skill_train.generated.jsonl",
    [string]$ProgressPath = "model_lab/data/qwen_skill_train.generated.progress.txt",
    [string]$Model = "gpt-5.2",
    [int]$MaxSkills = 15,
    [int]$MaxDescriptionChars = 0,
    [int]$StartIndex = -1,
    [int]$EndIndex = -1,
    [int]$StartDelaySeconds = 0,
    [int]$MaxRetries = 5
)

$ErrorActionPreference = "Stop"

function Get-DotEnvValue {
    param([string]$Name)
    if (-not (Test-Path ".env")) {
        return $null
    }
    $line = Get-Content ".env" | Where-Object { $_ -match "^$Name=.+" } | Select-Object -First 1
    if (-not $line) {
        return $null
    }
    return ($line -replace "^$Name=", "").Trim()
}

function ConvertTo-OpenAITrainingSchema {
    return @{
        type = "object"
        additionalProperties = $false
        required = @("top_skills", "core_responsibilities", "nice_to_have", "summary")
        properties = @{
            top_skills = @{
                type = "array"
                items = @{
                    type = "object"
                    additionalProperties = $false
                    required = @("name", "category", "importance", "evidence")
                    properties = @{
                        name = @{ type = "string" }
                        category = @{
                            type = "string"
                            enum = @(
                                "language",
                                "framework",
                                "tool",
                                "platform",
                                "domain",
                                "soft_skill",
                                "other"
                            )
                        }
                        importance = @{ type = "number"; minimum = 0; maximum = 1 }
                        evidence = @{ type = "array"; items = @{ type = "string" } }
                    }
                }
            }
            core_responsibilities = @{ type = "array"; items = @{ type = "string" } }
            nice_to_have = @{ type = "array"; items = @{ type = "string" } }
            summary = @{ type = "string" }
        }
    }
}

function Get-ErrorDetail {
    param([object]$ErrorRecord)
    try {
        $response = $ErrorRecord.Exception.Response
        if (-not $response) {
            return $ErrorRecord.Exception.Message
        }
        $stream = $response.GetResponseStream()
        if (-not $stream) {
            return $ErrorRecord.Exception.Message
        }
        $reader = New-Object System.IO.StreamReader($stream)
        $body = $reader.ReadToEnd()
        if ($body) {
            return "$($ErrorRecord.Exception.Message) BODY=$body"
        }
    }
    catch {
    }
    return $ErrorRecord.Exception.Message
}

function ConvertTo-CleanPromptText {
    param([string]$Text)
    if ($null -eq $Text) {
        return ""
    }
    $clean = $Text -replace "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " "
    return $clean.Normalize([Text.NormalizationForm]::FormC)
}

$openaiKey = Get-DotEnvValue "OPENAI_API_KEY"
if (-not $openaiKey) {
    throw "OPENAI_API_KEY must be set in .env."
}

if ($StartDelaySeconds -gt 0) {
    Start-Sleep -Seconds $StartDelaySeconds
}

New-Item -ItemType Directory -Force -Path (Split-Path $OutputPath) | Out-Null
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType File -Path $OutputPath | Out-Null
}
if (-not (Test-Path $ProgressPath)) {
    New-Item -ItemType File -Path $ProgressPath | Out-Null
}

$existing = (Get-Content $OutputPath | Where-Object { $_.Trim() }).Count
$items = New-Object System.Collections.Generic.List[object]
foreach ($line in Get-Content $InputPath) {
    if (-not $line.Trim()) {
        continue
    }
    $row = $line | ConvertFrom-Json
    foreach ($posting in @($row.postings)) {
        $items.Add([ordered]@{
            job_title = [string]$row.job_title
            posting = $posting
        })
    }
}

$system = @"
You create supervised fine-tuning labels for a Qwen job-skill extractor.

Return only valid JSON with top_skills, core_responsibilities, nice_to_have, and summary.

Rules:
- top_skills must be concrete learnable tools, platforms, frameworks, languages, methods, and role-specific domains.
- Do not include generic labels such as AI, machine learning, cloud platforms, APIs, collaboration, communication, project management, or software development unless the posting names a concrete subskill.
- Evidence must be copied from the provided posting and kept short.
- Deduplicate near-synonyms into one canonical skill name.
- If a skill is not supported by the posting, omit it.
"@

$schema = ConvertTo-OpenAITrainingSchema
$total = $items.Count
$firstIndex = $existing
if ($StartIndex -ge 0) {
    $firstIndex = $StartIndex
}
$lastIndexExclusive = $total
if ($EndIndex -ge 0 -and $EndIndex -lt $lastIndexExclusive) {
    $lastIndexExclusive = $EndIndex
}
Add-Content -Path $ProgressPath -Value "Started $(Get-Date -Format o); total=$total; range=$firstIndex..$($lastIndexExclusive - 1); existing=$existing; model=$Model" -Encoding UTF8

for ($index = $firstIndex; $index -lt $lastIndexExclusive; $index++) {
    $item = $items[$index]
    $posting = $item.posting
    $description = ConvertTo-CleanPromptText ([string]$posting.description)
    if ($MaxDescriptionChars -gt 0 -and $description.Length -gt $MaxDescriptionChars) {
        $description = $description.Substring(0, $MaxDescriptionChars)
    }
    $title = ConvertTo-CleanPromptText ([string]$posting.title)
    $company = ConvertTo-CleanPromptText ([string]$posting.company)
    $location = ConvertTo-CleanPromptText ([string]$posting.location)
    $user = @"
Target role: $($item.job_title)
Maximum top_skills: $MaxSkills

Posting:

[1] $title
Company: $company
Location: $location
$description
"@

    $body = @{
        model = $Model
        messages = @(
            @{ role = "system"; content = $system },
            @{ role = "user"; content = $user }
        )
        response_format = @{
            type = "json_schema"
            json_schema = @{
                name = "qwen_training_output"
                strict = $true
                schema = $schema
            }
        }
    } | ConvertTo-Json -Depth 40
    $bodyBytes = [Text.Encoding]::UTF8.GetBytes($body)

    $attempt = 0
    $succeeded = $false
    while ($true) {
        try {
            $response = Invoke-RestMethod `
                -Uri "https://api.openai.com/v1/chat/completions" `
                -Method Post `
                -Headers @{ Authorization = "Bearer $openaiKey"; "Content-Type" = "application/json; charset=utf-8" } `
                -Body $bodyBytes `
                -TimeoutSec 240
            $content = $response.choices[0].message.content
            $output = $content | ConvertFrom-Json
            $trainRow = [ordered]@{
                job_title = $item.job_title
                postings = @($posting)
                output = $output
            }
            Add-Content `
                -Path $OutputPath `
                -Value ($trainRow | ConvertTo-Json -Compress -Depth 40) `
                -Encoding UTF8
            $done = $index + 1
            $message = "Wrote $done/${total}: $($item.job_title) - $($posting.title)"
            Write-Output $message
            Add-Content -Path $ProgressPath -Value "$message at $(Get-Date -Format o)" -Encoding UTF8
            $succeeded = $true
            break
        }
        catch {
            $attempt += 1
            $detail = Get-ErrorDetail $_
            if ($attempt -gt $MaxRetries) {
                $message = "FAILED $($index + 1)/${total}: $($item.job_title) - $detail"
                Write-Output $message
                Add-Content -Path $ProgressPath -Value "$message at $(Get-Date -Format o)" -Encoding UTF8
                break
            }
            $sleep = [Math]::Min(120, [Math]::Pow(2, $attempt) * 5)
            $message = "Retry $attempt/${MaxRetries} after ${sleep}s for $($index + 1)/${total}: $detail"
            Write-Output $message
            Add-Content -Path $ProgressPath -Value "$message at $(Get-Date -Format o)" -Encoding UTF8
            Start-Sleep -Seconds $sleep
        }
    }
    if (-not $succeeded) {
        continue
    }
}

Add-Content -Path $ProgressPath -Value "Finished $(Get-Date -Format o)" -Encoding UTF8
