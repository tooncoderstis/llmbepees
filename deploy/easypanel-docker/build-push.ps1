<#
.SYNOPSIS
    Bangun & push image llmbepees ke GHCR (publik). Deploy ke EasyPanel dilakukan via MCP.

.DESCRIPTION
    Contoh:
        $env:GHCR_OWNER = "aasa"
        .\build-push.ps1 -Tag latest

    Hasil: ghcr.io/<owner>/llmbepees-web:<tag>
#>
param(
    [string]$Owner = $env:GHCR_OWNER,
    [string]$Image = "llmbepees-web",
    [string]$Tag = "latest",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"

if (-not $Owner) {
    throw "Set GHCR_OWNER (mis. `$env:GHCR_OWNER = 'aasa') atau pakai -Owner."
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$full = "ghcr.io/$Owner/${Image}:${Tag}"

Write-Host "Build  : $full" -ForegroundColor Cyan
docker build -f (Join-Path $PSScriptRoot "Dockerfile") -t $full $root
if ($LASTEXITCODE -ne 0) { throw "docker build gagal" }

if (-not $NoPush) {
    Write-Host "Push   : $full" -ForegroundColor Cyan
    docker push $full
    if ($LASTEXITCODE -ne 0) { throw "docker push gagal" }
}

Write-Host "Selesai. Deploy via MCP EasyPanel (lihat docs/runbooks/deploy.md)." -ForegroundColor Green
