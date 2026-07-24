[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$requiredMarkers = @(
    (Join-Path $projectRoot '.git'),
    (Join-Path $projectRoot 'pyproject.toml'),
    (Join-Path $projectRoot 'bas')
)

if (@($requiredMarkers | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -gt 0) {
    throw "Cleanup refused: $projectRoot is not the expected 2606BAS project root."
}

$protectedRoots = @(
    (Join-Path $projectRoot '.git'),
    (Join-Path $projectRoot '.venv')
)

function Test-IsProtectedPath {
    param([Parameter(Mandatory)][string]$Path)

    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    foreach ($protectedRoot in $protectedRoots) {
        if (
            $resolvedPath.Equals($protectedRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
            $resolvedPath.StartsWith(
                $protectedRoot + [System.IO.Path]::DirectorySeparatorChar,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        ) {
            return $true
        }
    }
    return $false
}

$targets = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)

foreach ($name in @('.pytest_cache', '.tmp', 'tmp', 'pytest_cache_local', 'logs')) {
    $path = Join-Path $projectRoot $name
    if (Test-Path -LiteralPath $path) {
        [void]$targets.Add($path)
    }
}

foreach ($directory in Get-ChildItem -LiteralPath $projectRoot -Directory -Force) {
    if ($directory.Name -like 'pytest_tmp*' -or $directory.Name -like 'python_tmp*') {
        [void]$targets.Add($directory.FullName)
    }
}

foreach ($directory in Get-ChildItem -LiteralPath $projectRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue) {
    if (
        -not (Test-IsProtectedPath -Path $directory.FullName) -and
        $directory.Name -in @('__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache')
    ) {
        [void]$targets.Add($directory.FullName)
    }
}

$orderedTargets = @($targets) | Sort-Object Length -Descending
$removedCount = 0
$failedTargets = @()
foreach ($target in $orderedTargets) {
    $fullTarget = [System.IO.Path]::GetFullPath($target)
    if (
        $fullTarget.Equals($projectRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $fullTarget.StartsWith(
            $projectRoot + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or
        (Test-IsProtectedPath -Path $fullTarget)
    ) {
        throw "Cleanup refused for an out-of-scope or protected path: $fullTarget"
    }

    if ($PSCmdlet.ShouldProcess($fullTarget, 'Remove regenerable temporary directory')) {
        try {
            Remove-Item -LiteralPath $fullTarget -Recurse -Force
            $removedCount++
        }
        catch {
            $failedTargets += $fullTarget
            Write-Warning "Could not remove $fullTarget`: $($_.Exception.Message)"
        }
    }
}

Write-Host "Workspace cleanup complete. Removed $removedCount temporary directories."
if ($failedTargets.Count -gt 0) {
    Write-Warning "The following paths require manual permission handling:"
    $failedTargets | ForEach-Object { Write-Warning "  $_" }
    exit 1
}
