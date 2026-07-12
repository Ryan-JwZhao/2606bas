[CmdletBinding()]
param(
    [string]$OpenSslPath = "",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\certs"),
    [int]$Days = 825,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OpenSslPath)) {
    $openssl = Get-Command openssl.exe -ErrorAction SilentlyContinue
    if ($null -eq $openssl) {
        throw "openssl.exe was not found on the server. Install or copy OpenSSL, or pass -OpenSslPath."
    }
    $OpenSslPath = $openssl.Source
}

if (-not (Test-Path -LiteralPath $OpenSslPath -PathType Leaf)) {
    throw "OpenSSL path does not exist: $OpenSslPath"
}

$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$certificatePath = Join-Path $OutputDirectory "bas-lan.crt"
$privateKeyPath = Join-Path $OutputDirectory "bas-lan.key"

if ((Test-Path -LiteralPath $certificatePath) -or (Test-Path -LiteralPath $privateKeyPath)) {
    if (-not $Force) {
        throw "Certificate files already exist. Pass -Force only when replacement is intended."
    }
    Remove-Item -LiteralPath $certificatePath, $privateKeyPath -Force
}

# Include both addresses in the certificate SAN. Nginx listens on IPv4 by default;
# the same certificate can be reused if the IPv6 listener is enabled later.
$subjectAltName = "subjectAltName=IP:10.1.5.175,IP:fc00::10:872e:e311:780f:f456"
$arguments = @(
    "req",
    "-x509",
    "-nodes",
    "-newkey", "rsa:2048",
    "-sha256",
    "-days", "$Days",
    "-keyout", $privateKeyPath,
    "-out", $certificatePath,
    "-subj", "/CN=10.1.5.175",
    "-addext", $subjectAltName
)

& $OpenSslPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "OpenSSL certificate generation failed with exit code $LASTEXITCODE."
}

Write-Host "Server certificate: $certificatePath"
Write-Host "Server private key: $privateKeyPath"
Write-Host "This is a server self-signed certificate. Clients do not need a client certificate, but browsers may show a trust warning."
