[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\certs"),
    [int]$Days = 825,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-DerLength {
    param(
        [System.IO.Stream]$Stream,
        [int]$Length
    )

    if ($Length -lt 128) {
        [void]$Stream.WriteByte([byte]$Length)
        return
    }

    $bytes = [BitConverter]::GetBytes([uint32]$Length)
    [Array]::Reverse($bytes)
    $offset = 0
    while ($offset -lt ($bytes.Length - 1) -and $bytes[$offset] -eq 0) {
        $offset++
    }
    $count = $bytes.Length - $offset
    [void]$Stream.WriteByte([byte](0x80 -bor $count))
    $Stream.Write($bytes, $offset, $count)
}

function Write-DerInteger {
    param(
        [System.IO.Stream]$Stream,
        [byte[]]$Value
    )

    $offset = 0
    while ($offset -lt ($Value.Length - 1) -and $Value[$offset] -eq 0) {
        $offset++
    }
    $length = $Value.Length - $offset
    $needsLeadingZero = (($Value[$offset] -band 0x80) -ne 0)
    if ($needsLeadingZero) {
        $length++
    }

    [void]$Stream.WriteByte(0x02)
    Write-DerLength -Stream $Stream -Length $length
    if ($needsLeadingZero) {
        [void]$Stream.WriteByte(0)
    }
    $Stream.Write($Value, $offset, ($Value.Length - $offset))
}

function New-RsaPrivateKeyDer {
    param(
        [System.Security.Cryptography.RSAParameters]$Parameters
    )

    $bodyStream = New-Object System.IO.MemoryStream
    try {
        Write-DerInteger -Stream $bodyStream -Value ([byte[]]@([byte]0))
        Write-DerInteger -Stream $bodyStream -Value $Parameters.Modulus
        Write-DerInteger -Stream $bodyStream -Value $Parameters.Exponent
        Write-DerInteger -Stream $bodyStream -Value $Parameters.D
        Write-DerInteger -Stream $bodyStream -Value $Parameters.P
        Write-DerInteger -Stream $bodyStream -Value $Parameters.Q
        Write-DerInteger -Stream $bodyStream -Value $Parameters.DP
        Write-DerInteger -Stream $bodyStream -Value $Parameters.DQ
        Write-DerInteger -Stream $bodyStream -Value $Parameters.InverseQ
        $body = $bodyStream.ToArray()
    }
    finally {
        $bodyStream.Dispose()
    }

    $resultStream = New-Object System.IO.MemoryStream
    try {
        [void]$resultStream.WriteByte(0x30)
        Write-DerLength -Stream $resultStream -Length $body.Length
        $resultStream.Write($body, 0, $body.Length)
        return $resultStream.ToArray()
    }
    finally {
        $resultStream.Dispose()
    }
}

function ConvertTo-Pem {
    param(
        [byte[]]$Bytes,
        [string]$Label
    )

    $base64 = [Convert]::ToBase64String($Bytes)
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append("-----BEGIN $Label-----`r`n")
    for ($offset = 0; $offset -lt $base64.Length; $offset += 64) {
        $count = [Math]::Min(64, $base64.Length - $offset)
        [void]$builder.Append($base64.Substring($offset, $count))
        [void]$builder.Append("`r`n")
    }
    [void]$builder.Append("-----END $Label-----`r`n")
    return $builder.ToString()
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

$san = "2.5.29.17={text}IPAddress=172.16.10.166&IPAddress=fc00::10:872e:e311:780f:f456"
$certificate = New-SelfSignedCertificate `
    -Subject "CN=172.16.10.166" `
    -TextExtension @($san) `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -KeyAlgorithm RSA `
    -KeyLength 2048 `
    -HashAlgorithm SHA256 `
    -KeyExportPolicy Exportable `
    -Provider "Microsoft Enhanced RSA and AES Cryptographic Provider" `
    -NotAfter (Get-Date).AddDays($Days)

try {
    $rsa = $certificate.PrivateKey
    if ($null -eq $rsa) {
        throw "Windows did not expose the generated RSA private key."
    }
    $parameters = $rsa.ExportParameters($true)
    $certificatePem = ConvertTo-Pem -Bytes $certificate.RawData -Label "CERTIFICATE"
    $privateKeyPem = ConvertTo-Pem -Bytes (New-RsaPrivateKeyDer -Parameters $parameters) -Label "RSA PRIVATE KEY"
    [IO.File]::WriteAllText($certificatePath, $certificatePem, [Text.Encoding]::ASCII)
    [IO.File]::WriteAllText($privateKeyPath, $privateKeyPem, [Text.Encoding]::ASCII)
}
finally {
    Remove-Item -LiteralPath ("Cert:\CurrentUser\My\" + $certificate.Thumbprint) -Force -ErrorAction SilentlyContinue
}

Write-Host "Server certificate: $certificatePath"
Write-Host "Server private key: $privateKeyPath"
Write-Host "This is a server self-signed certificate; clients do not need a client certificate."
