Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        py -3.14 -m venv (Join-Path $PSScriptRoot '.venv')
    }

    & $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости backend' }

    $configuration = Join-Path $PSScriptRoot '.env'
    if (-not (Test-Path -LiteralPath $configuration)) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot '.env.example') -Destination $configuration
    }

    & $python -m uvicorn app.main:app --reload
}
finally {
    Pop-Location
}
