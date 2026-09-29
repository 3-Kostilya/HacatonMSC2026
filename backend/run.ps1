if (-not (Test-Path .\.venv)) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

if (-not (Test-Path .\.env)) {
    Copy-Item .\.env.example .\.env
}

uvicorn app.main:app --reload