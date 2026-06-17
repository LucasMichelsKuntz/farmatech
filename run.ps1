if (-not (Test-Path "farmtech.db")) {
    Write-Host "Ingesting dataset into database..." -ForegroundColor Yellow
    .\venv\Scripts\python db\ingest.py
}

Write-Host "Starting dashboard at http://localhost:8501" -ForegroundColor Green
.\venv\Scripts\streamlit run dashboard\app.py
