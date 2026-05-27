@echo off
title TCC Backend - API (porta 8000)
cd /d C:\Projectos\TCC\backend

echo [BACKEND] A verificar porta 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [BACKEND] Porta 8000 ocupada pelo PID %%a. A encerrar...
    taskkill /pid %%a /f >nul 2>&1
    timeout /t 2 /nobreak >nul
)

echo [BACKEND] A iniciar servidor (uvicorn)...
C:\Projectos\TCC\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo [BACKEND] Servidor terminou.
pause
