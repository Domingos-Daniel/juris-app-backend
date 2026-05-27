@echo off
echo === Limpando TUDO ===
:: Mata Python
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul
:: Limpa cache
for /d /r "C:\Projectos\TCC\backend" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Cache limpo.

echo === Backend ===
cd /d C:\Projectos\TCC\backend
title TCC Backend (porta 8000)
C:\Projectos\TCC\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --log-level info
pause
