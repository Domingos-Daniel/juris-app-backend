@echo off
cd /d C:\Projectos\TCC\backend
title TCC Migracao e5-base (70k embeddings)
echo Migracao de embeddings: MiniLM -> multilingual-e5-base (768-d)
echo Tempo estimado: 10-15 horas
echo.
C:\Projectos\TCC\backend\.venv\Scripts\python.exe -m app.scripts.migrate_embeddings_e5
pause
