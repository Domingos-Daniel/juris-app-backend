@echo off
cd /d C:\Projectos\TCC\backend
call .venv\Scripts\activate
python -m app.scripts.ingest_lex_ao_bulk
pause
