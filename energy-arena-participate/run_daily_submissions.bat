@echo off
REM Run daily Arena submissions for all currently open challenges.
REM Schedule this at 11:30 CET via Task Scheduler.
REM Put ARENA_API_KEY in local .env (same folder).
REM The default baseline source is SMARD; ENTSOE_API_KEY is optional.
REM Use --use_global_env only if you want fallback to system/user env vars.
REM Pass --data_source entsoe only if you explicitly want ENTSO-E.
REM Logs are written to .\logs\run_daily_submissions_*.log.

setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_daily_submissions.ps1" %*
exit /b %ERRORLEVEL%
