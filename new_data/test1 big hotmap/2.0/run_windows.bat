@echo off
setlocal
set "WORKERS=%~1"
if "%WORKERS%"=="" set "WORKERS=10"

echo ============================================================
echo  Stage-1 global GRU search (30-step horizon)
echo  WORKERS=%WORKERS%
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0code\run_stage1_global_search.ps1" -Workers %WORKERS%
set "RC=%ERRORLEVEL%"

echo.
echo Exit code: %RC%
pause
endlocal
