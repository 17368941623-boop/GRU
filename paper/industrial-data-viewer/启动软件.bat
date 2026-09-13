@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "Industrial Vision Server" /min cmd /c "npm run dev"
timeout /t 3 /nobreak >nul
start "" "http://localhost:3000"
