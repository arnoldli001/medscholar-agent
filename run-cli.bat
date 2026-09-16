@echo off
rem ===========================================================================
rem  MedScholar Agent - command line menu.
rem  PURE ASCII on purpose: see the note in run.bat.
rem ===========================================================================

setlocal
cd /d "%~dp0"

set "PY="
if exist ".python\python.exe" set "PY=.python\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"

if not "%PY%"=="python" goto run
where python >nul 2>&1
if not errorlevel 1 goto run
echo.
echo   [ERROR] No Python interpreter found. See README.md.
echo.
pause
exit /b 1

:run
"%PY%" -X utf8 "scripts\bootstrap.py" --cli %*
if errorlevel 1 pause
endlocal
