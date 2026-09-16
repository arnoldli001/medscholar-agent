@echo off
rem ===========================================================================
rem  MedScholar Agent launcher.
rem
rem  IMPORTANT: this file is intentionally PURE ASCII.
rem  cmd.exe parses .bat files using the console OEM codepage (936/GBK on
rem  Chinese Windows). Non-ASCII bytes in a batch file can swallow the
rem  following CR/LF and corrupt the next command line, producing bogus
rem  errors like '"dp0" is not recognized as an internal or external command'.
rem  All Chinese user-facing text is therefore printed by Python instead
rem  (see scripts/bootstrap.py), which writes Unicode via WriteConsoleW and
rem  is unaffected by the console codepage.
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
echo   [ERROR] No Python interpreter found.
echo.
echo   This package normally ships with its own runtime at:
echo       .python\python.exe
echo   but that file is missing.
echo.
echo   Fix: install Python 3.10+ from https://www.python.org/downloads/
echo        (tick "Add python.exe to PATH"), then run this file again.
echo.
pause
exit /b 1

:run
"%PY%" -X utf8 "scripts\bootstrap.py" --serve %*
set "RC=%errorlevel%"
if not "%RC%"=="0" pause
endlocal & exit /b %RC%
