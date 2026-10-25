@echo off
setlocal enabledelayedexpansion

title Floor Plan ^& Maze Route Solver - ADSA

echo =====================================================================
echo       Floor Plan ^& Maze Route Solver (ADSA Course Project)
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Check for Python installation
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.9+ from https://www.python.org/
    echo and ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

:: 2. Setup or detect virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating local virtual environment in .venv...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Could not create virtual environment. Falling back to global python.
        set "PY_CMD=python"
    ) else (
        set "PY_CMD=.venv\Scripts\python.exe"
    )
) else (
    set "PY_CMD=.venv\Scripts\python.exe"
)

:: 3. Install required packages
echo [SETUP] Verifying dependencies (Pillow, NumPy)...
%PY_CMD% -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Pip install had a notice, attempting to continue...
)

:: 4. Run automated test suite
echo [TESTS] Running automated verification tests...
%PY_CMD% -m unittest discover -s tests -v
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] One or more tests reported an issue.
    echo Press any key to continue launching the application anyway...
    pause >nul
) else (
    echo [TESTS] All automated tests passed successfully!
)

:: 5. Launch the desktop GUI application
echo.
echo [LAUNCH] Starting Floor Plan ^& Maze Route Solver GUI...
echo (You can close this command window after you finish using the application.)
echo.
%PY_CMD% app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The application closed with an exit code: %ERRORLEVEL%
    pause
)

endlocal
