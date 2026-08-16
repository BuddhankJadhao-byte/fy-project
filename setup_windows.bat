@echo off
setlocal
cd /d "%~dp0"
echo Creating Python virtual environment...
py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv
if errorlevel 1 goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Setup completed successfully.
echo Next, run run_full_pipeline.bat
pause
exit /b 0
:error
echo.
echo Setup failed. Install Python 3.10-3.12, select "Add Python to PATH", and try again.
pause
exit /b 1
