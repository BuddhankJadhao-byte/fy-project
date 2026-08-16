@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python scripts\run_pipeline.py
if errorlevel 1 (
  echo Pipeline failed. Review the error above.
  pause
  exit /b 1
)
python -m unittest discover -s tests -v
python scripts\validate_project.py
echo.
echo Full pipeline and validation completed.
pause
