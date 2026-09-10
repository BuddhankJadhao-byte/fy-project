@echo off
setlocal
cd /d "%~dp0\.."

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo ========================================================
echo Running Data Cleaning: Outlier Removal and Missing Values
echo ========================================================
"%PYTHON_EXE%" data_cleaning\clean_outliers_and_missing.py --input data\raw\godishala_substation_2021.xlsx --output_dir data_cleaning --iqr_factor 1.5
if errorlevel 1 (
    echo [ERROR] Cleaning script failed.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Cleaned files and reports generated in data_cleaning\
echo ========================================================
pause
