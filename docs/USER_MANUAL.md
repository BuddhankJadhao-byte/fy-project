# User Manual

## First-time installation

1. Install Python 3.10, 3.11, or 3.12 and VS Code.
2. Extract the project into `F:\FULL PROJECT MATERIALS`.
3. Open the `AI_Load_Forecasting_Microgrid` folder in VS Code.
4. Run `setup_windows.bat` and wait for `Setup completed successfully`.
5. Run `run_full_pipeline.bat`. This reproduces every model and report.
6. Run `run_dashboard.bat`. The application opens in the default browser.

## Dashboard use

- **Overview:** verify dataset size and final Random Forest metrics.
- **Data Explorer:** choose dates, inspect load and weather, or download the cleaned CSV.
- **Model Results:** compare algorithms and inspect recent test predictions.
- **Future Forecast:** choose 1-168 hours and select Generate forecast.
- **Methodology:** review assumptions, data leakage controls, and metadata.

## Optional future weather CSV

Supply exactly one row per forecast hour:

```csv
timestamp,temperature_c,humidity_pct
2022-01-01 00:00:00,20.0,94
2022-01-01 01:00:00,20.2,93
```

The timestamps must begin one hour after the last historical record and cover the entire selected horizon.

## Common issues

- `Python was not found`: reinstall Python and enable **Add Python to PATH**.
- `No module named streamlit`: run `setup_windows.bat` again.
- `Project artifacts are missing`: run `run_full_pipeline.bat` before the dashboard.
- PowerShell blocks activation: use the supplied `.bat` files, or run `Set-ExecutionPolicy -Scope Process Bypass` for the current terminal only.
- Port already in use: run `streamlit run app.py --server.port 8502`.

## Final verification

```powershell
python -m unittest discover -s tests -v
python scripts\validate_project.py
```

Both commands must finish with `OK`/`PASS` messages.
