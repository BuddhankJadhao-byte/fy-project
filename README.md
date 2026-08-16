# AI-Based Load Forecasting for Microgrids

A complete B.Tech final-year software project for short-term electrical load forecasting using a Random Forest model, real Indian 33/11 kV substation data, chronological evaluation, an interactive Streamlit dashboard, downloadable forecasts, and automated PDF reporting.

## Project outcome

The application trains a leakage-safe, TimeSeriesSplit-tuned Random Forest model on one complete year (8,760 hourly samples) of active power load, temperature, humidity, and calendar data. Missing values are forward-filled, IQR outliers are removed from the signal and replaced without breaking the hourly timeline, and a training-only MinMaxScaler is saved with each supervised model. It compares Random Forest against Linear Regression, a conditional-least-squares ARIMA(2,1,0), and a 24-hour seasonal-naive baseline. It then produces evaluation metrics, plots, a 24-hour operational forecast, reusable model files, CSV reports, and a Streamlit web interface.

## Dataset

- Location: Godishala 33/11 kV Substation, Huzurabad, Telangana, India
- Coverage: 1 January 2021 to 31 December 2021
- Frequency: hourly
- Rows: 8,760
- Target: active power load in kW
- Weather: temperature and relative humidity
- Calendar: weekend/weekday and season
- Source: V. Veeramsetty et al., *Electric power load dataset*, Mendeley Data, DOI [10.17632/tj54nv46hj.2](https://doi.org/10.17632/tj54nv46hj.2), CC BY 4.0

The raw workbook also contains voltage, current, and power factor. These columns are intentionally excluded from model inputs because they directly calculate three-phase active power and would cause target leakage.

## Folder structure

```text
AI_Load_Forecasting_Microgrid/
├── app.py                         Streamlit dashboard
├── config.yaml                    Paths and model settings
├── requirements.txt               Python packages
├── setup_windows.bat              One-click Windows setup
├── run_full_pipeline.bat          Prepare, train, evaluate, report
├── run_dashboard.bat              Start the web dashboard
├── data/
│   ├── raw/                        Original published workbook
│   └── processed/                  Clean forecasting-ready CSV
├── src/                            Reusable application modules
├── scripts/                        Command-line entry points
├── tests/                          Automated unit tests
├── models/                         Trained model artifacts
├── outputs/                        Metrics, predictions, plots, reports
├── docs/                           Report, manual, viva preparation
└── .vscode/                        VS Code launch and task settings
```

## Run in VS Code on Windows

1. Extract/copy this folder to `F:\FULL PROJECT MATERIALS\AI_Load_Forecasting_Microgrid`.
2. Open that project folder in VS Code.
3. Double-click `setup_windows.bat` once. It creates `.venv` and installs all packages.
4. Double-click `run_full_pipeline.bat` to reproduce the cleaned CSV, train all models, and regenerate outputs.
5. Double-click `run_dashboard.bat` to open the interactive application.

Manual PowerShell commands:

```powershell
cd "F:\FULL PROJECT MATERIALS\AI_Load_Forecasting_Microgrid"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\run_pipeline.py
streamlit run app.py
```

If Python 3.11 is not available, Python 3.10 or 3.12 can be used.

## Main commands

```powershell
# Complete reproducible run (includes TimeSeriesSplit tuning)
python scripts\run_pipeline.py

# Quick development run without tuning
python scripts\run_pipeline.py --no-tune

# Separate stages
python scripts\prepare_data.py
python scripts\train_models.py
python scripts\generate_report.py

# Tests and final integrity checks
python -m unittest discover -s tests -v
python scripts\validate_project.py

# Dashboard
streamlit run app.py
```

## Input CSV format

The cleaned project dataset uses:

```csv
timestamp,load_kw,temperature_c,humidity_pct,is_weekend,season_code,substation_shutdown
2021-01-01 00:00:00,1967.388,18.333,90,0,1,0
```

Any future custom dataset should contain continuous hourly timestamps and at minimum these numerical fields:

- `timestamp`
- `load_kw`
- `temperature_c`
- `humidity_pct`

## Methodology

1. Validate a continuous, monotonic hourly sequence.
2. Convert Fahrenheit to Celsius and retain meaningful numerical columns.
3. Forward-fill missing numerical values and detect IQR outliers; remove flagged values from the signal and forward-fill them without deleting timestamps.
4. Engineer cyclical hour/day/month variables; 1, 24, and 168-hour lags; and shifted rolling statistics.
5. Use an 80/20 chronological train/test split without random shuffling.
6. Fit MinMaxScaler on training data only, tune Random Forest using TimeSeriesSplit, and compare it with Linear Regression, ARIMA, and seasonal-naive baselines.
7. Report MAE, RMSE, MAPE, and R².
8. Generate recursive 1-168 hour forecasts using supplied future weather or a clearly labelled repeated-weather scenario.

## Important engineering decisions

- Lag and rolling features are shifted, so the target hour never enters its own predictors.
- The final 20% is never used to fit the model.
- MinMax scaling is retained in the saved pipelines to match the approved synopsis and is fitted only on training data to prevent leakage.
- Hourly average load is modelled in kW. Multiplying by the one-hour interval gives the corresponding hourly energy in kWh.
- The Indian substation dataset is treated as a software-only microgrid-scale case study; the modular pipeline can be retrained on compatible residential/campus smart-meter data.
- Forecasts support energy-management decisions but do not directly control electrical equipment.

## Deliverables generated

- `models/random_forest.joblib`
- `models/linear_regression.joblib`
- `models/metadata.json`
- `outputs/model_metrics.csv`
- `outputs/test_predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/forecasts/next_24_hours.csv`
- `outputs/reports/AI_Load_Forecasting_Report.pdf`
- `outputs/figures/*.png`

## Team

- Buddhank Vijay Jadhao — Roll No. 08
- Khanderao Kamle — Roll No. 15
- Kavya Pandey — Roll No. 14
- Guide: Mrs. Ashwini Ramakant Sali

Department of Electrical Engineering, G H Raisoni College of Engineering and Management, Wagholi, Pune. Academic Year 2026-27.
