# Data Cleaning: Outlier Removal & Missing Value Pipeline

This folder contains the complete, standalone preprocessing module that identifies, treats, and visualizes **missing values** and **statistical outliers** from the raw Godishala Substation dataset.

---

## Folder Contents

| File | Description |
| :--- | :--- |
| `clean_outliers_and_missing.py` | Standalone Python pipeline for cleaning, outlier detection, data exports, and visualization. |
| `run_cleaning.bat` | One-click Windows batch runner to execute the cleaning pipeline. |
| `cleaned_data_imputed.csv` | **8,760 hourly records** with outliers imputed via forward/backward fill to preserve continuous hourly time-series structure. |
| `cleaned_data_dropped.csv` | **8,511 records** with all outlier rows strictly dropped (removes 249 outlier samples). |
| `before_after_comparison.csv` | Statistical comparison table (mean, std, min, quartiles, max, skewness) before and after cleaning. |
| `cleaning_summary.json` | Programmatic summary of outlier bounds, missing values, and row retention. |
| `CLEANING_REPORT.md` | Full markdown audit report of the cleaning process. |
| `boxplots_before_after.png` | Box plots comparing distributions across Raw, Imputed, and Dropped versions. |
| `distribution_comparison.png` | Probability density functions (KDE) comparing before vs. after cleaning. |
| `outliers_timeline.png` | Chronological plot of active power load highlighting detected outlier spikes. |

---

## How to Run

### Option 1: One-Click on Windows
Double-click `run_cleaning.bat`.

### Option 2: Command Line
```powershell
.\.venv\Scripts\python.exe data_cleaning\clean_outliers_and_missing.py --input data\raw\godishala_substation_2021.xlsx --output_dir data_cleaning --iqr_factor 1.5
```

You can customize `--iqr_factor` (e.g., `3.0` for extreme outliers only, or `1.5` for standard Tukey fences).
