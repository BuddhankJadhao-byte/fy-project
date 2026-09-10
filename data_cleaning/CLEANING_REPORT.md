# Data Cleaning & Outlier Removal Report

## 1. Overview
- **Dataset:** Godishala 33/11 kV Substation (Telangana, India)
- **Original Records:** 8,760 hourly samples
- **Initial Missing Values:** 8,694
- **IQR Multiplier:** 1.5× (Standard Tukey Fence)
- **Total Outlier Rows Detected:** 249 (2.84%)

---

## 2. Missing Value Resolution
- **Strategy:** Monotonic forward fill (`ffill`) followed by backward fill (`bfill`) for edge conditions.
- **Substation Shutdown:** 8694 missing cells filled with `0` (normal operational state).
- **Physical Sensor Bounds:**
  - `humidity_pct`: Values bounded to `[0.0, 100.0]%`. Corrected unphysical values exceeding 100%.
  - `load_kw`: Values bounded to `[0.0, ∞) kW`. Negative load values eliminated.

---

## 3. Statistical Outlier Detection (IQR Method)
Formulas applied:
- $IQR = Q_3 - Q_1$
- $\text{Lower Bound} = \max(0, Q_1 - 1.5 \times IQR)$
- $\text{Upper Bound} = Q_3 + 1.5 \times IQR$

| Feature | $Q_1$ (25%) | $Q_3$ (75%) | $IQR$ | Lower Bound | Upper Bound | Outliers Count | Outlier % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `load_kw` | 1062.656 | 3025.769 | 1963.112 | -1882.012 | 5970.438 | **7** | 0.08% |
| `temperature_c` | 24.444 | 30.0 | 5.556 | 16.111 | 38.333 | **242** | 2.763% |
| `humidity_pct` | 52.0 | 87.0 | 35.0 | -0.5 | 139.5 | **0** | 0.0% |

---

## 4. Generated Cleaned Datasets

1. **`cleaned_data_imputed.csv` (Recommended for Time-Series Forecasting)**:
   - **Rows:** 8,760 (100% continuous hourly coverage).
   - Outliers are neutralized via forward-fill and interpolation, ensuring zero gaps in lag features (`lag_1h`, `lag_24h`, `lag_168h`) and rolling windows.
2. **`cleaned_data_dropped.csv` (For Standard Tabular Regression)**:
   - **Rows:** 8,511 (249 rows removed).
   - Strictly removes any sample with outlier values.

---

## 5. Artifacts in this Folder
- `cleaned_data_imputed.csv`: Complete time-series cleaned dataset.
- `cleaned_data_dropped.csv`: Filtered tabular dataset.
- `before_after_comparison.csv`: Full statistical metrics before vs after cleaning.
- `cleaning_summary.json`: Programmatic metadata and statistics.
- `boxplots_before_after.png`: Multi-panel before/after boxplot chart.
- `distribution_comparison.png`: Probability density distributions.
- `outliers_timeline.png`: Time-series signal highlighting detected outlier spikes.
