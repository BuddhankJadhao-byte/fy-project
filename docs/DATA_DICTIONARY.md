# Data Dictionary

| Column | Type | Unit / values | Purpose |
|---|---|---|---|
| timestamp | datetime | Hourly, Asia/Kolkata context | Time index |
| load_kw | float | kW | Prediction target |
| temperature_c | float | °C | Weather predictor |
| humidity_pct | float | 0-100% | Weather predictor |
| is_weekend | integer | 0 weekday, 1 weekend | Calendar predictor |
| season_code | integer | 0 rainy, 1 winter, 2 summer | Seasonal predictor |
| substation_shutdown | integer | 0 no, 1 yes | Data-quality/context flag; not a model feature |

Engineered columns are created at runtime and not duplicated in the clean source CSV. All load lag and rolling columns contain only prior-hour information.
