from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.data import dataset_summary, load_dataset
from src.forecasting import forecast_future
from src.modeling import load_model_artifacts
from src.reporting import build_forecast_report

st.set_page_config(page_title="AI Microgrid Load Forecasting", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    :root {color-scheme: light;}
    .stApp, [data-testid="stAppViewContainer"] {background-color: #FFFFFF; color: #172B3A;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: #f1f7f8; border: 1px solid #d2e4e7; padding: 12px; border-radius: 10px;}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {color: #123B5D !important; opacity: 1 !important;}
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] div {color: #123B5D !important; opacity: 1 !important;}
    [data-testid="stSidebar"] {background-color: #F1F7F8;}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {color: #172B3A !important;}
    h1, h2, h3 {color: #123B5D;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data(path: Path) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_data
def get_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"] if "predictions" in path.name else None)


@st.cache_resource
def get_model(model_dir: Path):
    return load_model_artifacts(model_dir)


config = load_config()
data_path = config["paths"]["processed_data"]
model_path = config["paths"]["model_dir"] / "random_forest.joblib"
if not data_path.exists() or not model_path.exists():
    st.error("Project artifacts are missing. Run `python scripts/run_pipeline.py` first.")
    st.stop()

data = get_data(data_path)
model, metadata = get_model(config["paths"]["model_dir"])
metrics = pd.read_csv(config["paths"]["output_dir"] / "model_metrics.csv")
predictions = pd.read_csv(config["paths"]["output_dir"] / "test_predictions.csv", parse_dates=["timestamp"])
importance = pd.read_csv(config["paths"]["output_dir"] / "feature_importance.csv")
summary = dataset_summary(data)

st.title("⚡ AI-Based Load Forecasting for Microgrids")
st.caption("Random Forest short-term forecasting | Software-only microgrid-scale case study using Godishala 33/11 kV Substation data")

with st.sidebar:
    st.header("Project controls")
    display_days = st.slider("Chart window (days)", 2, 60, 14)
    st.info("Dataset: 8,760 hourly records covering all of 2021. Target: active power load (kW).")
    st.caption("For one-hour records, average kW × 1 hour gives hourly energy in kWh.")
    st.markdown("**Core model:** Random Forest Regressor")
    st.markdown("**Validation:** chronological 80/20 split")

tab_forecast, tab_overview, tab_data, tab_results, tab_about = st.tabs(
    ["Predict Load", "Overview", "Historical Data", "Model Results", "Methodology"]
)

with tab_overview:
    cols = st.columns(5)
    cols[0].metric("Hourly samples", f"{summary['rows']:,}")
    cols[1].metric("Average load", f"{summary['load_mean_kw']:,.0f} kW")
    cols[2].metric("Peak load", f"{summary['load_max_kw']:,.0f} kW")
    rf_metric = metrics.loc[metrics["model"] == "Random Forest"].iloc[0]
    cols[3].metric("RF MAE", f"{rf_metric['mae_kw']:.1f} kW")
    cols[4].metric("RF R²", f"{rf_metric['r2']:.3f}")

    monthly = data.set_index("timestamp").resample("ME").agg(load_kw=("load_kw", "mean")).reset_index()
    fig = px.line(monthly, x="timestamp", y="load_kw", markers=True, title="Monthly Average Electrical Load")
    fig.update_traces(line_color="#0B6E75", line_width=3)
    fig.update_layout(yaxis_title="Average load (kW)", xaxis_title="Month")
    st.plotly_chart(fig, width="stretch")

    st.subheader("System workflow")
    st.markdown("**Hourly data → forward-fill and IQR cleaning → time/lag features → chronological split → MinMax normalization → tuned Random Forest and baselines → dashboard, CSV and PDF reports**")
    st.success(f"Preprocessing verified: no missing values; {summary.get('iqr_outliers_replaced', 0)} IQR load outlier(s) replaced without breaking the hourly timeline.")

with tab_data:
    st.subheader("Historical data explorer (2021 only)")
    st.info("This tab displays measured historical data only. To select 2022 or a later date, use the **Predict Load** tab.")
    date_range = st.date_input(
        "Historical period (January–December 2021)",
        value=(data["timestamp"].min().date(), min((pd.Timestamp(data["timestamp"].min()) + pd.Timedelta(days=display_days)).date(), data["timestamp"].max().date())),
        min_value=data["timestamp"].min().date(),
        max_value=data["timestamp"].max().date(),
        key="historical_period_2021_v2",
        help="Only measured 2021 dates are available here.",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        view = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
    else:
        view = data.head(24 * display_days)

    st.subheader("Raw hourly records")
    st.caption(f"Showing {len(view):,} hourly rows. Scroll vertically, sort columns, or use the table toolbar to search and expand.")
    raw_metrics = st.columns(4)
    raw_metrics[0].metric("Rows shown", f"{len(view):,}")
    raw_metrics[1].metric("Average load", f"{view['load_kw'].mean():,.1f} kW")
    raw_metrics[2].metric("Maximum load", f"{view['load_kw'].max():,.1f} kW")
    raw_metrics[3].metric("Average temperature", f"{view['temperature_c'].mean():.1f} °C")

    st.dataframe(
        view,
        width="stretch",
        height=520,
        row_height=38,
        hide_index=True,
        key="historical_raw_data_table",
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Date and time", format="DD MMM YYYY, HH:mm", pinned=True),
            "load_kw": st.column_config.NumberColumn("Load (kW)", format="%.2f", pinned=True),
            "temperature_c": st.column_config.NumberColumn("Temperature (°C)", format="%.1f"),
            "humidity_pct": st.column_config.NumberColumn("Humidity (%)", format="%.1f"),
            "is_weekend": st.column_config.NumberColumn("Weekend (0/1)", format="%d"),
            "season_code": st.column_config.NumberColumn("Season code", format="%d"),
            "substation_shutdown": st.column_config.NumberColumn("Shutdown (0/1)", format="%d"),
            "load_outlier_flag": st.column_config.NumberColumn("Outlier replaced (0/1)", format="%d"),
        },
    )
    with st.container(horizontal=True):
        st.download_button(
            "Download displayed rows",
            view.to_csv(index=False).encode(),
            "selected_historical_data.csv",
            "text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Download all 2021 data",
            data.to_csv(index=False).encode(),
            "godishala_microgrid_hourly_2021.csv",
            "text/csv",
            icon=":material/download:",
        )

    st.subheader("Load and temperature chart")
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=view["timestamp"], y=view["load_kw"], name="Load (kW)", line=dict(color="#123B5D")))
    chart.add_trace(go.Scatter(x=view["timestamp"], y=view["temperature_c"], name="Temperature (°C)", yaxis="y2", line=dict(color="#F58518")))
    chart.update_layout(
        title="Hourly Load and Temperature", xaxis_title="Timestamp", yaxis=dict(title="Load (kW)"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"), legend=dict(orientation="h"),
    )
    st.plotly_chart(chart, width="stretch")

with tab_results:
    st.subheader("Leakage-safe holdout performance")
    formatted = metrics.copy()
    for column in ["mae_kw", "rmse_kw", "mape_pct", "r2"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    st.dataframe(formatted, width="stretch", hide_index=True)

    left, right = st.columns([1.45, 1])
    with left:
        recent = predictions.tail(display_days * 24)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["actual_kw"], name="Actual", line=dict(color="#123B5D", width=2)))
        fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["random_forest_kw"], name="Random Forest", line=dict(color="#0B6E75", width=1.7)))
        fig.update_layout(title=f"Actual vs Predicted - Last {display_days} Test Days", yaxis_title="Load (kW)", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    with right:
        imp_fig = px.bar(importance.head(12).sort_values("importance"), x="importance", y="feature", orientation="h", title="Feature Importance")
        imp_fig.update_traces(marker_color="#0B6E75")
        st.plotly_chart(imp_fig, width="stretch")
    st.download_button("Download test predictions", predictions.to_csv(index=False).encode(), "test_predictions.csv", "text/csv")

with tab_forecast:
    st.subheader("Predict electrical load")
    st.caption("Choose a day or period. The app automatically uses the existing 2021 load and weather dataset with the trained Random Forest model.")
    forecast_mode = st.segmented_control(
        "What do you want to predict?",
        ["Today", "Tomorrow", "Any day", "Date range", "Full year"],
        default="Tomorrow",
        key="forecast_period_mode_v4",
    )
    earliest_date = (pd.Timestamp(data["timestamp"].max()) + pd.Timedelta(hours=1)).date()
    today = max(pd.Timestamp.now().date(), earliest_date)
    tomorrow = max((pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).date(), earliest_date)
    max_forecast_date = (pd.Timestamp.now() + pd.DateOffset(years=10)).date()
    if forecast_mode == "Any day":
        start_date = st.date_input(
            "Day to predict",
            value=tomorrow,
            min_value=earliest_date,
            max_value=max_forecast_date,
            key="single_forecast_day_v4",
        )
        forecast_days = 1
    elif forecast_mode == "Date range":
        selected_range = st.date_input(
            "Forecast date range (up to 366 days)",
            value=(tomorrow, min(tomorrow + pd.Timedelta(days=6), max_forecast_date)),
            min_value=earliest_date,
            max_value=max_forecast_date,
            key="future_forecast_range_v4",
            help="Choose any period from 2022 onward, including a complete previous or future year.",
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date = end_date = selected_range[0] if isinstance(selected_range, tuple) else selected_range
        forecast_days = (end_date - start_date).days + 1
        if forecast_days > 366:
            st.error("Please select no more than 366 days for one forecast.")
            st.stop()
    elif forecast_mode == "Full year":
        available_years = list(range(2022, max_forecast_date.year + 1))
        selected_year = st.selectbox(
            "Year to predict",
            available_years,
            index=available_years.index(pd.Timestamp.now().year),
            key="full_forecast_year_v4",
        )
        start_date = pd.Timestamp(selected_year, 1, 1).date()
        end_date = pd.Timestamp(selected_year, 12, 31).date()
        forecast_days = (end_date - start_date).days + 1
        st.info(f"Full-year forecast: {start_date:%d %B %Y} to {end_date:%d %B %Y}")
    elif forecast_mode == "Today":
        start_date, forecast_days = today, 1
        st.info(f"Today's forecast: {start_date:%d %B %Y} (24 hourly values)")
    else:
        start_date, forecast_days = tomorrow, 1
        st.info(f"Tomorrow's forecast: {start_date:%d %B %Y} (24 hourly values)")
    horizon = forecast_days * 24
    start_timestamp = pd.Timestamp(start_date)
    if st.button("Predict load", type="primary", icon=":material/bolt:"):
        try:
            with st.spinner(f"Generating {forecast_days:,}-day forecast..."):
                forecast = forecast_future(data, model, horizon, start_timestamp=start_timestamp)
            st.session_state["forecast"] = forecast
        except Exception as exc:
            st.error(f"Forecast could not be generated: {exc}")

    if "forecast" in st.session_state:
        forecast = st.session_state["forecast"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Average forecast", f"{forecast['forecast_load_kw'].mean():,.1f} kW")
        c2.metric("Peak forecast", f"{forecast['forecast_load_kw'].max():,.1f} kW")
        c3.metric("Minimum forecast", f"{forecast['forecast_load_kw'].min():,.1f} kW")
        st.caption(f"Forecast period: {forecast['timestamp'].min():%d %b %Y %H:%M} to {forecast['timestamp'].max():%d %b %Y %H:%M}")
        fig = px.line(forecast, x="timestamp", y="forecast_load_kw", markers=True, title="Future Load Forecast")
        fig.update_traces(line_color="#0B6E75", line_width=3)
        fig.update_layout(yaxis_title="Forecast load (kW)")
        st.plotly_chart(fig, width="stretch")
        col1, col2 = st.columns(2)
        col1.download_button("Download forecast CSV", forecast.to_csv(index=False).encode(), "future_load_forecast.csv", "text/csv")
        pdf = build_forecast_report(metrics, summary, forecast)
        col2.download_button("Download forecast PDF report", pdf, "AI_Load_Forecasting_Report.pdf", "application/pdf")
        st.dataframe(forecast, width="stretch", hide_index=True)

with tab_about:
    st.subheader("Methodology and safeguards")
    st.markdown(
        """
        - **Objective:** predict short-term active power demand for localized energy management.
        - **Inputs:** historical load, temperature, humidity, calendar cycles, 1/24/168-hour lags, and shifted rolling statistics.
        - **Preprocessing:** forward-fill missing values, IQR outlier detection/replacement, and training-only MinMax normalization.
        - **Core model:** TimeSeriesSplit-tuned Random Forest Regressor; compared with Linear Regression, ARIMA, and a 24-hour seasonal-naive baseline.
        - **Validation:** the last 20% of records are held out chronologically; no random shuffling is used.
        - **Leakage prevention:** voltage, current, and power factor are excluded because they directly calculate active power.
        - **Scope:** the substation dataset is used as a software-only microgrid-scale forecasting case study; the modular pipeline can be retrained on residential/campus smart-meter data.
        - **Limitations:** this is decision-support software, not an automatic grid controller. Forecast quality depends on data and future weather quality.
        """
    )
    st.markdown("**Dataset attribution:** Veeramsetty et al., *Electric power load dataset*, Mendeley Data, DOI 10.17632/tj54nv46hj.2, CC BY 4.0.")
    st.json(metadata, expanded=False)
