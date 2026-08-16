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

tab_overview, tab_data, tab_results, tab_forecast, tab_about = st.tabs(
    ["Overview", "Data Explorer", "Model Results", "Future Forecast", "Methodology"]
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
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("System workflow")
    st.markdown("**Hourly data → forward-fill and IQR cleaning → time/lag features → chronological split → MinMax normalization → tuned Random Forest and baselines → dashboard, CSV and PDF reports**")
    st.success(f"Preprocessing verified: no missing values; {summary.get('iqr_outliers_replaced', 0)} IQR load outlier(s) replaced without breaking the hourly timeline.")

with tab_data:
    st.subheader("Historical demand and weather")
    date_range = st.date_input(
        "Select period",
        value=(data["timestamp"].min().date(), min(data["timestamp"].min().date() + pd.Timedelta(days=display_days), data["timestamp"].max().date())),
        min_value=data["timestamp"].min().date(),
        max_value=data["timestamp"].max().date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        view = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
    else:
        view = data.head(24 * display_days)
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=view["timestamp"], y=view["load_kw"], name="Load (kW)", line=dict(color="#123B5D")))
    chart.add_trace(go.Scatter(x=view["timestamp"], y=view["temperature_c"], name="Temperature (°C)", yaxis="y2", line=dict(color="#F58518")))
    chart.update_layout(
        title="Hourly Load and Temperature", xaxis_title="Timestamp", yaxis=dict(title="Load (kW)"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"), legend=dict(orientation="h"),
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download cleaned full-year CSV", data.to_csv(index=False).encode(), "godishala_microgrid_hourly_2021.csv", "text/csv")

with tab_results:
    st.subheader("Leakage-safe holdout performance")
    formatted = metrics.copy()
    for column in ["mae_kw", "rmse_kw", "mape_pct", "r2"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    left, right = st.columns([1.45, 1])
    with left:
        recent = predictions.tail(display_days * 24)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["actual_kw"], name="Actual", line=dict(color="#123B5D", width=2)))
        fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["random_forest_kw"], name="Random Forest", line=dict(color="#0B6E75", width=1.7)))
        fig.update_layout(title=f"Actual vs Predicted - Last {display_days} Test Days", yaxis_title="Load (kW)", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        imp_fig = px.bar(importance.head(12).sort_values("importance"), x="importance", y="feature", orientation="h", title="Feature Importance")
        imp_fig.update_traces(marker_color="#0B6E75")
        st.plotly_chart(imp_fig, use_container_width=True)
    st.download_button("Download test predictions", predictions.to_csv(index=False).encode(), "test_predictions.csv", "text/csv")

with tab_forecast:
    st.subheader("Operational forecast")
    st.caption("When a future weather file is not supplied, the latest 24-hour temperature and humidity profile is repeated as a transparent scenario assumption.")
    horizon = st.slider("Forecast horizon (hours)", 1, 168, 24)
    weather_file = st.file_uploader("Optional future weather CSV", type="csv", help="Columns: timestamp, temperature_c, humidity_pct")
    future_weather = None
    if weather_file:
        future_weather = pd.read_csv(weather_file)
    if st.button("Generate forecast", type="primary"):
        try:
            forecast = forecast_future(data, model, horizon, future_weather)
            st.session_state["forecast"] = forecast
        except Exception as exc:
            st.error(f"Forecast could not be generated: {exc}")

    if "forecast" in st.session_state:
        forecast = st.session_state["forecast"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Average forecast", f"{forecast['forecast_load_kw'].mean():,.1f} kW")
        c2.metric("Peak forecast", f"{forecast['forecast_load_kw'].max():,.1f} kW")
        c3.metric("Minimum forecast", f"{forecast['forecast_load_kw'].min():,.1f} kW")
        fig = px.line(forecast, x="timestamp", y="forecast_load_kw", markers=True, title="Future Load Forecast")
        fig.update_traces(line_color="#0B6E75", line_width=3)
        fig.update_layout(yaxis_title="Forecast load (kW)")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.download_button("Download forecast CSV", forecast.to_csv(index=False).encode(), "future_load_forecast.csv", "text/csv")
        pdf = build_forecast_report(metrics, summary, forecast)
        col2.download_button("Download forecast PDF report", pdf, "AI_Load_Forecasting_Report.pdf", "application/pdf")
        st.dataframe(forecast, use_container_width=True, hide_index=True)

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
