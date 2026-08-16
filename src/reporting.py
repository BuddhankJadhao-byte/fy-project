from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
if (FONT_DIR / "DejaVuSans.ttf").exists():
    pdfmetrics.registerFont(TTFont("ProjectSans", FONT_DIR / "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("ProjectSans-Bold", FONT_DIR / "DejaVuSans-Bold.ttf"))
else:
    # Helvetica is available in every PDF reader and is the safe Windows fallback.
    FONT_DIR = None

FONT_NORMAL = "ProjectSans" if FONT_DIR else "Helvetica"
FONT_BOLD = "ProjectSans-Bold" if FONT_DIR else "Helvetica-Bold"


def build_forecast_report(
    metrics: pd.DataFrame,
    dataset_summary: dict[str, object],
    forecast: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> bytes:
    """Create a concise, reproducible project/forecast PDF report."""
    buffer = BytesIO()
    target = str(output_path) if output_path else buffer
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(target, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    for style_name in ["Normal", "BodyText", "Heading3"]:
        styles[style_name].fontName = FONT_NORMAL
    title = ParagraphStyle("ProjectTitle", parent=styles["Title"], fontName=FONT_BOLD, alignment=TA_CENTER, textColor=colors.HexColor("#123B5D"), fontSize=19, leading=23)
    heading = ParagraphStyle("Section", parent=styles["Heading2"], fontName=FONT_BOLD, textColor=colors.HexColor("#0B6E75"), spaceBefore=10, spaceAfter=6)
    story = [
        Paragraph("AI-Based Load Forecasting for Microgrids", title),
        Paragraph("Godishala 33/11 kV Substation, Telangana, India", styles["Heading3"]),
        Spacer(1, 8),
        Paragraph(
            "This report summarizes the leakage-safe Random Forest short-term load forecasting system developed from one complete year of hourly Indian substation demand and weather data.",
            styles["BodyText"],
        ),
        Paragraph("Dataset Summary", heading),
    ]
    summary_rows = [["Item", "Value"]] + [
        [str(k).replace("_", " ").title(), f"{v:.3f}" if isinstance(v, float) else str(v)]
        for k, v in dataset_summary.items()
    ]
    summary_table = Table(summary_rows, colWidths=[65 * mm, 95 * mm], repeatRows=1)
    summary_table.setStyle(_table_style())
    story.extend([summary_table, Paragraph("Model Performance", heading)])

    metric_rows = [["Model", "MAE (kW)", "RMSE (kW)", "MAPE (%)", "R2"]]
    for _, row in metrics.iterrows():
        metric_rows.append([
            row["model"], f'{row["mae_kw"]:.2f}', f'{row["rmse_kw"]:.2f}',
            f'{row["mape_pct"]:.2f}', f'{row["r2"]:.4f}',
        ])
    metric_table = Table(metric_rows, colWidths=[55 * mm, 27 * mm, 30 * mm, 27 * mm, 22 * mm], repeatRows=1)
    metric_table.setStyle(_table_style())
    story.append(metric_table)

    if forecast is not None and not forecast.empty:
        story.append(Paragraph("Future Forecast", heading))
        story.append(Paragraph(
            f"Forecast period: {forecast['timestamp'].min()} to {forecast['timestamp'].max()}. "
            f"Peak forecast: {forecast['forecast_load_kw'].max():.2f} kW; "
            f"average forecast: {forecast['forecast_load_kw'].mean():.2f} kW.",
            styles["BodyText"],
        ))
        display = forecast.head(48)
        forecast_rows = [["Timestamp", "Forecast load (kW)", "Temperature (C)", "Humidity (%)"]]
        for _, row in display.iterrows():
            forecast_rows.append([
                pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M"),
                f'{row["forecast_load_kw"]:.2f}', f'{row["temperature_c"]:.1f}', f'{row["humidity_pct"]:.1f}',
            ])
        forecast_table = Table(forecast_rows, colWidths=[48 * mm, 40 * mm, 38 * mm, 34 * mm], repeatRows=1)
        forecast_table.setStyle(_table_style(font_size=7.5))
        story.append(forecast_table)

    story.extend([
        Paragraph("Methodology Note", heading),
        Paragraph(
            "Missing values are forward-filled, IQR outliers are removed from the signal and replaced without deleting hourly timestamps, and MinMaxScaler is fitted only on training data. Random Forest tuning uses TimeSeriesSplit before evaluation on a chronological holdout set. Lag and rolling statistics use only past load values. Voltage, current, and power factor were excluded because they directly determine active power and would leak the prediction target.",
            styles["BodyText"],
        ),
    ])
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    if output_path:
        return Path(output_path).read_bytes()
    return buffer.getvalue()


def _table_style(font_size: float = 8.5) -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B5D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_NORMAL),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EAF4F4")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9CB3BF")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])


def _page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#9CB3BF"))
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 11 * mm, A4[0] - 18 * mm, 11 * mm)
    canvas.setFillColor(colors.HexColor("#536A76"))
    canvas.setFont(FONT_NORMAL, 7.5)
    canvas.drawString(18 * mm, 7 * mm, "G H Raisoni College of Engineering and Management - Electrical Engineering")
    canvas.drawRightString(A4[0] - 18 * mm, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()
