from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget


# =========================================================
# Configuration
# =========================================================
APP_DIR = Path(__file__).parent
DATA_FILE = APP_DIR / "irish_climate_data.csv"
JSON_FILES = sorted(APP_DIR.glob("*.json"))

MONTH_MAP = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Sept": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_NUM_MAP = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}

STATION_NAME_MAP = {
    "athenry": "Athenry",
    "belmullet": "Belmullet",
    "cork airport": "Cork Airport",
    "dublin airport": "Dublin Airport",
    "shannon airport": "Shannon Airport",
}

STATION_COORDS = pd.DataFrame(
    [
        {"Station": "Dublin Airport", "lat": 53.4264, "lon": -6.2499, "Region": "East"},
        {"Station": "Cork Airport", "lat": 51.8472, "lon": -8.4911, "Region": "South"},
        {"Station": "Athenry", "lat": 53.2964, "lon": -8.7431, "Region": "West"},
        {"Station": "Belmullet", "lat": 54.2253, "lon": -10.0064, "Region": "Northwest"},
        {"Station": "Shannon Airport", "lat": 52.7020, "lon": -8.9248, "Region": "Mid-West"},
    ]
)

PLOT_TEMPLATE = "plotly_white"
PLOT_HEIGHT = 500


# =========================================================
# Data preparation
# =========================================================
def build_dataset_from_json(folder: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for path in sorted(folder.glob("*.json")):
        station_name = path.stem

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        rainfall = data.get("total_rainfall", {}).get("report", {})
        temp = data.get("mean_temperature", {}).get("report", {})

        for year, months in rainfall.items():
            if year == "LTA":
                continue

            for month, rain_val in months.items():
                if month == "annual":
                    continue

                temp_val = temp.get(year, {}).get(month)
                if rain_val in ("", None) or temp_val in ("", None):
                    continue

                rows.append(
                    {
                        "Station": station_name,
                        "Year": year,
                        "Month": str(month).capitalize(),
                        "Rainfall": rain_val,
                        "Temperature": temp_val,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid climate records were found in the JSON files.")

    return df


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[int], int]:
    if JSON_FILES:
        raw = build_dataset_from_json(APP_DIR)
        raw.to_csv(DATA_FILE, index=False)
    elif DATA_FILE.exists():
        raw = pd.read_csv(DATA_FILE)
    else:
        raise FileNotFoundError("No JSON files or CSV dataset found in the app directory.")

    df = raw.copy()

    df["Station"] = (
        df["Station"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(STATION_NAME_MAP)
    )

    df["Month"] = df["Month"].astype(str).str.strip().replace(MONTH_MAP)

    for col in ["Year", "Rainfall", "Temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Month_Num"] = df["Month"].map(MONTH_NUM_MAP)
    df["Date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month_Num"], day=1),
        errors="coerce",
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Station", "Year", "Month", "Month_Num", "Date", "Rainfall", "Temperature"]
    )

    df["Year"] = df["Year"].astype(int)
    df["Month_Num"] = df["Month_Num"].astype(int)

    monthly_baseline = (
        df.groupby(["Station", "Month_Num", "Month"], as_index=False)
        .agg(
            Avg_Rainfall=("Rainfall", "mean"),
            Avg_Temperature=("Temperature", "mean"),
        )
    )

    df = df.merge(monthly_baseline, on=["Station", "Month_Num", "Month"], how="left")
    df["Rainfall_Anomaly"] = df["Rainfall"] - df["Avg_Rainfall"]
    df["Temperature_Anomaly"] = df["Temperature"] - df["Avg_Temperature"]

    annual_counts = (
        df.groupby(["Station", "Year"], as_index=False)
        .agg(month_count=("Month_Num", "nunique"))
    )
    full_years = annual_counts.loc[annual_counts["month_count"] == 12, ["Station", "Year"]]

    annual_summary = (
        df.merge(full_years, on=["Station", "Year"], how="inner")
        .groupby(["Station", "Year"], as_index=False)
        .agg(
            Annual_Rainfall=("Rainfall", "sum"),
            Annual_Temperature=("Temperature", "mean"),
            Annual_Rainfall_Anomaly=("Rainfall_Anomaly", "sum"),
            Annual_Temperature_Anomaly=("Temperature_Anomaly", "mean"),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    stations = sorted(df["Station"].unique().tolist())
    years = sorted(df["Year"].unique().tolist())
    latest_full_year = int(annual_summary["Year"].max()) if not annual_summary.empty else int(df["Year"].max())

    return df, annual_summary, stations, years, latest_full_year


df, annual_summary, stations, years, latest_full_year = load_and_prepare_data()

DEFAULT_OVERVIEW_STATION = "Dublin Airport" if "Dublin Airport" in stations else stations[0]
DEFAULT_COMPARE_STATION = "Cork Airport" if "Cork Airport" in stations else stations[0]
DEFAULT_OVERVIEW_YEAR = str(latest_full_year)

DEFAULT_TREND_STATIONS = stations[: min(2, len(stations))]
DEFAULT_TREND_METRIC = "Rainfall"
DEFAULT_YEAR_RANGE = (min(years), max(years))

DEFAULT_SEASON_STATIONS = stations
DEFAULT_SEASON_METRIC = "Rainfall"
DEFAULT_SHOW_POINTS = True

DEFAULT_ANOM_STATION = stations[0]
DEFAULT_ANOM_YEAR = str(latest_full_year)
DEFAULT_ANOM_METRIC = "Rainfall_Anomaly"


# =========================================================
# Helper functions
# =========================================================
def fmt_num(x: float | int | None, decimals: int = 1, suffix: str = "") -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}{suffix}"


def metric_card(title: str, value: str, subtitle: str = "", accent_class: str = "accent-blue"):
    return ui.div(
        {"class": f"metric-card {accent_class}"},
        ui.div(title, class_="metric-title"),
        ui.div(value, class_="metric-value"),
        ui.div(subtitle, class_="metric-subtitle"),
    )


def section_title(title: str, subtitle: str):
    return ui.div(
        {"class": "section-header"},
        ui.h2(title, class_="section-title"),
        ui.p(subtitle, class_="section-subtitle"),
    )


def empty_figure(message: str = "No data available for this selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def apply_common_layout(fig: go.Figure, title: str | None = None, height: int = PLOT_HEIGHT) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        title=title,
        margin=dict(l=24, r=24, t=56, b=24),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(15,23,42,0.08)", zeroline=False)
    return fig


def get_annual_row(station: str, year: int) -> pd.Series | None:
    row = annual_summary[(annual_summary["Station"] == station) & (annual_summary["Year"] == year)]
    if row.empty:
        return None
    return row.iloc[0]


APP_CSS = """
:root {
  --bg: #f4f7fb;
  --panel: #ffffff;
  --border: #d9e2ec;
  --text: #102a43;
  --muted: #627d98;
  --blue: #2563eb;
  --cyan: #0ea5e9;
  --green: #16a34a;
  --orange: #ea580c;
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  --radius: 20px;
}

html, body {
  background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
  color: var(--text);
}

.container-fluid {
  max-width: 1380px;
}

.navbar {
  border-bottom: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 6px 24px rgba(15, 23, 42, 0.04);
}

.section-header {
  margin-bottom: 1rem;
}

.section-title {
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 0.35rem;
}

.section-subtitle {
  color: var(--muted);
  font-size: 1rem;
  max-width: 900px;
}

.dashboard-shell {
  display: grid;
  gap: 1rem;
}

.panel-card,
.metric-card,
.control-card {
  background: var(--panel);
  border: 1px solid rgba(15, 23, 42, 0.07);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.control-card,
.panel-card {
  padding: 1rem 1.1rem;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
  margin: 0.5rem 0 1rem;
}

.metric-card {
  padding: 1rem 1.1rem;
  position: relative;
  overflow: hidden;
}

.metric-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 4px;
  opacity: 0.95;
}

.metric-card.accent-blue::before { background: linear-gradient(90deg, #2563eb, #60a5fa); }
.metric-card.accent-green::before { background: linear-gradient(90deg, #16a34a, #4ade80); }
.metric-card.accent-orange::before { background: linear-gradient(90deg, #ea580c, #fb923c); }

.metric-title {
  font-size: 0.9rem;
  color: var(--muted);
  margin-bottom: 0.4rem;
}

.metric-value {
  font-size: 1.85rem;
  font-weight: 800;
  line-height: 1.1;
  margin-bottom: 0.3rem;
}

.metric-subtitle {
  color: var(--muted);
  font-size: 0.9rem;
}

.chart-card {
  background: var(--panel);
  border-radius: var(--radius);
  border: 1px solid rgba(15, 23, 42, 0.07);
  box-shadow: var(--shadow);
  padding: 0.85rem 1rem 0.35rem;
}

.chart-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0.2rem 0 0.8rem;
}

.insight-box {
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  border: 1px solid rgba(37, 99, 235, 0.12);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1rem 1.1rem;
}

.insight-title {
  font-weight: 800;
  margin-bottom: 0.65rem;
}

.form-label, .control-label {
  font-weight: 700;
  color: var(--text);
}

.shiny-input-container {
  margin-bottom: 0.75rem;
}

.table {
  background: white;
  border-radius: 14px;
  overflow: hidden;
}

.panel-card ul {
  margin-bottom: 0;
  padding-left: 1.25rem;
}

.panel-card li {
  margin-bottom: 0.45rem;
  color: var(--text);
}

.info-note {
  color: var(--muted);
  font-size: 0.95rem;
}

.panel-card h4 {
  font-weight: 800;
  margin-bottom: 0.7rem;
}

.reset-wrap {
  display: flex;
  align-items: end;
  justify-content: flex-end;
  padding-bottom: 0.75rem;
}

.btn-reset {
  width: 100%;
  border-radius: 12px;
  font-weight: 700;
}

@media (max-width: 992px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
"""


# =========================================================
# UI
# =========================================================
app_ui = ui.page_navbar(
    ui.head_content(ui.tags.style(APP_CSS)),

    ui.nav_panel(
        "Overview",
        ui.div(
            {"class": "dashboard-shell"},
            section_title(
                "Ireland Climate Explorer",
                "Compare stations, inspect full-year climate summaries, and spot patterns across rainfall and temperature.",
            ),
            ui.div(
                {"class": "panel-card"},
                ui.h4("How to read the overview"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Annual rainfall"), " is the total rainfall across all 12 months of the selected full year."),
                    ui.tags.li(ui.tags.b("Mean temperature"), " is the average monthly temperature across that same year."),
                    ui.tags.li(ui.tags.b("Comparison cards"), " show how the primary station differs from the comparison station for the chosen year."),
                    ui.tags.li(ui.tags.b("Rainfall and temperature anomalies"), " show whether the selected year was above or below that station's usual monthly baseline."),
                    ui.tags.li(ui.tags.b("Station map"), " helps you compare where stations are located and how annual conditions vary across Ireland."),
                ),
            ),
            ui.div(
                {"class": "control-card"},
                ui.layout_columns(
                    ui.input_selectize(
                        "station_overview",
                        "Primary station",
                        choices=stations,
                        selected=DEFAULT_OVERVIEW_STATION,
                    ),
                    ui.input_selectize(
                        "compare_overview",
                        "Compare with",
                        choices=stations,
                        selected=DEFAULT_COMPARE_STATION,
                    ),
                    ui.input_selectize(
                        "year_overview",
                        "Year",
                        choices=[str(y) for y in sorted(annual_summary["Year"].unique(), reverse=True)],
                        selected=DEFAULT_OVERVIEW_YEAR,
                    ),
                    ui.div(
                        {"class": "reset-wrap"},
                        ui.input_action_button(
                            "reset_overview",
                            "Reset overview",
                            class_="btn btn-outline-primary btn-reset",
                        ),
                    ),
                    col_widths=[3, 3, 3, 3],
                ),
            ),
            ui.output_ui("overview_cards"),
            ui.layout_columns(
                ui.div(
                    {"class": "chart-card"},
                    ui.div("Annual comparison", class_="chart-title"),
                    output_widget("annual_compare_plot"),
                ),
                ui.div(
                    {"class": "insight-box"},
                    ui.div("Quick insight", class_="insight-title"),
                    ui.output_ui("overview_insight"),
                ),
                col_widths=[8, 4],
            ),
            ui.div(
                {"class": "chart-card"},
                ui.div("Station map", class_="chart-title"),
                output_widget("station_map"),
            ),
        ),
    ),

    ui.nav_panel(
        "Trends",
        ui.div(
            {"class": "dashboard-shell"},
            section_title(
                "Long-term trends",
                "Track rainfall or temperature over time across multiple stations and narrow the view to a chosen year range.",
            ),
            ui.div(
                {"class": "panel-card"},
                ui.h4("How to read trends"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Trend lines"), " show how rainfall or temperature changes month by month over time for each selected station."),
                    ui.tags.li(ui.tags.b("Station summary"), " condenses the selected time window into latest value, average, minimum, maximum, and overall change."),
                    ui.tags.li(ui.tags.b("Net change"), " compares the first visible value in the selected range with the latest visible value."),
                    ui.tags.li(ui.tags.b("Use the year slider"), " to focus on shorter or longer periods and see whether a pattern is persistent or temporary."),
                ),
            ),
            ui.div(
                {"class": "control-card"},
                ui.layout_columns(
                    ui.input_checkbox_group(
                        "stations_selected",
                        "Select stations",
                        choices=stations,
                        selected=DEFAULT_TREND_STATIONS,
                        inline=False,
                    ),
                    ui.input_selectize(
                        "metric",
                        "Metric",
                        choices={"Rainfall": "Rainfall (mm)", "Temperature": "Temperature (°C)"},
                        selected=DEFAULT_TREND_METRIC,
                    ),
                    ui.input_slider(
                        "year_range",
                        "Year range",
                        min=min(years),
                        max=max(years),
                        value=DEFAULT_YEAR_RANGE,
                        step=1,
                        sep="",
                    ),
                    ui.div(
                        {"class": "reset-wrap"},
                        ui.input_action_button(
                            "reset_trends",
                            "Reset trends",
                            class_="btn btn-outline-primary btn-reset",
                        ),
                    ),
                    col_widths=[3, 3, 4, 2],
                ),
            ),
            ui.output_ui("summary_cards"),
            ui.div(
                {"class": "chart-card"},
                ui.div("Time-series view", class_="chart-title"),
                output_widget("trend_plot"),
            ),
            ui.div(
                {"class": "panel-card"},
                ui.h4("Station summary"),
                ui.p(
                    "A compact station-level summary showing latest level, long-run average, range, and net change across the selected time window.",
                    class_="section-subtitle",
                ),
                ui.output_table("trend_table"),
            ),
        ),
    ),

    ui.nav_panel(
        "Seasonality",
        ui.div(
            {"class": "dashboard-shell"},
            section_title(
                "Seasonality",
                "Compare average monthly climate patterns to see when each station is typically wetter or warmer.",
            ),
            ui.div(
                {"class": "panel-card"},
                ui.h4("How to read seasonality"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Seasonality"), " shows the typical pattern across the calendar year by averaging each month over the full dataset."),
                    ui.tags.li(ui.tags.b("Peak month"), " is the month with the highest average rainfall or temperature for that station."),
                    ui.tags.li(ui.tags.b("Low month"), " is the month with the lowest average value."),
                    ui.tags.li(ui.tags.b("Seasonal range"), " is the difference between the highest and lowest average month, showing how strong the seasonal swing is."),
                    ui.tags.li(ui.tags.b("Use this page"), " to compare whether stations have similar timing but different magnitudes, or completely different seasonal patterns."),
                ),
            ),
            ui.div(
                {"class": "control-card"},
                ui.layout_columns(
                    ui.input_checkbox_group(
                        "stations_season",
                        "Select stations",
                        choices=stations,
                        selected=DEFAULT_SEASON_STATIONS,
                        inline=False,
                    ),
                    ui.input_selectize(
                        "metric_season",
                        "Metric",
                        choices={"Rainfall": "Rainfall (mm)", "Temperature": "Temperature (°C)"},
                        selected=DEFAULT_SEASON_METRIC,
                    ),
                    ui.input_switch("show_points", "Show markers", value=DEFAULT_SHOW_POINTS),
                    ui.div(
                        {"class": "reset-wrap"},
                        ui.input_action_button(
                            "reset_seasonality",
                            "Reset seasonality",
                            class_="btn btn-outline-primary btn-reset",
                        ),
                    ),
                    col_widths=[4, 3, 2, 3],
                ),
            ),
            ui.div(
                {"class": "chart-card"},
                ui.div("Average monthly pattern", class_="chart-title"),
                output_widget("season_plot"),
            ),
            ui.div(
                {"class": "panel-card"},
                ui.h4("Seasonality summary"),
                ui.p(
                    "A compact summary of peak and low months, annual average, and the size of the seasonal swing for each station.",
                    class_="section-subtitle",
                ),
                ui.output_table("season_table"),
            ),
        ),
    ),

    ui.nav_panel(
        "Anomalies",
        ui.div(
            {"class": "dashboard-shell"},
            section_title(
                "Anomalies",
                "See how a selected year compares with each station's own long-run monthly average.",
            ),
            ui.div(
                {"class": "panel-card"},
                ui.h4("How to read anomalies"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.b("Anomaly"), " means the difference between an observed value and that station's usual monthly average."),
                    ui.tags.li(ui.tags.b("Positive rainfall anomaly"), " means that month was wetter than normal for that station."),
                    ui.tags.li(ui.tags.b("Negative rainfall anomaly"), " means that month was drier than normal."),
                    ui.tags.li(ui.tags.b("Positive temperature anomaly"), " means that month was warmer than normal."),
                    ui.tags.li(ui.tags.b("Negative temperature anomaly"), " means that month was cooler than normal."),
                    ui.tags.li(ui.tags.b("Zero line"), " marks the baseline. Bars above zero are above normal; bars below zero are below normal."),
                    ui.tags.li(ui.tags.b("Important"), " anomalies compare each station against its own history, so they show departures from normal rather than direct station-to-station differences."),
                ),
            ),
            ui.div(
                {"class": "control-card"},
                ui.layout_columns(
                    ui.input_selectize(
                        "station_anom",
                        "Select station",
                        choices=stations,
                        selected=DEFAULT_ANOM_STATION,
                    ),
                    ui.input_selectize(
                        "year_anom",
                        "Select year",
                        choices=[str(y) for y in years],
                        selected=DEFAULT_ANOM_YEAR,
                    ),
                    ui.input_selectize(
                        "metric_anom",
                        "Metric",
                        choices={
                            "Rainfall_Anomaly": "Rainfall anomaly (mm)",
                            "Temperature_Anomaly": "Temperature anomaly (°C)",
                        },
                        selected=DEFAULT_ANOM_METRIC,
                    ),
                    ui.div(
                        {"class": "reset-wrap"},
                        ui.input_action_button(
                            "reset_anomalies",
                            "Reset anomalies",
                            class_="btn btn-outline-primary btn-reset",
                        ),
                    ),
                    col_widths=[3, 3, 3, 3],
                ),
            ),
            ui.div(
                {"class": "chart-card"},
                ui.div("Monthly anomaly profile", class_="chart-title"),
                output_widget("anom_plot"),
            ),
            ui.div(
                {"class": "panel-card"},
                ui.h4("Anomaly summary"),
                ui.p(
                    "A short insight table highlighting how many months were above or below baseline and where the strongest deviations occurred.",
                    class_="section-subtitle",
                ),
                ui.output_table("anom_table"),
            ),
        ),
    ),

    title="Ireland Climate Explorer",
    id="page",
    fillable=True,
)


# =========================================================
# Server
# =========================================================
def server(input, output, session):
    @reactive.effect
    @reactive.event(input.reset_overview)
    def _reset_overview():
        ui.update_selectize("station_overview", selected=DEFAULT_OVERVIEW_STATION, session=session)
        ui.update_selectize("compare_overview", selected=DEFAULT_COMPARE_STATION, session=session)
        ui.update_selectize("year_overview", selected=DEFAULT_OVERVIEW_YEAR, session=session)

    @reactive.effect
    @reactive.event(input.reset_trends)
    def _reset_trends():
        ui.update_checkbox_group("stations_selected", selected=DEFAULT_TREND_STATIONS, session=session)
        ui.update_selectize("metric", selected=DEFAULT_TREND_METRIC, session=session)
        ui.update_slider("year_range", value=DEFAULT_YEAR_RANGE, session=session)

    @reactive.effect
    @reactive.event(input.reset_seasonality)
    def _reset_seasonality():
        ui.update_checkbox_group("stations_season", selected=DEFAULT_SEASON_STATIONS, session=session)
        ui.update_selectize("metric_season", selected=DEFAULT_SEASON_METRIC, session=session)
        try:
            ui.update_switch("show_points", value=DEFAULT_SHOW_POINTS, session=session)
        except AttributeError:
            pass

    @reactive.effect
    @reactive.event(input.reset_anomalies)
    def _reset_anomalies():
        ui.update_selectize("station_anom", selected=DEFAULT_ANOM_STATION, session=session)
        ui.update_selectize("year_anom", selected=DEFAULT_ANOM_YEAR, session=session)
        ui.update_selectize("metric_anom", selected=DEFAULT_ANOM_METRIC, session=session)

    @reactive.calc
    def overview_year_data() -> pd.DataFrame:
        year = int(input.year_overview())
        return annual_summary[annual_summary["Year"] == year].copy()

    @reactive.calc
    def filtered_data() -> tuple[pd.DataFrame, str]:
        selected_stations = input.stations_selected()
        metric = input.metric()
        yr_min, yr_max = input.year_range()

        d = df[
            df["Station"].isin(selected_stations)
            & df["Year"].between(yr_min, yr_max)
        ].sort_values("Date").copy()
        return d, metric

    @reactive.calc
    def season_filtered() -> tuple[pd.DataFrame, str]:
        selected_stations = input.stations_season()
        metric = input.metric_season()

        d = df[df["Station"].isin(selected_stations)].copy()
        if d.empty:
            return pd.DataFrame(), metric

        season = (
            d.groupby(["Station", "Month_Num", "Month"], as_index=False)
            .agg(Value=(metric, "mean"))
            .sort_values(["Station", "Month_Num"])
        )
        return season, metric

    @reactive.calc
    def anomaly_filtered() -> tuple[pd.DataFrame, str]:
        station = input.station_anom()
        year = int(input.year_anom())
        metric = input.metric_anom()

        d = df[(df["Station"] == station) & (df["Year"] == year)].sort_values("Month_Num").copy()
        return d, metric

    @output
    @render.ui
    def overview_cards():
        station_a = input.station_overview()
        station_b = input.compare_overview()
        year = int(input.year_overview())

        a = get_annual_row(station_a, year)
        b = get_annual_row(station_b, year)

        if a is None:
            return ui.div({"class": "panel-card"}, "No annual data available for the selected station and year.")

        month_rows = df[(df["Station"] == station_a) & (df["Year"] == year)]
        rain_anom = month_rows["Rainfall_Anomaly"].sum() if not month_rows.empty else np.nan
        temp_anom = month_rows["Temperature_Anomaly"].mean() if not month_rows.empty else np.nan

        if b is not None:
            rain_sub = f"{a['Annual_Rainfall'] - b['Annual_Rainfall']:+.1f} mm vs {station_b}"
            temp_sub = f"{a['Annual_Temperature'] - b['Annual_Temperature']:+.1f} °C vs {station_b}"
        else:
            rain_sub = "Comparison unavailable"
            temp_sub = "Comparison unavailable"

        return ui.div(
            {"class": "metric-grid"},
            metric_card("Annual rainfall", fmt_num(a["Annual_Rainfall"], suffix=" mm"), rain_sub, "accent-blue"),
            metric_card("Mean temperature", fmt_num(a["Annual_Temperature"], suffix=" °C"), temp_sub, "accent-orange"),
            metric_card("Rainfall anomaly", fmt_num(rain_anom, suffix=" mm"), "Against station monthly baseline", "accent-green"),
            metric_card("Temperature anomaly", fmt_num(temp_anom, suffix=" °C"), "Against station monthly baseline", "accent-blue"),
        )

    @output
    @render_widget
    def annual_compare_plot():
        plot_df = overview_year_data()
        year = int(input.year_overview())

        if plot_df.empty:
            return empty_figure()

        fig = px.bar(
            plot_df.sort_values("Annual_Rainfall", ascending=False),
            x="Station",
            y="Annual_Rainfall",
            color="Annual_Temperature",
            text_auto=".0f",
            color_continuous_scale="Blues",
            hover_data={
                "Annual_Rainfall": ":.1f",
                "Annual_Temperature": ":.1f",
                "Station": True,
            },
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title="Rainfall (mm)")
        fig.update_xaxes(title="")
        return apply_common_layout(fig, title=f"Annual rainfall by station ({year})", height=460)

    @output
    @render.ui
    def overview_insight():
        station_a = input.station_overview()
        station_b = input.compare_overview()
        year = int(input.year_overview())

        a = get_annual_row(station_a, year)
        b = get_annual_row(station_b, year)

        if a is None or b is None:
            return ui.p("Not enough full-year data is available for a comparison summary.")

        wetter = station_a if a["Annual_Rainfall"] > b["Annual_Rainfall"] else station_b
        wetter_diff = abs(a["Annual_Rainfall"] - b["Annual_Rainfall"])
        warmer = station_a if a["Annual_Temperature"] > b["Annual_Temperature"] else station_b
        warmer_diff = abs(a["Annual_Temperature"] - b["Annual_Temperature"])

        return ui.div(
            ui.p(ui.tags.b(f"{year} summary")),
            ui.p(f"{wetter} was wetter by {wetter_diff:.1f} mm of annual rainfall."),
            ui.p(f"{warmer} was warmer by {warmer_diff:.1f} °C on annual mean temperature."),
            ui.p("Use this top-level comparison to identify the biggest regional differences before drilling into long-term trends or anomalies."),
        )

    @output
    @render_widget
    def station_map():
        map_df = overview_year_data().merge(STATION_COORDS, on="Station", how="left")
        map_df = map_df.dropna(subset=["lat", "lon", "Annual_Rainfall", "Annual_Temperature"])

        if map_df.empty:
            return empty_figure()

        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="Annual_Rainfall",
            color="Annual_Temperature",
            hover_name="Station",
            hover_data={
                "Annual_Rainfall": ":.1f",
                "Annual_Temperature": ":.1f",
                "Region": True,
                "lat": False,
                "lon": False,
            },
            zoom=5.4,
            center={"lat": 53.4, "lon": -8.15},
            height=520,
            size_max=34,
            color_continuous_scale="Turbo",
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="white",
        )
        return fig

    @output
    @render.ui
    def summary_cards():
        d, metric = filtered_data()
        if d.empty:
            return ui.div({"class": "panel-card"}, "No data available for the selected filters.")

        avg_value = d[metric].mean()
        min_value = d[metric].min()
        max_value = d[metric].max()
        latest_date = d["Date"].max().strftime("%Y-%m")

        suffix = " mm" if metric == "Rainfall" else " °C"
        return ui.div(
            {"class": "metric-grid", "style": "grid-template-columns: repeat(4, minmax(0, 1fr));"},
            metric_card("Average value", fmt_num(avg_value, suffix=suffix), f"Across selected stations ({metric.lower()})", "accent-blue"),
            metric_card("Minimum value", fmt_num(min_value, suffix=suffix), f"Lowest observed {metric.lower()}", "accent-green"),
            metric_card("Maximum value", fmt_num(max_value, suffix=suffix), f"Highest observed {metric.lower()}", "accent-orange"),
            metric_card("Latest month", latest_date, "Most recent month in filtered view", "accent-blue"),
        )

    @output
    @render_widget
    def trend_plot():
        d, metric = filtered_data()
        if d.empty:
            return empty_figure()

        y_title = "Rainfall (mm)" if metric == "Rainfall" else "Temperature (°C)"
        fig = px.line(
            d,
            x="Date",
            y=metric,
            color="Station",
            markers=True,
            line_shape="linear",
            hover_data={metric: ":.1f", "Date": "|%Y-%m", "Station": True},
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        fig.update_xaxes(title="")
        fig.update_yaxes(title=y_title)
        return apply_common_layout(fig, title=f"{metric} over time")

    @output
    @render.table
    def trend_table():
        d, metric = filtered_data()
        if d.empty:
            return pd.DataFrame({"Message": ["No data available"]})

        summary = (
            d.sort_values("Date")
            .groupby("Station", as_index=False)
            .agg(
                Latest_Value=(metric, "last"),
                Average_Value=(metric, "mean"),
                Minimum_Value=(metric, "min"),
                Maximum_Value=(metric, "max"),
                First_Value=(metric, "first"),
                Last_Month=("Date", "last"),
            )
        )
        summary["Change"] = summary["Latest_Value"] - summary["First_Value"]
        summary = summary.drop(columns=["First_Value"])
        summary["Last_Month"] = pd.to_datetime(summary["Last_Month"]).dt.strftime("%Y-%m")
        for col in ["Latest_Value", "Average_Value", "Minimum_Value", "Maximum_Value", "Change"]:
            summary[col] = summary[col].round(1)

        label = "Rainfall (mm)" if metric == "Rainfall" else "Temperature (°C)"
        return summary.rename(
            columns={
                "Latest_Value": f"Latest {label}",
                "Average_Value": f"Average {label}",
                "Minimum_Value": f"Min {label}",
                "Maximum_Value": f"Max {label}",
                "Change": f"Change in {label}",
                "Last_Month": "Latest month",
            }
        ).sort_values(f"Latest {label}", ascending=False).reset_index(drop=True)

    @output
    @render_widget
    def season_plot():
        season, metric = season_filtered()
        if season.empty:
            return empty_figure()

        y_title = "Rainfall (mm)" if metric == "Rainfall" else "Temperature (°C)"
        fig = px.line(
            season,
            x="Month",
            y="Value",
            color="Station",
            markers=bool(input.show_points()),
            category_orders={"Month": MONTH_ORDER},
            hover_data={"Value": ":.1f", "Station": True},
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        fig.update_xaxes(title="")
        fig.update_yaxes(title=y_title)
        return apply_common_layout(fig, title=f"Average monthly {metric.lower()} by station")

    @output
    @render.table
    def season_table():
        season, metric = season_filtered()
        if season.empty:
            return pd.DataFrame({"Message": ["No data available"]})

        rows = []
        for station, grp in season.groupby("Station"):
            grp = grp.sort_values("Month_Num").reset_index(drop=True)
            peak = grp.loc[grp["Value"].idxmax()]
            trough = grp.loc[grp["Value"].idxmin()]
            rows.append(
                {
                    "Station": station,
                    "Peak month": peak["Month"],
                    "Peak value": round(float(peak["Value"]), 1),
                    "Low month": trough["Month"],
                    "Low value": round(float(trough["Value"]), 1),
                    "Annual average": round(float(grp["Value"].mean()), 1),
                    "Seasonal range": round(float(grp["Value"].max() - grp["Value"].min()), 1),
                }
            )

        out = pd.DataFrame(rows)
        unit = "mm" if metric == "Rainfall" else "°C"
        return out.rename(
            columns={
                "Peak value": f"Peak value ({unit})",
                "Low value": f"Low value ({unit})",
                "Annual average": f"Annual average ({unit})",
                "Seasonal range": f"Seasonal range ({unit})",
            }
        ).sort_values(f"Seasonal range ({unit})", ascending=False).reset_index(drop=True)

    @output
    @render_widget
    def anom_plot():
        d, metric = anomaly_filtered()
        if d.empty:
            return empty_figure()

        title_text = "Rainfall anomaly" if metric == "Rainfall_Anomaly" else "Temperature anomaly"
        color_scale = "RdBu" if metric == "Temperature_Anomaly" else "BrBG"
        fig = px.bar(
            d,
            x="Month",
            y=metric,
            color=metric,
            category_orders={"Month": MONTH_ORDER},
            color_continuous_scale=color_scale,
            hover_data={metric: ":.1f", "Month": True},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)
        fig.update_xaxes(title="")
        fig.update_yaxes(title=title_text)
        fig.update_layout(coloraxis_showscale=False)
        return apply_common_layout(fig, title=f"{title_text.title()} by month")

    @output
    @render.table
    def anom_table():
        d, metric = anomaly_filtered()
        if d.empty:
            return pd.DataFrame({"Message": ["No data available"]})

        pos = d[d[metric] > 0]
        neg = d[d[metric] < 0]
        peak_pos = pos.loc[pos[metric].idxmax()] if not pos.empty else None
        peak_neg = neg.loc[neg[metric].idxmin()] if not neg.empty else None

        summary_rows = [
            {
                "Insight": "Positive anomaly months",
                "Value": int((d[metric] > 0).sum()),
            },
            {
                "Insight": "Negative anomaly months",
                "Value": int((d[metric] < 0).sum()),
            },
            {
                "Insight": "Strongest positive month",
                "Value": f"{peak_pos['Month']} ({peak_pos[metric]:+.1f})" if peak_pos is not None else "None",
            },
            {
                "Insight": "Strongest negative month",
                "Value": f"{peak_neg['Month']} ({peak_neg[metric]:+.1f})" if peak_neg is not None else "None",
            },
            {
                "Insight": "Annual anomaly summary",
                "Value": round(float(d[metric].sum()), 1) if metric == "Rainfall_Anomaly" else round(float(d[metric].mean()), 1),
            },
        ]

        unit = "mm" if metric == "Rainfall_Anomaly" else "°C"
        out = pd.DataFrame(summary_rows)
        out.loc[out["Insight"] == "Annual anomaly summary", "Value"] = out.loc[
            out["Insight"] == "Annual anomaly summary", "Value"
        ].astype(str) + f" {unit}"
        return out


app = App(app_ui, server)