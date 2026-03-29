import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget


# Data loading and preparation
APP_DIR = Path(__file__).parent
DATA_FILE = APP_DIR / "irish_climate_data.csv"
folder_path = APP_DIR

all_data = []

# Build CSV from JSON files
for file in os.listdir(folder_path):
    if file.endswith(".json"):
        station_name = file.replace(".json", "")

        with open(os.path.join(folder_path, file), encoding="utf-8") as f:
            data = json.load(f)

        rainfall = data["total_rainfall"]["report"]
        temp = data["mean_temperature"]["report"]

        for year in rainfall:
            if year == "LTA":
                continue

            for month in rainfall[year]:
                if month == "annual":
                    continue

                rain_val = rainfall[year][month]
                temp_val = temp.get(year, {}).get(month, None)

                if rain_val in ["", None] or temp_val in ["", None]:
                    continue

                all_data.append(
                    {
                        "Station": station_name,
                        "Year": int(year),
                        "Month": month.capitalize(),
                        "Rainfall": float(rain_val),
                        "Temperature": float(temp_val),
                    }
                )

df = pd.DataFrame(all_data)
df.to_csv(DATA_FILE, index=False)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"CSV not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

# Standardise month names
month_map = {
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

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
month_num_map = {m: i + 1 for i, m in enumerate(month_order)}

# Clean station names
df["Station"] = df["Station"].astype(str).str.strip().str.lower()
df["Station"] = df["Station"].replace(
    {
        "athenry": "Athenry",
        "belmullet": "Belmullet",
        "cork airport": "Cork Airport",
        "dublin airport": "Dublin Airport",
        "shannon airport": "Shannon Airport",
    }
)

# Clean month names
df["Month"] = df["Month"].astype(str).str.strip().replace(month_map)

# Convert numeric columns safely
for col in ["Year", "Rainfall", "Temperature"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create month number and date
df["Month_Num"] = df["Month"].map(month_num_map)
df["Date"] = pd.to_datetime(
    dict(year=df["Year"], month=df["Month_Num"], day=1),
    errors="coerce",
)

# Replace bad numeric values and drop incomplete rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=["Station", "Year", "Month", "Month_Num", "Date", "Rainfall", "Temperature"]
).copy()

df["Year"] = df["Year"].astype(int)
df["Month_Num"] = df["Month_Num"].astype(int)

# Monthly baseline and anomalies
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

for col in ["Avg_Rainfall", "Avg_Temperature", "Rainfall_Anomaly", "Temperature_Anomaly"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# Annual summaries (full years only)
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
)

annual_summary = annual_summary.replace([np.inf, -np.inf], np.nan).dropna(
    subset=[
        "Station",
        "Year",
        "Annual_Rainfall",
        "Annual_Temperature",
        "Annual_Rainfall_Anomaly",
        "Annual_Temperature_Anomaly",
    ]
).copy()

latest_full_year = (
    int(annual_summary["Year"].max())
    if not annual_summary.empty
    else int(df["Year"].max())
)

# UI-ready variables and map data
stations = sorted(df["Station"].unique().tolist())
years = sorted(df["Year"].unique().tolist())

station_coords = pd.DataFrame(
    [
        {"Station": "Dublin Airport", "lat": 53.4264, "lon": -6.2499, "Region": "East"},
        {"Station": "Cork Airport", "lat": 51.8472, "lon": -8.4911, "Region": "South"},
        {"Station": "Athenry", "lat": 53.2964, "lon": -8.7431, "Region": "West"},
        {"Station": "Belmullet", "lat": 54.2253, "lon": -10.0064, "Region": "Northwest"},
        {"Station": "Shannon Airport", "lat": 52.7020, "lon": -8.9248, "Region": "Mid-West"},
    ]
)


# Helper functions
def fmt_num(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}"


def metric_card(title, value, subtitle="", accent_class="accent-blue"):
    return ui.div(
        {"class": f"metric-card {accent_class}"},
        ui.div(title, class_="metric-title"),
        ui.div(value, class_="metric-value"),
        ui.div(subtitle, class_="metric-subtitle"),
    )


def empty_figure(message="No data available for this selection"):
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
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
    )
    return fig


# UI
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Overview",
        ui.div(
            ui.h3("Overview"),
            ui.p("This section will contain high-level climate summaries, comparisons, and map insights."),
        ),
    ),

    ui.nav_panel(
        "Trends",
        ui.div(
            ui.h3("Trends"),

            ui.layout_columns(
                ui.input_checkbox_group(
                    "stations_selected",
                    "Select stations",
                    choices=stations,
                    selected=stations[:2],
                    inline=False,
                ),
                ui.input_selectize(
                    "metric",
                    "Metric",
                    choices=["Rainfall", "Temperature"],
                    selected="Rainfall",
                ),
                ui.input_slider(
                    "year_range",
                    "Year range",
                    min=min(years),
                    max=max(years),
                    value=(min(years), max(years)),
                    step=1,
                    sep="",
                ),
                col_widths=[4, 4, 4],
            ),

            ui.output_ui("summary_cards"),
            output_widget("trend_plot"),
            ui.h4("Recent values"),
            ui.output_table("trend_table"),
        ),
    ),

    ui.nav_panel(
        "Seasonality",
        ui.div(
            ui.h3("Seasonality"),
            ui.p("This section will show average monthly climate patterns by station."),
        ),
    ),

    ui.nav_panel(
        "Anomalies",
        ui.div(
            ui.h3("Anomalies"),
            ui.p("This section will show rainfall and temperature anomalies."),
        ),
    ),

    title="Ireland Climate Explorer",
    id="page",
)


# Server
def server(input, output, session):
    @reactive.calc
    def filtered_data():
        selected_stations = input.stations_selected()
        metric = input.metric()
        yr_min, yr_max = input.year_range()

        d = df[df["Station"].isin(selected_stations)].copy()
        d = d[(d["Year"] >= yr_min) & (d["Year"] <= yr_max)].copy()
        d = d.sort_values("Date")
        return d, metric

    @output
    @render.ui
    def summary_cards():
        d, metric = filtered_data()

        if d.empty:
            return ui.div("No data available for selected filters.")

        avg_value = d[metric].mean()
        min_value = d[metric].min()
        max_value = d[metric].max()

        return ui.div(
            {
                "style": (
                    "display:grid; grid-template-columns: repeat(3, 1fr); "
                    "gap: 1rem; margin: 1rem 0;"
                )
            },
            metric_card(
                "Average value",
                fmt_num(avg_value),
                f"Across selected stations ({metric})",
                "accent-blue",
            ),
            metric_card(
                "Minimum value",
                fmt_num(min_value),
                f"Lowest observed {metric.lower()}",
                "accent-green",
            ),
            metric_card(
                "Maximum value",
                fmt_num(max_value),
                f"Highest observed {metric.lower()}",
                "accent-orange",
            ),
        )

    @output
    @render_widget
    def trend_plot():
        d, metric = filtered_data()

        if d.empty:
            return empty_figure()

        fig = px.line(
            d,
            x="Date",
            y=metric,
            color="Station",
            markers=True,
            title=f"{metric} over time",
        )

        fig.update_layout(
            template="plotly_white",
            height=500,
            legend_title="Station",
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title=metric)
        return fig

    @output
    @render.table
    def trend_table():
        d, metric = filtered_data()

        if d.empty:
            return pd.DataFrame({"Message": ["No data available"]})

        out = d[["Date", "Station", metric]].copy()
        out["Date"] = out["Date"].dt.strftime("%Y-%m")
        return out.tail(15).reset_index(drop=True)


app = App(app_ui, server)