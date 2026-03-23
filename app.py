import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Folder where your JSON files are
folder_path = r"C:\Users\killi\OneDrive\Data Vis\CA2\WeatherApp"

all_data = []

# Loop through each file
for file in os.listdir(folder_path):
    if file.endswith(".json"):
        station_name = file.replace(".json", "")
        
        with open(os.path.join(folder_path, file)) as f:
            data = json.load(f)
        
        # Extract rainfall and temperature
        rainfall = data["total_rainfall"]["report"]
        temp = data["mean_temperature"]["report"]
        
        # Loop through years
        for year in rainfall:
            if year == "LTA":
                continue
            
            for month in rainfall[year]:
                if month in ["annual"]:
                    continue
                
                rain_val = rainfall[year][month]
                temp_val = temp[year].get(month, None)
                
                # Skip missing values
                if rain_val == "" or temp_val == "":
                    continue
                
                all_data.append({
                    "Station": station_name,
                    "Year": int(year),
                    "Month": month.capitalize(),
                    "Rainfall": float(rain_val),
                    "Temperature": float(temp_val)
                })

# Create DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
df.to_csv("irish_climate_data.csv", index=False)

print("CSV file created successfully!")

APP_DIR = Path(__file__).parent
DATA_FILE = APP_DIR / "irish_climate_data.csv"

print("Looking for:", DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"CSV not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("Columns found:")
print(df.columns.tolist())
print(df.head())

# Clean and prepare the dataset

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
    "July", "August", "September", "October", "November", "December"
]

month_num_map = {m: i + 1 for i, m in enumerate(month_order)}

# Clean station names
df["Station"] = df["Station"].astype(str).str.strip()

# Clean month names
df["Month"] = df["Month"].astype(str).str.strip().replace(month_map)

# Convert numeric columns safely
for col in ["Year", "Rainfall", "Temperature"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create month number
df["Month_Num"] = df["Month"].map(month_num_map)

# Create proper date column
df["Date"] = pd.to_datetime(
    dict(year=df["Year"], month=df["Month_Num"], day=1),
    errors="coerce"
)

# Replace bad numeric values
df = df.replace([np.inf, -np.inf], np.nan)

# Drop incomplete rows
df = df.dropna(subset=["Station", "Year", "Month", "Month_Num", "Date", "Rainfall", "Temperature"]).copy()

# Convert types
df["Year"] = df["Year"].astype(int)
df["Month_Num"] = df["Month_Num"].astype(int)

print("\nCleaned data preview:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nRows, Columns:", df.shape)

# Monthly baseline and anomalies

monthly_baseline = (
    df.groupby(["Station", "Month_Num", "Month"], as_index=False)
    .agg(
        Avg_Rainfall=("Rainfall", "mean"),
        Avg_Temperature=("Temperature", "mean"),
    )
)

# Merge baseline back into main dataframe
df = df.merge(
    monthly_baseline,
    on=["Station", "Month_Num", "Month"],
    how="left"
)

# Create anomaly columns
df["Rainfall_Anomaly"] = df["Rainfall"] - df["Avg_Rainfall"]
df["Temperature_Anomaly"] = df["Temperature"] - df["Avg_Temperature"]

# Clean anomaly columns
for col in ["Avg_Rainfall", "Avg_Temperature", "Rainfall_Anomaly", "Temperature_Anomaly"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

print("\nMonthly baseline preview:")
print(monthly_baseline.head())

print("\nData with anomalies preview:")
print(df[[
    "Station", "Year", "Month", "Rainfall", "Avg_Rainfall", "Rainfall_Anomaly",
    "Temperature", "Avg_Temperature", "Temperature_Anomaly"
]].head())

# Annual summaries (full years only)

# Count how many months exist for each station-year
annual_counts = (
    df.groupby(["Station", "Year"], as_index=False)
    .agg(month_count=("Month_Num", "nunique"))
)

# Keep only years with all 12 months
full_years = annual_counts.loc[
    annual_counts["month_count"] == 12,
    ["Station", "Year"]
]

# Build annual summary table
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

annual_summary = annual_summary.replace([np.inf, -np.inf], np.nan)
annual_summary = annual_summary.dropna(
    subset=[
        "Station",
        "Year",
        "Annual_Rainfall",
        "Annual_Temperature",
        "Annual_Rainfall_Anomaly",
        "Annual_Temperature_Anomaly",
    ]
).copy()

latest_full_year = int(annual_summary["Year"].max()) if not annual_summary.empty else int(df["Year"].max())

print("\nAnnual counts preview:")
print(annual_counts.head())

print("\nFull years preview:")
print(full_years.head())

print("\nAnnual summary preview:")
print(annual_summary.head())

print("\nLatest full year:", latest_full_year)