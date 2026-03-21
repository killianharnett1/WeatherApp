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