from pathlib import Path
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).parent
DATA_FILE = APP_DIR / "irish_climate_data.csv"

print("Looking for:", DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"CSV not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("Columns found:")
print(df.columns.tolist())
print(df.head())