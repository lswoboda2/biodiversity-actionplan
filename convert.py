import pandas as pd
from pathlib import Path

print("Starting Excel to Parquet conversion...")

DATA_DIR = Path(__file__).parent / "data"
EXCEL_FILE = DATA_DIR / "source.xlsx"
PARQUET_FILE = DATA_DIR / "source.parquet"

if not EXCEL_FILE.exists():
    print(f"Error: Source file not found at {EXCEL_FILE}")
    exit(1)


df = pd.read_excel(EXCEL_FILE, engine="openpyxl", skiprows=1)

if 'Notes 2026' in df.columns:
    df['Notes 2026'] = df['Notes 2026'].astype(str)

df.to_parquet(PARQUET_FILE, index=False)

print(f"Successfully converted {EXCEL_FILE} to {PARQUET_FILE}")