import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List
from math import ceil

app = FastAPI(
    title="Biodiversity Action Plan Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / "data"
XLSX_FILE_NAME = "Biodiversity Action Plan Master copy.xlsx"
_cached_df = None

SPECIES_COLUMNS = [
    'all', 'invertebrate', 'bat', 'mammal', 'bird', 'amphibians',
    'reptiles', 'invasive', 'plants', 'freshwater', 'coastal'
]

def clean_column_names(df):
    """Standardizes column names for easier use."""
    cols = df.columns
    new_cols = [col.strip() for col in cols]
    df.columns = new_cols
    df = df.rename(columns={
        "Priority 2025 (3 high, 1 low)": "priority_2025",
        "Lead Contact  (provisional)": "lead_contact",
        "Implementation ranking": "implementation_ranking"
    })
    return df

def get_dataframe() -> pd.DataFrame:
    """
    Loads the XLSX file into a pandas DataFrame, cleans it, and caches it.
    """
    global _cached_df
    if _cached_df is not None:
        return _cached_df

    file_path = DATA_PATH / XLSX_FILE_NAME
    if not file_path.exists():
        raise HTTPException(status_code=500, detail=f"Data file not found: {XLSX_FILE_NAME}")

    try:
        df = pd.read_excel(file_path, engine='openpyxl', skiprows=1)
        df = clean_column_names(df)
        df.dropna(subset=['Action'], inplace=True)

        if 'Status' in df.columns:
            df['Status'] = df['Status'].str.strip().str.lower().str.capitalize()
            df['Status'] = df['Status'].replace({'Achieved and ongoing': 'Achieved and ongoing'})

        def get_affected_species(row):
            species_list = [col for col in SPECIES_COLUMNS if row.get(col) == 1]
            return species_list
        
        df['affected_species'] = df.apply(get_affected_species, axis=1)
        
        _cached_df = df
        print("Data loaded and cached successfully.")
        return _cached_df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {e}")

def apply_filters(
    df: pd.DataFrame,
    status: Optional[str] = None,
    strategy: Optional[str] = None,
    timescale: Optional[str] = None,
    implementation: Optional[str] = None,
    impact: Optional[str] = None,
    priority: Optional[str] = None,
    species: Optional[str] = None,
) -> pd.DataFrame:
    """Applies all active filters to the DataFrame."""
    query_df = df.copy()
    if status:
        query_df = query_df[query_df['Status'] == status]
    if strategy:
        query_df = query_df[query_df['Strategy 2025'] == strategy]
    if timescale:
        query_df = query_df[query_df['Timescale'] == timescale]
    if implementation:
        query_df = query_df[query_df['implementation_ranking'] == implementation]
    if impact:
        query_df = query_df[query_df['Impact'] == impact]
    if priority:
        try:
            query_df = query_df[query_df['priority_2025'] == float(priority)]
        except (ValueError, TypeError):
            pass
            
    if species:
        selected_species = species.split(',')
        query_df = query_df[query_df['affected_species'].apply(lambda lst: any(s in lst for s in selected_species))]

    return query_df

def get_summary_stats(df: pd.DataFrame) -> dict:
    """Calculates summary statistics for the given DataFrame."""
    total_actions = len(df)
    status_counts = df['Status'].value_counts().to_dict()

    for status_key in ['Achieved and ongoing', 'Underway', 'Not started']:
        if status_key not in status_counts:
            status_counts[status_key] = 0

    status_percentages = {
        key: round((value / total_actions) * 100, 1) if total_actions > 0 else 0
        for key, value in status_counts.items()
    }

    return {
        "total_actions": total_actions,
        "status_counts": status_counts,
        "status_percentages": status_percentages,
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Biodiversity Action Plan API"}

@app.get("/api/filter-options")
def get_filter_options():
    """Returns a dictionary of unique values for each filter dropdown."""
    df = get_dataframe()
    options = {
        "status": sorted(df['Status'].dropna().unique().tolist()),
        "strategy": sorted(df['Strategy 2025'].dropna().unique().tolist()),
        "timescale": sorted(df['Timescale'].dropna().unique().tolist()),
        "implementation": sorted(df['implementation_ranking'].dropna().unique().tolist()),
        "impact": sorted(df['Impact'].dropna().unique().tolist()),
        "priority": sorted([str(int(p)) for p in df['priority_2025'].dropna().unique()]),
        "species": SPECIES_COLUMNS,
    }
    return options

@app.get("/api/actions")
def get_actions(
    status: Optional[str] = None,
    strategy: Optional[str] = None,
    timescale: Optional[str] = None,
    implementation: Optional[str] = None,
    impact: Optional[str] = None,
    priority: Optional[str] = None,
    species: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
):
    """
    Returns the filtered and paginated list of actions and dynamic summary statistics.
    """
    df = get_dataframe()
    filtered_df = apply_filters(df, status, strategy, timescale, implementation, impact, priority, species)
    
    summary_stats = get_summary_stats(filtered_df)
    
    total_records = len(filtered_df)
    total_pages = ceil(total_records / page_size)
    
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_df = filtered_df.iloc[start_index:end_index]

    df_to_send = paginated_df.drop(columns=SPECIES_COLUMNS, errors='ignore')
    cleaned_df = df_to_send.astype(object).where(pd.notnull(df_to_send), None)
    actions_list = cleaned_df.to_dict(orient='records')

    return {
        "summary_stats": summary_stats,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
        },
        "actions": actions_list,
    }