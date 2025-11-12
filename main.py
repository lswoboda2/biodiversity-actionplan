import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
from math import ceil

app = FastAPI(title="Biodiversity Action Plan Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PARQUET = Path(__file__).parent / "data" / "source.parquet"
_cached_df = None

SPECIES_COLUMNS = [
    "all", "invertebrate", "bat", "mammal", "bird", "amphibians",
    "reptiles", "invasive", "plants", "freshwater", "coastal"
]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns={
        "Priority 2025 (3 high, 1 low)": "priority_2025",
        "Lead Contact  (provisional)": "lead_contact",
        "Implementation ranking": "implementation_ranking",
    })
    if "priority_2025" in df.columns:
        df["priority_2025"] = pd.to_numeric(df["priority_2025"], errors="coerce")
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all cleaning steps on the raw dataframe."""
    df = clean_column_names(df)
    
    if "Action" in df.columns:
        df = df.dropna(subset=["Action"])
    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.strip().str.lower().str.capitalize()

    for col in SPECIES_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0 
            
    species_matrix = df[SPECIES_COLUMNS].values == 1
    species_names = np.array(SPECIES_COLUMNS)
    df["affected_species"] = [list(species_names[row]) for row in species_matrix]
    
    return df

def _load_dataframe_from_disk() -> pd.DataFrame:
    if not DATA_PARQUET.exists():
        raise RuntimeError(f"Data file not found: {DATA_PARQUET.name}.")
    
    df = pd.read_parquet(DATA_PARQUET)
    return df

@app.on_event("startup")
def load_data_on_startup():
    global _cached_df
    try:
        raw_df = _load_dataframe_from_disk()
        _cached_df = preprocess_dataframe(raw_df)
        print(f"Successfully loaded and processed {DATA_PARQUET.name} into cache.")
    except Exception as e:
        print(f"FATAL: Failed to load data on startup: {e}")
        _cached_df = None

def get_dataframe() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Data is not loaded.")
    return _cached_df

@app.head("/")
def read_root_head():
    return Response(status_code=200)

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
    q = df.copy()
    if status:
        q = q[q.get("Status") == status]
    if strategy:
        q = q[q.get("Strategy 2025") == strategy]
    if timescale:
        q = q[q.get("Timescale") == timescale]
    if implementation:
        q = q[q.get("implementation_ranking") == implementation]
    if impact:
        q = q[q.get("Impact") == impact]
    if priority:
        try:
            q = q[q.get("priority_2025") == float(priority)]
        except (ValueError, TypeError):
            pass
            
    if species:
        selected = [s.strip() for s in species.split(",") if s.strip()]
        if selected:
            valid_selected_species = [s for s in selected if s in SPECIES_COLUMNS and s in q.columns]
            
            if valid_selected_species:
                query_str = " | ".join([f"`{s}` == 1" for s in valid_selected_species])
                q = q.query(query_str)
    return q

def get_summary_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    counts = df["Status"].value_counts().to_dict() if "Status" in df.columns else {}
    for k in ["Achieved and ongoing", "Underway", "Not started"]:
        counts.setdefault(k, 0)
    pct = {k: round((v / total) * 100, 1) if total else 0 for k, v in counts.items()}
    return {"total_actions": total, "status_counts": counts, "status_percentages": pct}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Biodiversity Action Plan API"}

@app.get("/api/filter-options")
def get_filter_options():
    df = get_dataframe()
    opt = {
        "status": sorted(df["Status"].dropna().unique().tolist()) if "Status" in df.columns else [],
        "strategy": sorted(df["Strategy 2025"].dropna().unique().tolist()) if "Strategy 2025" in df.columns else [],
        "timescale": sorted(df["Timescale"].dropna().unique().tolist()) if "Timescale" in df.columns else [],
        "implementation": sorted(df["implementation_ranking"].dropna().unique().tolist()) if "implementation_ranking" in df.columns else [],
        "impact": sorted(df["Impact"].dropna().unique().tolist()) if "Impact" in df.columns else [],
        "priority": sorted([str(int(p)) for p in df["priority_2025"].dropna().unique()]) if "priority_2025" in df.columns else [],
        "species": SPECIES_COLUMNS,
    }
    return opt

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
    df = get_dataframe()
    filtered = apply_filters(df, status, strategy, timescale, implementation, impact, priority, species)
    stats = get_summary_stats(filtered)

    total_records = len(filtered)
    total_pages = ceil(total_records / page_size) if page_size > 0 else 1
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    page_df = filtered.iloc[start:end] if page_size > 0 else filtered

    out_df = page_df.drop(columns=SPECIES_COLUMNS, errors="ignore")
    cleaned = out_df.astype(object).where(pd.notnull(out_df), None)
    actions = cleaned.to_dict(orient="records")

    return {
        "summary_stats": stats,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
        },
        "actions": actions,
    }