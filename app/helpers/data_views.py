# helpers/data_views.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# We use PyArrow to inspect the schema safely (works for files or directory datasets)
def _as_str_path(p: Path | str) -> str:
    return str(p if isinstance(p, Path) else Path(p))

@st.cache_data(ttl=600, show_spinner=False)
def parquet_columns(processed_path: Path) -> list[str]:
    """
    Return the list of column names in the Parquet file or directory.
    Falls back gracefully across file/dataset cases.
    """
    path_str = _as_str_path(processed_path)
    try:
        # Directory dataset or single file – dataset schema handles both
        import pyarrow.dataset as ds
        schema = ds.dataset(path_str).schema
        return list(schema.names)
    except Exception:
        # Fallback to ParquetFile for single-file case
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path_str)
            return list(pf.schema_arrow.names)
        except Exception:
            # Last resort: try reading a tiny sample with pandas to infer columns
            try:
                df = pd.read_parquet(path_str)
                return list(df.columns)
            except Exception:
                return []


def _safe_read_parquet(processed_path: Path, desired_cols: list[str]) -> pd.DataFrame:
    """
    Read only existing columns from Parquet to avoid ArrowInvalid when a requested column is missing.
    If *none* of the desired columns exist, return an empty DataFrame (no crash).
    """
    exist = set(parquet_columns(processed_path))
    use_cols = [c for c in desired_cols if c in exist]

    if not use_cols:
        # Return empty frame – caller logic should handle empty results gracefully.
        return pd.DataFrame(columns=[])

    return pd.read_parquet(_as_str_path(processed_path), columns=use_cols)


# ---- Team list (display names) ----
@st.cache_data(ttl=600, show_spinner=False)
def list_team_choices(processed_path: Path, college_map: dict[str, str]) -> list[str]:
    df = _safe_read_parquet(processed_path, ["college"])
    if df.empty or "college" not in df.columns:
        return []
    df = df.dropna(subset=["college"]).copy()
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    disp = df[" college_norm"].map(college_map).fillna(df["college"].astype(str))
    teams = sorted(disp.dropna().unique().tolist())
    return teams


# ---- Seasons for a given team (norm key) ----
@st.cache_data(ttl=600, show_spinner=False)
def list_seasons_for_team(processed_path: Path, college_norm: str) -> list[str]:
    df = _safe_read_parquet(processed_path, ["college", "season"])
    if df.empty or not {"college", "season"}.issubset(df.columns):
        return []
    df = df.dropna(subset=["college", "season"]).copy()
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    seasons = sorted(
        df.loc[df[" college_norm"] == str(college_norm), "season"]
          .astype(str)
          .unique()
          .tolist()
    )
    return seasons


# ---- Pruned roster slice for team+season ----
@st.cache_data(ttl=600, show_spinner=False)
def team_season_view(processed_path: Path, college_norm: str, season: str) -> pd.DataFrame:
    cols = [
        "college","season","player_ind","player_number_ind","position","Archetype",
        "minutes_tot_ind","mins_per_game","pts_per_game","ast_per_game","reb_per_game",
        "stl_per_game","blk_per_game","eFG_pct","USG_pct","gp_ind"
    ]
    df = _safe_read_parquet(processed_path, cols)
    if df.empty:
        return df  # empty; caller will handle gracefully

    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower() if "college" in df.columns else ""
    if "season" in df.columns:
        df = df[df["season"].astype(str) == str(season)]
    if " college_norm" in df.columns:
        df = df[df[" college_norm"] == str(college_norm)]
    return df.copy()


# ---- Fetch rows for compare cart (by triples) ----
@st.cache_data(ttl=600, show_spinner=False)
def rows_for_compare(processed_path: Path, items: list[dict]) -> list[pd.Series]:
    """
    items: [{"player_ind": "...", "season": "...", "college": "..."}]
    Returns list of matching rows in the same order as items (first match per triple).
    """
    if not items:
        return []

    want = pd.DataFrame(items, columns=["player_ind", "season", "college"]).dropna()
    if want.empty:
        return []

    want["player_ind"] = want["player_ind"].astype(str)
    want["season"] = want["season"].astype(str)
    want["college"] = want["college"].astype(str)

    cols = [
        "player_ind","season","college","position","Archetype",
        "pts_per_game","reb_per_game","ast_per_game","stl_per_game","blk_per_game",
        "eFG_pct","USG_pct","mins_per_game","minutes_tot_ind","player_number_ind"
    ]
    df = _safe_read_parquet(processed_path, cols)
    if df.empty:
        return []

    # Ensure string compare compatibility
    for c in ["player_ind", "season", "college"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Merge to preserve order and 1:1 mapping
    merged = want.merge(df, on=["player_ind", "season", "college"], how="left", suffixes=("", "_df"))

    out: list[pd.Series] = []
    for _, row in want.iterrows():
        hit = merged[
            (merged["player_ind"] == row["player_ind"]) &
            (merged["season"] == row["season"]) &
            (merged["college"] == row["college"])
        ]
        if not hit.empty:
            out.append(hit.iloc[0])
    return out
