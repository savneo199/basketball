# helpers/data_views.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# ---- Team list (display names) ----
@st.cache_data(ttl=600)
def list_team_choices(processed_path: Path, college_map: dict[str, str]) -> list[str]:
    cols = ["college"]
    df = pd.read_parquet(processed_path, columns=cols).dropna(subset=["college"])
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    # map norm -> display if available; else show raw college
    disp = df[" college_norm"].map(college_map).fillna(df["college"].astype(str))
    teams = sorted(disp.dropna().unique().tolist())
    return teams

# ---- Seasons for a given team (norm key) ----
@st.cache_data(ttl=600)
def list_seasons_for_team(processed_path: Path, college_norm: str) -> list[str]:
    cols = ["college", "season"]
    df = pd.read_parquet(processed_path, columns=cols).dropna(subset=["college", "season"])
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    seasons = sorted(df.loc[df[" college_norm"] == str(college_norm), "season"].astype(str).unique().tolist())
    return seasons

# ---- Pruned roster slice for team+season ----
@st.cache_data(ttl=600)
def team_season_view(processed_path: Path, college_norm: str, season: str) -> pd.DataFrame:
    cols = [
        "college","season","player_ind","player_number_ind","position","Archetype",
        "minutes_tot_ind","mins_per_game","pts_per_game","ast_per_game","reb_per_game",
        "stl_per_game","blk_per_game","eFG_pct","USG_pct","gp_ind"
    ]
    df = pd.read_parquet(processed_path, columns=[c for c in cols if c is not None])
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    return df[(df[" college_norm"] == str(college_norm)) & (df["season"].astype(str) == str(season))].copy()

# ---- Fetch rows for compare cart (by triples) ----
@st.cache_data(ttl=600)
def rows_for_compare(processed_path: Path, items: list[dict]) -> list[pd.Series]:
    """
    items: [{"player_ind": "...", "season": "...", "college": "..."}]
    Returns list of matching rows in the same order as items (first match per triple).
    """
    if not items:
        return []
    want = pd.DataFrame(items, columns=["player_ind","season","college"]).dropna()
    want["player_ind"] = want["player_ind"].astype(str)
    want["season"] = want["season"].astype(str)
    want["college"] = want["college"].astype(str)

    # Only load columns we need for the compare view
    cols = [
        "player_ind","season","college","position","Archetype",
        "pts_per_game","reb_per_game","ast_per_game","stl_per_game","blk_per_game",
        "eFG_pct","USG_pct","mins_per_game","minutes_tot_ind","player_number_ind"
    ]
    df = pd.read_parquet(processed_path, columns=cols)
    df["player_ind"] = df["player_ind"].astype(str)
    df["season"] = df["season"].astype(str)
    df["college"] = df["college"].astype(str)

    merged = want.merge(df, on=["player_ind","season","college"], how="left", suffixes=("", "_df"))

    out = []
    for _, row in want.iterrows():
        hit = merged[
            (merged["player_ind"] == row["player_ind"]) &
            (merged["season"] == row["season"]) &
            (merged["college"] == row["college"])
        ]
        if not hit.empty:
            out.append(hit.iloc[0])
    return out
