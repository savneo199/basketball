# tabs/player_finder.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox  # pip install streamlit-searchbox

from helpers.helpers import latest_artifacts, load_parquet

# top of player_finder.py
import bisect

@st.cache_data(show_spinner=False)
def build_name_index(names: list[str]):
    names_sorted = sorted(set(names))
    lowers = [n.lower() for n in names_sorted]
    return names_sorted, lowers

def suggest_prefix(names_sorted: list[str], lowers: list[str], q: str, limit: int = 12) -> list[str]:
    ql = q.lower()
    i = bisect.bisect_left(lowers, ql)
    out = []
    while i < len(lowers) and lowers[i].startswith(ql) and len(out) < limit:
        out.append(names_sorted[i]); i += 1
    return out


CORE_SHOW = [
    "player_ind", "college", "season", "position", "player_number_ind",
    "minutes_tot_ind", "mins_per_game",
    "pts_per_game", "reb_per_game", "ast_per_game",
    "stl_per_game", "blk_per_game",
    "eFG_pct", "USG_pct", "Archetype"
]

# ---------- data helpers ----------
def _load_df() -> pd.DataFrame | None:
    paths = latest_artifacts()
    if not paths or not Path(paths["processed"]).exists():
        return None
    return load_parquet(Path(paths["processed"]))

def _unique_names(df: pd.DataFrame) -> list[str]:
    if "player_ind" not in df.columns:
        return []
    return sorted(df["player_ind"].astype(str).dropna().unique().tolist())

def _suggest(names: list[str], q: str, limit: int = 20) -> list[str]:
    ql = q.lower().strip()
    starts = [n for n in names if n.lower().startswith(ql)]
    contains = [n for n in names if ql in n.lower() and not n.lower().startswith(ql)]
    return (starts + contains)[:limit]

# ---------- UI ----------
def render():
    st.subheader("Player finder")

    df = _load_df()
    if df is None or df.empty:
        st.info("No processed data found. Run the pipeline first.")
        return

    names = _unique_names(df)
    if not names:
        st.info("No player names in the dataset.")
        return

    if "pf_query" not in st.session_state:
        st.session_state["pf_query"] = ""

    names = _unique_names(df)
    names_sorted, lowers = build_name_index(names)

    def _search(q: str) -> list[str]:
        return suggest_prefix(names_sorted, lowers, q, 12) if len(q.strip()) >= 2 else []


    picked = st_searchbox(
        search_function=_search,
        key="pf_searchbox",
        default=st.session_state["pf_query"],
        placeholder="Search player…",
    )

    if not picked:
        st.caption("Start typing (≥2 chars) and pick a player.")
        return

    st.session_state["pf_query"] = picked
    selected_name = picked

    sub = df[df["player_ind"].astype(str) == str(selected_name)].copy()
    if sub.empty:
        st.warning("No rows found for that player.")
        return

    if "season" in sub.columns:
        seasons = sorted(sub["season"].dropna().unique().tolist())
        if len(seasons) > 1:
            season_opt = st.selectbox("Season", seasons, index=0)
            sub = sub[sub["season"] == season_opt]

    if len(sub) > 1 and "college" in sub.columns:
        colleges = sorted(sub["college"].dropna().unique().tolist())
        if len(colleges) > 1:
            college_opt = st.selectbox("College", colleges, index=0)
            sub = sub[sub["college"] == college_opt]

    if sub.empty:
        st.warning("No row left after filters.")
        return

    row = sub.iloc[0]
    title = f"{row.get('player_ind','Player')} — {row.get('college','')} ({row.get('season','')})"
    st.markdown(f"### {title}")

    def _num(v):
        try:
            return float(v) if v is not None and not pd.isna(v) else 0.0
        except Exception:
            return 0.0
