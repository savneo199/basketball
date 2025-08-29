# tabs/compare_player.py
from __future__ import annotations
from pathlib import Path
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from helpers.helpers import latest_artifacts, load_parquet

COMPARE_METRICS = [
    ("pts_per_game", "PTS/G"),
    ("reb_per_game", "REB/G"),
    ("ast_per_game", "AST/G"),
    ("stl_per_game", "STL/G"),
    ("blk_per_game", "BLK/G"),
    ("eFG_pct", "eFG%"),   # assume 0–1; we’ll display as %
    ("USG_pct", "USG%"),
]

ID_COLS = ["player_ind", "college", "season", "position", "Archetype"]

def _load_df() -> pd.DataFrame | None:
    paths = latest_artifacts()
    if not paths or not Path(paths["processed"]).exists():
        return None
    try:
        return load_parquet(Path(paths["processed"]))
    except Exception:
        return None

def _option_label(row: pd.Series) -> str:
    return f"{row.get('player_ind','Player')} — {row.get('college','')} ({row.get('season','')})"

def _radar_figure(rows: list[pd.Series]) -> go.Figure:
    cats = [name for _, name in COMPARE_METRICS]
    fig = go.Figure()
    # collect max for normalization (avoid divide-by-zero)
    max_by_metric = []
    for key, _ in COMPARE_METRICS:
        vals = []
        for r in rows:
            v = r.get(key)
            try:
                v = float(v) if v is not None and not pd.isna(v) else 0.0
            except Exception:
                v = 0.0
            if key == "eFG_pct" or key.endswith("_pct"):
                # convert to percentage domain
                if v <= 1.0:
                    v = v * 100.0
            vals.append(v)
        max_by_metric.append(max(vals) if vals else 1.0)

    for r in rows:
        vals = []
        for (key, _), vmax in zip(COMPARE_METRICS, max_by_metric):
            v = r.get(key)
            try:
                v = float(v) if v is not None and not pd.isna(v) else 0.0
            except Exception:
                v = 0.0
            if key == "eFG_pct" or key.endswith("_pct"):
                if v <= 1.0:
                    v = v * 100.0
            vals.append(0.0 if vmax == 0 else v / vmax)
        vals.append(vals[0])  # close shape

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats + [cats[0]],
            fill="toself",
            name=_option_label(r),
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def render():
    st.subheader("Compare players")

    df = _load_df()
    if df is None or df.empty:
        st.info("No processed data found. Run the pipeline first.")
        return

    # Build options over all rows
    needed = set([k for k, _ in COMPARE_METRICS] + ID_COLS)
    sub = df[[c for c in df.columns if c in needed]].copy()
    sub = sub.dropna(subset=["player_ind"]).reset_index(drop=True)

    if sub.empty:
        st.info("No players to show.")
        return

    sub["__label"] = sub.apply(_option_label, axis=1)
    pick = st.multiselect(
        "Select up to 3 players",
        options=sub["__label"].tolist(),
        max_selections=3,
    )

    if not pick:
        st.caption("Choose one to three players to compare.")
        return

    rows = [sub[sub["__label"] == p].iloc[0] for p in pick]
    # KPI strip
    for r in rows:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PTS/G", f"{float(r.get('pts_per_game',0) or 0):.1f}")
        c2.metric("REB/G", f"{float(r.get('reb_per_game',0) or 0):.1f}")
        c3.metric("AST/G", f"{float(r.get('ast_per_game',0) or 0):.1f}")
        c4.metric("STL/G", f"{float(r.get('stl_per_game',0) or 0):.1f}")
        c5.metric("BLK/G", f"{float(r.get('blk_per_game',0) or 0):.1f}")
        st.caption(_option_label(r))

    st.markdown("---")
    st.plotly_chart(_radar_figure(rows), use_container_width=True)

    # Side-by-side table
    show_cols = ID_COLS + [k for k, _ in COMPARE_METRICS]
    table = pd.DataFrame([{c: r.get(c) for c in show_cols} for r in rows])
    # format percentages nicely
    if "eFG_pct" in table.columns:
        table["eFG_pct"] = table["eFG_pct"].apply(lambda v: (float(v)*100 if (v is not None and v <= 1) else float(v or 0)))
    if "USG_pct" in table.columns:
        table["USG_pct"] = table["USG_pct"].apply(lambda v: float(v or 0))
    st.dataframe(table, use_container_width=True)
