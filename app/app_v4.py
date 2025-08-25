import os
import json
import time
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import plotly.express as px
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Config / Paths
st.set_page_config(page_title="Coach Scouting Dashboard", layout="wide")

DATA_DIR = Path("data")
ART_DIR = Path("artifacts")
PIPELINE_DIR = Path("pipeline")  
CFG_PATH = PIPELINE_DIR / "config.yaml"

# Optional API runner (Option B). If set, the app will use the API instead of subprocess.
PIPELINE_API_URL = os.environ.get("PIPELINE_API_URL", "").strip()

# Utilities & Caching
def hash_data_folder() -> str:
    h = hashlib.sha256()
    for p in sorted(DATA_DIR.rglob("*.csv")):
        try:
            h.update(p.name.encode())
            h.update(p.read_bytes())
        except Exception:
            pass
    return h.hexdigest()

@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def latest_artifacts():
    latest = ART_DIR / "latest"
    if not latest.exists():
        return None
    return {
        "root": latest,
        "processed": latest / "processed.parquet",
        "model": latest / "kmeans_model.joblib",
        "summary": latest / "cluster_summary.json",
        "selection": latest / "selection.json",
        "elbow": latest / "elbow_plot.png",
        "silhouette": latest / "silhouette_plot.png",
        "db_plot": latest / "db_plot.png",
        "ch_plot": latest / "ch_plot.png",
    }

# Pretty column name utilities
BASE_MAP = {
    "college": "College",  
    "season": "Season",
    "player_number_ind": "Player #",
    "player_ind": "Player Name",
    "gp_ind": "Games Played",
    "gs_ind": "Games Started",
    "minutes_tot_ind": "Minutes (Total)",
    "scoring_pts_ind": "Points",
    "position": "Position",
    "rebounds_tot_ind": "Rebounds",
    "ast_ind": "Assists",
    "stl_ind": "Steals",
    "blk_ind": "Blocks",
    "to_ind": "Turnovers",
    "pts_per40": "PTS/40 mins",
    "reb_per40": "REB/40 mins",
    "ast_per40": "AST/40 mins",
    "stl_per40": "STL/40 mins",
    "blk_per40": "BLK/40 mins",
    "to_per40": "TOV/40 mins",
    "eFG_pct": "Effective Field Goal %",
    "TS_pct": "True Shooting %",
    "USG_pct": "Usage %",
    "ORB_pct": "Offensive Rebound %",
    "DRB_pct": "Defensive Rebound %",
    "AST_pct": "Assist %",
    "AST_per_TO": "AST/TOV",
    "3pt_3pt_pct_ind": "3PT %",
    "three_per40": "3PT/40 mins",
    "threeA_per40": "3PT Attempts/40 mins",
    "three_per100": "3PT/100 Possessions",
    "threeA_rate": "3PT Attempts Rate",
    "DRCR": "Defensive Rebound Conversion Rate",
    "STL_TO_ratio": "Steal-to-Turnover Ratio",
    "def_stops_per100": "Defensive Stops per 100 Possessions",
    "DPMR": "Defensive Plus-Minus Rating",
    "TUSG_pct": "True Usage %",  
    "Gravity": "Gravity (Off-Ball Impact)",
    "PPT": "Points per Touch",
    "Spacing": "Spacing Score",
    "Assist_to_Usage": "Assist-to-Usage Ratio",
    "APC": "Adjusted Playmaking Creation",
    "PEF": "Physical Efficiency Factor",
    "OEFF": "Offensive Efficiency",
    "TOV_pct": "Turnover %",  
    "SEM": "Shot Efficiency Metric",
    "PEI": "Player Efficiency Impact",
    "BoxCreation": "Box Creation (Playmaking Opportunities)",
    "OLI": "Offensive Load Index",
    "IPM": "Impact Metric",
    "threeA_per100": "3PA per 100 Possessions",
    "2pt_pct": "2PT FG%",
    "FTr": "Free Throw Rate",
    "PPP": "Points per Possession",
    "possessions": "Possessions",
    "scoring_pts_per100": "Points per 100 Possessions",
    "ast_per100": "Assists per 100 Possessions",
    "rebounds_tot_per100": "Rebounds per 100 Possessions",
    "stl_per100": "Steals per 100 Possessions",
    "blk_per100": "Blocks per 100 Possessions",
    "to_per100": "Turnovers per 100 Possessions",
    "mins_per_game": "Minutes per Game",
    "pts_per_game": "Points per Game",
    "ast_per_game": "Assists per Game",
    "reb_per_game": "Rebounds per Game",
    "stl_per_game": "Steals per Game",
    "blk_per_game": "Blocks per Game",
    "to_per_game": "Turnovers per Game",
    "scoring_pts_share": "Scoring Share",
    "ast_share": "Assist Share",
    "rebounds_tot_share": "Rebound Share",
    "stl_share": "Steal Share",
    "blk_share": "Block Share",
    "to_share": "Turnover Share",
    "team_TS_pct": "Team True Shooting %",
    "TS_diff": "True Shooting Differential",
    "ast_per_fgm": "Assists per FGM",
    "tov_rate": "Turnover Rate",
    "game_score": "Game Score",
    "game_score_per40": "Game Score per 40",
    "min_share": "Minute Share",
    "Archetype": "Player Archetype",
}

COLLEGE_MAP = {
    "manhattan":       "Manhattan Jaspers",
    "mount st marys":  "Mount St. Mary’s Mountaineers",
    "niagara":         "Niagara Purple Eagles",
    "sacred heart":    "Sacred Heart Pioneers",
    "quinnipiac":      "Quinnipiac Bobcats",
    "merrimack":       "Merrimack Warriors",
    "marist":          "Marist Red Foxes",
    "fairfield":       "Fairfield Stags",
    "iona":            "Iona Gaels",
    "siena":           "Siena Saints",
    "canisius":        "Canisius Golden Griffins",
    "saint peters":    "Saint Peter’s Peacocks",
    "rider":           "Rider Broncs",
}
COLLEGE_MAP_INV = {v: k for k, v in COLLEGE_MAP.items()} 


def make_unique_cols(names):
    seen = {}
    out = []
    for n in map(str, names):
        key = n.strip()
        if key in seen:
            seen[key] += 1
            out.append(f"{key}_{seen[key]}")  
        else:
            seen[key] = 0
            out.append(key)
    return out

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns using ONLY BASE_MAP (exact matches).
    All other columns keep their original names.
    """
    mapping = {c: BASE_MAP.get(c, c) for c in df.columns}
    out = df.rename(columns=mapping)
    # Ensure uniqueness for Streamlit/PyArrow
    out.columns = make_unique_cols(out.columns)
    return out


# Pipeline runners
def run_pipeline_local():
    """Run orchestrate.py locally (Option A)."""
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env["RUN_ID"] = run_id

    cmd = ["python", str(PIPELINE_DIR / "orchestrate.py")]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PIPELINE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    with st.status("Running pipeline...", expanded=True) as status:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            st.write(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            status.update(label="Pipeline failed", state="error")
            st.error("Pipeline failed. See logs above.")
            return None

        # Update artifacts/latest to the new run (symlink; fallback to copy)
        latest = ART_DIR / "latest"
        current = ART_DIR / run_id
        try:
            if latest.exists():
                if latest.is_symlink() or latest.is_file():
                    latest.unlink()
                elif latest.is_dir():
                    import shutil
                    shutil.rmtree(latest)
            latest.symlink_to(current, target_is_directory=True)
        except Exception:
            import shutil
            if latest.exists():
                if latest.is_dir():
                    shutil.rmtree(latest)
                else:
                    latest.unlink()
            shutil.copytree(current, latest)

        # Clear caches so new artifacts load
        st.cache_data.clear()
        st.cache_resource.clear()
        status.update(label=f"Finished: {run_id}", state="complete")

    return run_id

def run_pipeline_api():
    """Run via FastAPI Runner (Option B). Requires PIPELINE_API_URL env."""
    import requests
    API = PIPELINE_API_URL.rstrip("/")

    with st.status("Requesting pipeline run via API...", expanded=True) as status:
        r = requests.post(f"{API}/runs", json={})
        r.raise_for_status()
        run_id = r.json()["run_id"]
        st.write(f"Run queued: {run_id}")

        # poll
        while True:
            s = requests.get(f"{API}/runs/{run_id}")
            s.raise_for_status()
            state = s.json().get("state")
            st.write(f"State: {state}")
            if state in ("succeeded", "failed", "error"):
                break
            time.sleep(2)

        if state != "succeeded":
            status.update(label="Pipeline failed", state="error")
            st.error(f"Run {run_id} ended in state: {state}")
            return None

        st.cache_data.clear()
        st.cache_resource.clear()
        status.update(label=f"Finished: {run_id}", state="complete")
        return run_id

def run_pipeline():
    if PIPELINE_API_URL:
        return run_pipeline_api()
    return run_pipeline_local()

# Sidebar: Data Manager
st.sidebar.header("Data Manager")
DATA_DIR.mkdir(parents=True, exist_ok=True)
uploads = st.sidebar.file_uploader(
    "Upload CSVs (manual placement currently)",
    accept_multiple_files=True,
    type=["csv"],
)
if uploads:
    for f in uploads:
        out = DATA_DIR / f.name
        out.write_bytes(f.getvalue())
    st.sidebar.success(
        "Uploaded to data/. Please place/rename into output_by_college_clean/<team>/<season>/ and run the pipeline."
    )

st.sidebar.caption("Data snapshot hash")
st.sidebar.code(hash_data_folder())

# Tabs
tab_train, tab_roster, tab_matchups = st.tabs(
    ["Train & Explore", "Roster (Team & Season)", "Match-ups"]
)

# Train & Explore
with tab_train:
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Run pipeline on current data")
        if st.button("Run pipeline now"):
            rid = run_pipeline()
            if rid:
                st.success(f"Artifacts updated: {rid}")

    with colB:
        if st.button("Refresh artifacts"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Refreshed")

    st.subheader("Latest artifacts")
    paths = latest_artifacts()
    if not paths:
        st.info("No artifacts yet. Run the pipeline.")
    else:
        cols = st.columns(3)
        cols[0].metric("Processed parquet", "✅" if paths["processed"].exists() else "❌")
        cols[1].metric("Model", "✅" if paths["model"].exists() else "❌")
        cols[2].metric("Summary", "✅" if paths["summary"].exists() else "❌")

        sel = load_json_file(paths["selection"]) if paths["selection"].exists() else {}
        summary = load_json_file(paths["summary"]) if paths["summary"].exists() else {}
        n_pca = sel.get("n_pca") or summary.get("selected", {}).get("pca_components")
        best_k = sel.get("best_k") or summary.get("selected", {}).get("n_clusters")
        sil = (summary.get("scores", {}) or {}).get("silhouette")

        c1, c2, c3 = st.columns(3)
        c1.metric("PCA components", n_pca if n_pca is not None else "—")
        c2.metric("Clusters (k)", best_k if best_k is not None else "—")
        c3.metric("Silhouette", f"{sil:.3f}" if isinstance(sil, (int, float)) else "—")

        # pcols = st.columns(2)
        # if paths["elbow"].exists():
        #     pcols[0].image(str(paths["elbow"]), caption="Elbow Plot", use_container_width=True)
        # if paths["silhouette"].exists():
        #     pcols[1].image(str(paths["silhouette"]), caption="Silhouette vs k", use_container_width=True)
        
        # pcols2 = st.columns(2)
        # if paths["ch_plot"].exists():
        #     pcols2[0].image(str(paths["ch_plot"]), caption="CH Plot", use_container_width=True)
        # if paths["db_plot"].exists():
        #     pcols2[1].image(str(paths["db_plot"]), caption="DB Plot", use_container_width=True)

        # with st.expander("cluster_summary.json"):
        #     st.json(summary if summary else {"info": "No summary yet"})

        # Interactive pie chart of cluster composition
        cluster_sizes = (summary.get("cluster_sizes") or {})
        if not cluster_sizes:
            st.info("No 'cluster_sizes' found in cluster_summary.json.")
        else:
            # Optional mapping if your JSON ever stores cluster-id->name separately (backward compat)
            arch_map = (summary.get("archetype_names")
                        or summary.get("cluster_archetypes")
                        or summary.get("cluster_to_archetype")
                        
                        or {})
            
            items = sorted(((str(name), int(v)) for name, v in cluster_sizes.items()),
                        key=lambda x: (-x[1], x[0]))
            labels = [name for name, _ in items]
            counts = [cnt for _, cnt in items]

            total = max(sum(counts), 1)
            df_pie = pd.DataFrame({
                "Archetype": labels,
                "Count": counts,
                "Percent": [c * 100.0 / total for c in counts],
            })

        fig = px.pie(
            df_pie,
            names="Archetype",
            values="Count",
            hole=0.35,  # donut look; set to 0 for full pie
        )
        fig.update_traces(
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>% of total: %{percent}<extra></extra>"
        )
        fig.update_layout(
            legend_title_text="Archetypes",
            margin=dict(l=10, r=10, t=30, b=10),
            title_text="Cluster Composition by Archetype"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Counts by archetype")
        st.dataframe(
            df_pie[["Archetype", "Count", "Percent"]].sort_values("Count", ascending=False),
            use_container_width=True
        )

# Roster (Team & Season)
with tab_roster:
    st.subheader("Roster & Metrics")
    paths = latest_artifacts()
    if not paths or not paths["processed"].exists():
        st.info("Run pipeline to populate processed data.")
    else:
        df = load_parquet(paths["processed"]).copy()

        # Normalize & map college names for display
        if "college" in df.columns:
            df["_college_norm"] = (
                df["college"].astype(str).str.strip().str.lower()
            )
            # Show full names if we have them; otherwise fall back to original
            df["college_display"] = df["_college_norm"].map(COLLEGE_MAP).fillna(df["college"])

       
    TEAM_COL = "college" if "college" in df.columns else None
    if TEAM_COL is None:
        st.warning("'college' not found in processed data.")
    elif "season" not in df.columns:
        st.warning("Expected column 'season' not found in processed data.")
    else:
        # Use display names in the dropdown
        TEAM_COL_DISPLAY = "college_display"
        teams = sorted(df[TEAM_COL_DISPLAY].dropna().unique().tolist())
        team_display = st.selectbox("Team", teams, index=0 if teams else None)

        # Convert selected display back to normalized key for filtering
        selected_norm = COLLEGE_MAP_INV.get(
            team_display, str(team_display).strip().lower()
        )

        seasons_for_team = sorted(
            df.loc[df["_college_norm"] == selected_norm, "season"].dropna().unique().tolist()
        ) if team_display else []

        if not seasons_for_team:
            st.info("No seasons found for the selected team.")
        season = st.selectbox("Season", seasons_for_team, index=0 if seasons_for_team else None)

        show_adv = st.checkbox("Show advanced metrics", value=False)

        # Only proceed when both selections exist
        if team_display and season:
            # Filter to selection using normalized value
            filt = df[(df["_college_norm"] == selected_norm) & (df["season"] == season)].copy()

            if show_adv:
                view = filt.copy()
            else:
                minimal_cols = [
                    "player_ind", "player_number_ind", "scoring_pts_ind",
                    "ast_ind", "rebounds_tot_ind", "blk_ind", "stl_ind",
                    "gp_ind", "to_ind", "Archetype"
                ]
                cols = [c for c in minimal_cols if c in filt.columns]
                view = filt[cols].copy()

            # Rename columns nicely
            view = rename_columns(view)

            st.dataframe(view, use_container_width=True)

            # Use the pretty display name in the export filename and notes key
            st.download_button(
                "Download CSV (current selection)",
                data=view.to_csv(index=False).encode(),
                file_name=f"{team_display}_{season}_players.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.subheader("Notes & Comments")
            notes_file = Path("app_notes.json")
            notes = load_json_file(notes_file) if notes_file.exists() else {}
            key = f"{team_display}__{season}"
            existing_text = notes.get(key, "")
            txt = st.text_area(
                "Write notes for this team/season",
                value=existing_text,
                height=160,
                key="notes_roster",
            )
            if st.button("Save notes", key="save_notes_roster"):
                notes[key] = txt
                notes_file.write_text(json.dumps(notes, indent=2))
                st.success("Notes saved")
        else:
            st.info("Select both Team and Season to view the roster.")

# Match-ups
with tab_matchups:
    st.subheader("Match-up Planner")
    paths = latest_artifacts()
    if not paths or not paths["processed"].exists():
        st.info("Run pipeline first.")
    else:
        df = load_parquet(paths["processed"]).copy()

        TEAM_COL = "team" if "team" in df.columns else ("college" if "college" in df.columns else None)
        if TEAM_COL is None or "cluster" not in df.columns:
            st.warning("Expected columns not found (need 'cluster' and either 'team' or 'college').")
        else:
            summary = load_json_file(paths["summary"]) if paths["summary"].exists() else {}
            arch_map = (summary.get("archetype_names") or {})

            teams = sorted(df[TEAM_COL].dropna().unique()) if TEAM_COL in df.columns else []
            left, right = st.columns(2)
            my_team = left.selectbox("My Team", teams, key="myteam")
            opp_team = right.selectbox("Opponent", teams, key="oppteam")

            if "season" in df.columns and my_team and opp_team:
                seasons_left = sorted(df.loc[df[TEAM_COL] == my_team, "season"].dropna().unique())
                seasons_right = sorted(df.loc[df[TEAM_COL] == opp_team, "season"].dropna().unique())
                cL, cR = st.columns(2)
                my_season = cL.selectbox("Season (My Team)", seasons_left) if seasons_left else None
                opp_season = cR.selectbox("Season (Opponent)", seasons_right) if seasons_right else None
            else:
                my_season = opp_season = None

            def mix(d: pd.DataFrame):
                m = d.groupby("cluster").size().rename("count").reset_index()
                m["archetype"] = m["cluster"].astype(str).map(arch_map).fillna(m["cluster"].astype(str))
                return m[["cluster", "archetype", "count"]].sort_values("count", ascending=False)

            dL = df[df[TEAM_COL] == my_team]
            dR = df[df[TEAM_COL] == opp_team]
            if my_season is not None and "season" in df.columns:
                dL = dL[dL["season"] == my_season]
            if opp_season is not None and "season" in df.columns:
                dR = dR[dR["season"] == opp_season]

            cols = st.columns(2)
            with cols[0]:
                st.caption(f"{my_team} — archetype mix")
                st.dataframe(mix(dL), use_container_width=True)
            with cols[1]:
                st.caption(f"{opp_team} — archetype mix")
                st.dataframe(mix(dR), use_container_width=True)

            st.markdown("**Coaching heuristics**")
            st.write("- Add rim protection vs heavy paint attacks.")
            st.write("- Add shooting vs packed paint or zone looks.")
            st.write("- Add on-ball creation vs switch-heavy, low-foul teams.")
