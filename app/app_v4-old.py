import os
import re
import json
import time
import sys
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import io
import zipfile
import plotly.express as px
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Config / Paths
st.set_page_config(page_title="Coach Scouting Dashboard", layout="wide")

APP_DIR = Path(__file__).resolve().parent

def _find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):  # climb up to 6 levels just in case
        if (cur / "pipeline" / "config.yaml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start  # fallback (shouldn't happen if your tree is standard)

ROOT = _find_project_root(APP_DIR)

DATA_DIR = ROOT / "data"
RAW_BASE = DATA_DIR / "output_by_college_clean"
ART_DIR = ROOT / "artifacts"
PIPELINE_DIR = ROOT / "pipeline"
CFG_PATH = PIPELINE_DIR / "config.yaml"

# Optional API runner (Option B)
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

# Pretty column name utilities (exact matches only)
BASE_MAP = {
    "college": "College",
    "season": "Season",
    "player_number_ind": "Player Number",
    "player_ind": "Player Name",
    "Archetype": "Player Archetype",
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
    seen, out = {}, []
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
    mapping = {c: BASE_MAP.get(c, c) for c in df.columns}
    out = df.rename(columns=mapping)
    out.columns = make_unique_cols(out.columns)
    return out

# Helpers for upload validation
def slugify_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def slugify_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [slugify_name(c) for c in df2.columns]
    return df2

def validate_individual_cols(cols: set) -> list:
    """Return a list of missing requirement messages for individual_stats_overall.csv."""
    req = {
        "player name": {"player", "player_name", "name"},
        "games played": {"gp"},
        "minutes": {"minutes_tot", "minutes"},
        "points": {"scoring_pts", "pts"},
        "rebounds": {"rebounds_tot", "reb"},
        "assists": {"ast"},
        "steals": {"stl"},
        "blocks": {"blk"},
        "turnovers": {"to", "tov"},
        "FG made": {"fg_fgm", "fgm"},
        "FG attempts": {"fg_fga", "fga"},
        "3PT made": {"3pt"},
        "FT attempts": {"ft_fta", "fta"},
    }
    missing = []
    for label, alts in req.items():
        if not any(a in cols for a in alts):
            missing.append(f"- {label} (one of: {', '.join(sorted(alts))})")
    return missing

def validate_team_cols(cols: set) -> list:
    """Return a list of missing requirement messages for team_stats.csv."""
    req = {
        "team FG attempts": {"fg_fga", "fga"},
        "team FT attempts": {"ft_fta", "fta"},
        "team turnovers": {"to", "tov"},
        "team minutes": {"minutes_tot", "minutes"},
    }
    missing = []
    for label, alts in req.items():
        if not any(a in cols for a in alts):
            missing.append(f"- {label} (one of: {', '.join(sorted(alts))})")
    return missing

def safe_join(base: Path, *parts: str) -> Path:
        """Join and ensure final path stays under base to prevent zip slip."""
        final = (base / Path(*parts)).resolve()
        if not str(final).startswith(str(base.resolve())):
            raise ValueError("Unsafe path detected during extraction.")
        return final

def is_csv(name: str) -> bool:
    return name.lower().endswith(".csv")

def norm_key(s: str) -> str:
    return COLLEGE_MAP_INV.get(s.strip(), s.strip().lower())

def extract_zip_file(zf: zipfile.ZipFile, to_dir: Path) -> list[Path]:
    """Extracts zip safely. Returns list of extracted CSV file paths (noise skipped)."""
    extracted = []
    for zi in zf.infolist():
        name = zi.filename

        # Skip macOS noise and hidden files
        # Entire __MACOSX tree
        if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
            continue
        # AppleDouble resource-fork files like "._file.csv"
        if Path(name).name.startswith("._"):
            continue
        # Hidden dotfiles/folders anywhere
        if any(part.startswith(".") for part in Path(name).parts):
            # still allow normal ".csv" files, but typical dotfiles here are junk
            continue

        if zi.is_dir():
            out_dir = safe_join(to_dir, name)
            out_dir.mkdir(parents=True, exist_ok=True)
            continue

        # Only extract CSVs we actually care about
        if not is_csv(name):
            continue

        out_path = safe_join(to_dir, name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        extracted.append(out_path)
    return extracted


def validate_pair(df_ind: pd.DataFrame, df_team: pd.DataFrame) -> tuple[list, list]:
    miss_ind = validate_individual_cols(set(df_ind.columns))
    miss_team = validate_team_cols(set(df_team.columns))
    return miss_ind, miss_team

def read_slug_csv(path: Path) -> pd.DataFrame:
    return slugify_columns(pd.read_csv(path))

def summarise_saved(rows: list[tuple[str, str]]):
    # rows: [(college_display, season), ...]
    if not rows:
        st.info("No valid (college, season) pairs were found.")
        return
    df_sum = pd.DataFrame(rows, columns=["College", "Season"]).drop_duplicates()
    st.success(f"Saved {len(df_sum)} season(s).")
    st.dataframe(df_sum.sort_values(["College", "Season"]), use_container_width=True)

# def _load_inference_assets(latest_dir: Path):
#     """
#     Load either a single inference pipeline or separate pieces.
#     Accepts multiple common filenames so you don't have to rename artifacts.
#     """
#     def _first_existing(root: Path, names):
#         for nm in names:
#             p = root / nm
#             if p.exists():
#                 return p
#         return None

#     assets = {}

#     # # 1) Try unified pipeline first
#     # pipe_path = _first_existing(latest_dir, [
#     #     "inference_pipeline.joblib",
#     #     "inference_pipeline.pkl",
#     # ])
#     # if pipe_path:
#     #     assets["pipeline"] = joblib.load(pipe_path)
#     #     assets["mode"] = "pipeline"
#     #     return assets

#     # Fall back to separate pieces
#     kmeans_path = latest_dir / "kmeans_model.joblib"
#     scaler_path = latest_dir / "scaler.joblib"
#     pca_path = latest_dir / "pca.joblib"

#     feats_path = _first_existing(latest_dir, [
#         "features.json", "feature_order.json", "feature_names.json",
#     ])

#     if kmeans_path:
#         assets["kmeans"] = joblib.load(kmeans_path)
#     if scaler_path:
#         assets["scaler"] = joblib.load(scaler_path)
#     if pca_path:
#         assets["pca"] = joblib.load(pca_path)
#     if feats_path:
#         assets["features"] = json.loads(Path(feats_path).read_text())

#     # Enough info?
#     has_names = getattr(assets.get("kmeans"), "feature_names_in_", None) is not None
#     if "kmeans" in assets and ("features" in assets or has_names):
#         assets["mode"] = "pieces"
#         return assets

#     assets["mode"] = "none"
#     return assets


# def _predict_archetypes(df_players: pd.DataFrame, latest_dir: Path) -> pd.DataFrame:
#     """
#     Takes a player-level DataFrame (columns slugified to match training),
#     runs the latest inference assets, and returns df with ['cluster', 'Archetype'].
#     """
#     assets = _load_inference_assets(latest_dir)
#     if assets.get("mode") in ("none", "error"):
#         raise RuntimeError(
#             "Inference assets not found. Expected 'inference_pipeline.joblib' or "
#             "kmeans/scaler/pca + features. Run training once with saving enabled (see snippet below)."
#         )

#     # Load cluster name mapping (best-effort)
#     arch_map = {}
#     summary_path = latest_dir / "cluster_summary.json"
#     if summary_path.exists():
#         try:
#             summary = json.loads(summary_path.read_text())
#             # try a few common keys
#             arch_map = summary.get("archetype_names") or summary.get("archetype_map") or {}
#             # normalize keys to str
#             arch_map = {str(k): v for k, v in arch_map.items()}
#         except Exception:
#             pass

#     # Prepare X and predict
#     if assets["mode"] == "pipeline":
#         pipe = assets["pipeline"]
#         X = df_players.copy()
#         preds = pipe.predict(X)
#     else:
#         # pieces mode
#         model = assets["kmeans"]
#         # Resolve feature order
#         if "features" in assets:
#             feat_cols = list(assets["features"])
#         elif hasattr(model, "feature_names_in_"):
#             feat_cols = list(model.feature_names_in_)
#         else:
#             raise RuntimeError("No feature list available for inference.")

#         # keep only needed features, coerce numeric
#         X = df_players.copy()
#         missing = [c for c in feat_cols if c not in X.columns]
#         if missing:
#             raise RuntimeError(f"Your uploaded data is missing expected columns: {missing[:10]}{'...' if len(missing)>10 else ''}")

#         X = X[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

#         # transform
#         if "scaler" in assets:
#             X = assets["scaler"].transform(X)
#         if "pca" in assets:
#             X = assets["pca"].transform(X)

#         preds = model.predict(X)

#     # Build result
#     out = df_players.copy()
#     out["cluster"] = preds
#     out["Archetype"] = [arch_map.get(str(int(c)), f"Cluster {int(c)}") for c in preds]
#     return out



# Pipeline runner
def run_pipeline():
    """Run orchestrate.py locally"""
    print("Running pipeline locally...")
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    env = os.environ.copy()
    env["RUN_ID"] = run_id

    orch = (PIPELINE_DIR / "orchestrate.py").resolve()
    if not orch.exists():
        st.error(f"orchestrate.py not found at: {orch}")
        return None

    # Use absolute path to the script; keep cwd=PIPELINE_DIR so any relative paths inside
    # orchestrate.py (like notebooks/ …) resolve correctly.
    cmd = [sys.executable, str(orch)]  # <-- use the same venv as the app
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

        # Update artifacts/latest to the new run (symlink; fallback copy)
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

        st.cache_data.clear()
        st.cache_resource.clear()
        status.update(label=f"Finished: {run_id}", state="complete")

    return run_id

# Tabs
tab_train, tab_hist, tab_matchups, tab_upload = st.tabs(
    ["Train & Explore", "Historical Data", "Match-ups", "Upload & Classify"]
)

# Train & Explore
with tab_train:
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Run pipeline on current data - GitHub")
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

        cluster_sizes = (summary.get("cluster_sizes") or {})
        if not cluster_sizes:
            st.info("No 'cluster_sizes' found in cluster_summary.json.")
        else:
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
                df_pie, names="Archetype", values="Count", hole=0.35,
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

# Historical datat
with tab_hist:
    st.subheader("Roster & Metrics")
    paths = latest_artifacts()
    if not paths or not paths["processed"].exists():
        st.info("Run pipeline to populate processed data.")
    else:
        df = load_parquet(paths["processed"]).copy()
        # Normalize & map college names for display
        if "college" in df.columns:
            df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
            df["college_display"] = df[" college_norm"].map(COLLEGE_MAP).fillna(df["college"])
        elif "college" not in df.columns:
            st.info("College names not found.")
        elif not df:
            st.info("No data available.")

    TEAM_COL = "college" if "college" in df.columns else None
    if TEAM_COL is None:
        st.warning("'college' not found in processed data.")
    elif "season" not in df.columns:
        st.warning("Expected column 'season' not found in processed data.")
    else:
        TEAM_COL_DISPLAY = "college_display"
        teams = sorted(df[TEAM_COL_DISPLAY].dropna().unique().tolist())
        team_display = st.selectbox("Team", teams, index=0 if teams else None)
        selected_norm = COLLEGE_MAP_INV.get(team_display, str(team_display).strip().lower())
        seasons_for_team = sorted(
            df.loc[df[" college_norm"] == selected_norm, "season"].dropna().unique().tolist()
        ) if team_display else []
        if not seasons_for_team:
            st.info("No seasons found for the selected team.")
        season = st.selectbox("Season", seasons_for_team, index=0 if seasons_for_team else None)
        show_adv = st.checkbox("Show advanced metrics", value=False)

        if team_display and season:
            filt = df[(df[" college_norm"] == selected_norm) & (df["season"] == season)].copy()
            
            if show_adv:
                drop_cols = [c for c in [" college_norm", "college_display", "college", "season"] if c in filt.columns]
                view = filt.drop(columns=drop_cols).copy()
            else:
                minimal_cols = [
                    "player_ind", "player_number_ind", "minutes_tot_ind", "Archetype", "scoring_pts_ind",
                    "ast_ind", "rebounds_tot_ind", "stl_ind",
                    "gp_ind"
                ]
                cols = [c for c in minimal_cols if c in filt.columns]
                view = filt[cols].copy()
            view = rename_columns(view)
            st.dataframe(view, use_container_width=True)
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
                "Write notes for this team/season", value=existing_text, height=160, key="notes_roster",
            )
            if st.button("Save notes", key="save_notes_roster"):
                notes[key] = txt
                notes_file.write_text(json.dumps(notes, indent=2))
                st.success("Notes saved")
        else:
            st.info("Select both Team and Season to view the roster.")

# Match-ups
with tab_matchups:
    st.subheader("Coming Soon")
    
    # st.subheader("Match-up Planner")
    # paths = latest_artifacts()
    # if not paths or not paths["processed"].exists():
    #     st.info("Run pipeline first.")
    # else:
    #     df = load_parquet(paths["processed"]).copy()
    #     TEAM_COL = "team" if "team" in df.columns else ("college" if "college" in df.columns else None)
    #     if TEAM_COL is None or "cluster" not in df.columns:
    #         st.warning("Expected columns not found (need 'cluster' and either 'team' or 'college').")
    #     else:
    #         summary = load_json_file(paths["summary"]) if paths["summary"].exists() else {}
    #         arch_map = (summary.get("archetype_names") or {})
    #         teams = sorted(df[TEAM_COL].dropna().unique()) if TEAM_COL in df.columns else []
    #         left, right = st.columns(2)
    #         my_team = left.selectbox("My Team", teams, key="myteam")
    #         opp_team = right.selectbox("Opponent", teams, key="oppteam")
    #         if "season" in df.columns and my_team and opp_team:
    #             seasons_left = sorted(df.loc[df[TEAM_COL] == my_team, "season"].dropna().unique())
    #             seasons_right = sorted(df.loc[df[TEAM_COL] == opp_team, "season"].dropna().unique())
    #             cL, cR = st.columns(2)
    #             my_season = cL.selectbox("Season (My Team)", seasons_left) if seasons_left else None
    #             opp_season = cR.selectbox("Season (Opponent)", seasons_right) if seasons_right else None
    #         else:
    #             my_season = opp_season = None
    #         def mix(d: pd.DataFrame):
    #             m = d.groupby("cluster").size().rename("count").reset_index()
    #             m["archetype"] = m["cluster"].astype(str).map(arch_map).fillna(m["cluster"].astype(str))
    #             return m[["cluster", "archetype", "count"]].sort_values("count", ascending=False)
    #         dL = df[df[TEAM_COL] == my_team]; dR = df[df[TEAM_COL] == opp_team]
    #         if my_season is not None and "season" in df.columns:
    #             dL = dL[dL["season"] == my_season]
    #         if opp_season is not None and "season" in df.columns:
    #             dR = dR[dR["season"] == opp_season]
    #         cols = st.columns(2)
    #         with cols[0]:
    #             st.caption(f"{my_team} — archetype mix")
    #             st.dataframe(mix(dL), use_container_width=True)
    #         with cols[1]:
    #             st.caption(f"{opp_team} — archetype mix")
    #             st.dataframe(mix(dR), use_container_width=True)
    #         st.markdown("**Coaching heuristics**")
    #         st.write("- Add rim protection vs heavy paint attacks.")
    #         st.write("- Add shooting vs packed paint or zone looks.")
    #         st.write("- Add on-ball creation vs switch-heavy, low-foul teams.")

# Upload & Classify
with tab_upload:
    st.subheader("Upload new team-season CSVs and classify archetypes")

    # # Inputs
    # col_t1, col_t2 = st.columns([2, 1])
    # team_display_in = col_t1.text_input("Team (display name)", placeholder="e.g., Fairfield Stags")
    # season_in = col_t2.text_input("Season (folder name)", placeholder="e.g., 2024-25")

    # st.markdown("**Files required:**")
    # st.write("- `individual_stats_overall.csv` (per-player stats)")
    # st.write("- `team_stats.csv` (team totals)")

    # up_ind = st.file_uploader("Upload individual_stats_overall.csv", type=["csv"], key="up_ind")
    # up_team = st.file_uploader("Upload team_stats.csv", type=["csv"], key="up_team")

    # if st.button("Validate & Save"):
    #     if not team_display_in or not season_in or not up_ind or not up_team:
    #         st.error("Please provide team, season, and both CSV files.")
    #     else:
    #         # Normalize team to our internal key (lowercased words)
    #         team_key = COLLEGE_MAP_INV.get(team_display_in.strip(), team_display_in.strip().lower())
    #         # Read and slugify both CSVs for validation
    #         try:
    #             df_ind_raw = pd.read_csv(up_ind)
    #             df_ind = slugify_columns(df_ind_raw)
    #             df_team_raw = pd.read_csv(up_team)
    #             df_team = slugify_columns(df_team_raw)
    #         except Exception as e:
    #             st.error(f"Failed to read CSVs: {e}")
    #             st.stop()

    #         miss_ind = validate_individual_cols(set(df_ind.columns))
    #         miss_team = validate_team_cols(set(df_team.columns))

    #         if miss_ind or miss_team:
    #             st.error("Validation failed. Please fix the following:")
    #             if miss_ind:
    #                 st.write("**individual_stats_overall.csv**")
    #                 st.code("\n".join(miss_ind))
    #             if miss_team:
    #                 st.write("**team_stats.csv**")
    #                 st.code("\n".join(miss_team))
    #         else:
    #             # Save originals to the expected raw folder structure
    #             dest_dir = RAW_BASE / team_key / season_in
    #             dest_dir.mkdir(parents=True, exist_ok=True)
    #             ind_path = dest_dir / "individual_stats_overall.csv"
    #             team_path = dest_dir / "team_stats.csv"
    #             ind_path.write_bytes(up_ind.getvalue())
    #             team_path.write_bytes(up_team.getvalue())
    #             st.success(f"Saved files to: {dest_dir}")

    #             # Quick preview (slugified headers) for user sanity
    #             with st.expander("Preview (first 5 rows, individual)"):
    #                 st.dataframe(df_ind.head(), use_container_width=True)
    #             with st.expander("Preview (first 5 rows, team)"):
        
    #                 st.dataframe(df_team.head(), use_container_width=True)
#____________________________________________________________________________________________________________
#LIVE INFERENCE
    # if st.button("Validate & Save"):
    #     if not team_display_in or not season_in or not up_ind or not up_team:
    #         st.error("Please provide team, season, and both CSV files.")
    #     else:
    #         # Normalize team to our internal key (lowercased words)
    #         team_key = COLLEGE_MAP_INV.get(team_display_in.strip(), team_display_in.strip().lower())
    #         # Read and slugify both CSVs for validation
    #         try:
    #             df_ind_raw = pd.read_csv(up_ind)
    #             df_ind = slugify_columns(df_ind_raw)
    #             df_team_raw = pd.read_csv(up_team)
    #             df_team = slugify_columns(df_team_raw)
    #         except Exception as e:
    #             st.error(f"Failed to read CSVs: {e}")
    #             st.stop()

    #         miss_ind = validate_individual_cols(set(df_ind.columns))
    #         miss_team = validate_team_cols(set(df_team.columns))

    #         if miss_ind or miss_team:
    #             st.error("Validation failed. Please fix the following:")
    #             if miss_ind:
    #                 st.write("**individual_stats_overall.csv**")
    #                 st.code("\n".join(miss_ind))
    #             if miss_team:
    #                 st.write("**team_stats.csv**")
    #                 st.code("\n".join(miss_team))
    #         else:
    #             # Save originals to the expected raw folder structure
    #             dest_dir = RAW_BASE / team_key / season_in
    #             dest_dir.mkdir(parents=True, exist_ok=True)
    #             ind_path = dest_dir / "individual_stats_overall.csv"
    #             team_path = dest_dir / "team_stats.csv"
    #             ind_path.write_bytes(up_ind.getvalue())
    #             team_path.write_bytes(up_team.getvalue())
    #             st.success(f"Saved files to: {dest_dir}")

    #             # Keep context for the next rerun (when user clicks "Classify now")
    #             st.session_state["last_upload"] = {
    #                 "team_display": team_display_in,
    #                 "team_key": team_key,
    #                 "season": season_in,
    #                 "dest_dir": str(dest_dir),
    #                 "ind_path": str(ind_path),
    #                 "team_path": str(team_path),
    #             }

    #             # Quick preview (slugified headers) for user sanity
    #             with st.expander("Preview (first 5 rows, individual)"):
    #                 st.dataframe(df_ind.head(), use_container_width=True)
    #             with st.expander("Preview (first 5 rows, team)"):
    #                 st.dataframe(df_team.head(), use_container_width=True)

    # --- NEW: Infer-only classification (outside the Validate & Save block) ---
    # st.markdown("---")
    # st.subheader("Classify now (no retrain)")
    # 
    # if st.button("Classify uploaded roster now", key="infer_now"):
    #     latest = latest_artifacts()
    #     if not latest or not Path(latest["root"]).exists():
    #         st.error("No latest artifacts found. Run training once to produce a model.")
    #         st.stop()

    #     # Prefer the most recent saved upload from session_state, otherwise derive from inputs
    #     ctx = st.session_state.get("last_upload", None)
    #     if ctx:
    #         ind_path = Path(ctx["ind_path"])
    #         team_display = ctx["team_display"]
    #         season = ctx["season"]
    #     else:
    #         # Fallback: build from current inputs
    #         if not team_display_in or not season_in:
    #             st.error("Provide Team and Season, or upload/save first.")
    #             st.stop()
    #         team_key = COLLEGE_MAP_INV.get(team_display_in.strip(), team_display_in.strip().lower())
    #         ind_path = RAW_BASE / team_key / season_in / "individual_stats_overall.csv"
    #         team_display = team_display_in
    #         season = season_in

    #     if not ind_path.exists():
    #         st.error(f"Expected roster file not found at: {ind_path}")
    #         st.stop()

    #     # Read & slugify, then infer
    #     try:
    #         df_ind_now = pd.read_csv(ind_path)
    #         df_ind_now = slugify_columns(df_ind_now)
    #     except Exception as e:
    #         st.error(f"Failed to read saved individual CSV: {e}")
    #         st.stop()

    #     try:
    #         preds_df = _predict_archetypes(df_ind_now, latest["root"])
            
    #     except Exception as e:
    #         st.info(f"Latest artifacts found at: {latest['root']}")
    #         st.error(f"Inference failed: {e}")
    #         st.info(
    #             "If this is the first time, ensure your training saves an "
    #             "'inference_pipeline.joblib' (or separate scaler/pca/features)."
    #         )
    #         st.stop()

    #     # Display a compact view
    #     show_cols = []
    #     for c in ["player_name", "player", "player_ind", "player_number_ind", "position", "minutes_tot_ind", "scoring_pts_ind"]:
    #         if c in preds_df.columns:
    #             show_cols.append(c)
    #     show_cols += [c for c in ["cluster", "Archetype"] if c not in show_cols]

    #     st.success(f"Classification complete for {team_display} — {season}.")
    #     st.dataframe(preds_df[show_cols] if show_cols else preds_df, use_container_width=True)

        # st.download_button(
        #     "Download classifications (CSV)",
        #     data=preds_df.to_csv(index=False).encode(),
        #     file_name=f"{team_display}_{season}_archetypes.csv",
        #     mime="text/csv",
        # )

    # Mode Switcher
    mode = st.radio(
        "Upload mode",
        (
            "Single season of one college",
            "Multiple seasons of one college (ZIP)",
            "Multiple seasons of multiple colleges (ZIP)",
        ),
        index=0,
    )

    st.markdown("**Each season needs two CSVs:** `individual_stats_overall.csv` and `team_stats.csv`")

    # Single season / one college
    if mode == "Single season of one college":
        col_t1, col_t2 = st.columns([2, 1])
        team_display_in = col_t1.text_input("Team (display name)", placeholder="e.g., Fairfield Stags")
        season_in = col_t2.text_input("Season (folder name)", placeholder="e.g., 2024-25")

        up_ind = st.file_uploader("Upload individual_stats_overall.csv", type=["csv"], key="up_ind_single")
        up_team = st.file_uploader("Upload team_stats.csv", type=["csv"], key="up_team_single")

        if st.button("Validate & Save (single)"):
            if not team_display_in or not season_in or not up_ind or not up_team:
                st.error("Please provide team, season, and both CSV files.")
            else:
                team_key = norm_key(team_display_in)
                # Read + validate
                try:
                    df_ind_raw = pd.read_csv(up_ind); df_ind = slugify_columns(df_ind_raw)
                    df_team_raw = pd.read_csv(up_team); df_team = slugify_columns(df_team_raw)
                except Exception as e:
                    st.error(f"Failed to read CSVs: {e}")
                    st.stop()

                miss_ind, miss_team = validate_pair(df_ind, df_team)
                if miss_ind or miss_team:
                    st.error("Validation failed. Please fix the following:")
                    if miss_ind:
                        st.write("**individual_stats_overall.csv**")
                        st.code("\n".join(miss_ind))
                    if miss_team:
                        st.write("**team_stats.csv**")
                        st.code("\n".join(miss_team))
                else:
                    dest_dir = RAW_BASE / team_key / season_in
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    (dest_dir / "individual_stats_overall.csv").write_bytes(up_ind.getvalue())
                    (dest_dir / "team_stats.csv").write_bytes(up_team.getvalue())
                    st.success(f"Saved files to: {dest_dir}")

                    with st.expander("Preview (first 5 rows, individual)"):
                        st.dataframe(df_ind.head(), use_container_width=True)
                    with st.expander("Preview (first 5 rows, team)"):
                        st.dataframe(df_team.head(), use_container_width=True)

#  Multi seasons / one college (ZIP)
    elif mode == "Multiple seasons of one college (ZIP)":
        team_display_in = st.text_input("Team (display name)", placeholder="e.g., Fairfield Stags")
        st.caption(
            "Upload a ZIP with structure: `college/season/{individual_stats_overall.csv, team_stats.csv}`.\n"
            "The older layout `season/{...}` is also accepted."
        )
        up_zip = st.file_uploader("Upload ZIP", type=["zip"], key="up_zip_one_college")

        if st.button("Validate & Save (multi seasons, one college)"):
            if not team_display_in or not up_zip:
                st.error("Please provide team and a ZIP file.")
            else:
                team_key = norm_key(team_display_in)
                dest_root = RAW_BASE / team_key
                dest_root.mkdir(parents=True, exist_ok=True)

                try:
                    zf = zipfile.ZipFile(io.BytesIO(up_zip.getvalue()))
                except Exception as e:
                    st.error(f"Could not open ZIP: {e}")
                    st.stop()

                # Stage to a temp dir first, then copy validated pairs
                temp_extract = ART_DIR / "_tmp_extract_one_college"
                if temp_extract.exists():
                    import shutil; shutil.rmtree(temp_extract)
                temp_extract.mkdir(parents=True, exist_ok=True)

                extracted = extract_zip_file(zf, temp_extract)
                seasons_found = {}
                top_level_colleges = set()

                for p in extracted:
                    rel = p.relative_to(temp_extract)
                    parts = rel.parts
                    if len(parts) < 3:
                        continue

                    college_dir, season, fname = parts[0], parts[1], parts[-1]
                    top_level_colleges.add(norm_key(college_dir))

                    if fname.lower() == "individual_stats_overall.csv":
                        seasons_found.setdefault(season, {})["ind"] = p
                    elif fname.lower() == "team_stats.csv":
                        seasons_found.setdefault(season, {})["team"] = p

                saved_rows = []
                errors = []
                
                for season, files in sorted(seasons_found.items()):
                    ind_p, team_p = files["ind"], files["team"]
                    if not ind_p or not team_p:
                        errors.append(f"{season}: missing required CSV(s).")
                        continue
                    try:
                        df_ind = read_slug_csv(ind_p)
                        df_team = read_slug_csv(team_p)
                        miss_ind, miss_team = validate_pair(df_ind, df_team)
                        if miss_ind or miss_team:
                            msg = []
                            if miss_ind: msg.append("individual_stats_overall.csv: " + "; ".join(miss_ind))
                            if miss_team: msg.append("team_stats.csv: " + "; ".join(miss_team))
                            errors.append(f"{season}: validation failed -> " + " | ".join(msg))
                            continue
                        # Save to college/season
                        dst = dest_root / season
                        dst.mkdir(parents=True, exist_ok=True)
                        (dst / "individual_stats_overall.csv").write_bytes(ind_p.read_bytes())
                        (dst / "team_stats.csv").write_bytes(team_p.read_bytes())
                        saved_rows.append((COLLEGE_MAP.get(team_key, team_display_in), season))
                    except Exception as e:
                        errors.append(f"{season}: {e}")

                if errors:
                    st.warning("Some seasons could not be saved:")
                    st.code("\n".join(errors))

                summarise_saved(saved_rows)

    # Multi seasons / multi colleges (ZIP, college/season/file.csv)
    elif mode == "Multiple seasons of multiple colleges (ZIP)":
        st.caption("Upload a ZIP with structure: `college/season/{individual_stats_overall.csv, team_stats.csv}` (strict).")
        up_zip = st.file_uploader("Upload ZIP", type=["zip"], key="up_zip_multi")

        if st.button("Validate & Save (multi colleges)"):
            if not up_zip:
                st.error("Please upload a ZIP file.")
            else:
                import io, zipfile, shutil

                # Open ZIP
                try:
                    zf = zipfile.ZipFile(io.BytesIO(up_zip.getvalue()))
                except Exception as e:
                    st.error(f"Could not open ZIP: {e}")
                    st.stop()

                # Prepare a clean temp extract dir
                temp_extract = ART_DIR / "_tmp_extract_multi"
                if temp_extract.exists():
                    shutil.rmtree(temp_extract)
                temp_extract.mkdir(parents=True, exist_ok=True)

                # Safely extract CSVs (uses your helper; skips non-CSV)
                extracted = extract_zip_file(zf, temp_extract)

                # Build (college, season) -> {ind, team} pairs from LAST THREE path parts
                pairs = {}          # key: (college_key, season) -> {"ind": Path, "team": Path}
                ignored_paths = []  # paths not matching spec

                for p in extracted:
                    rel = p.relative_to(temp_extract)
                    parts = rel.parts
                    if len(parts) < 3:
                        ignored_paths.append(str(rel))
                        continue

                    college_raw, season, fname = parts[-3], parts[-2], parts[-1]
                    fname_l = fname.lower()
                    if fname_l not in ("individual_stats_overall.csv", "team_stats.csv"):
                        ignored_paths.append(str(rel))
                        continue

                    # Optional: enforce a strict season name pattern (uncomment to require YYYY-YY)
                    # import re
                    # if not re.match(r"^\d{4}-\d{2}$", season):
                    #     ignored_paths.append(str(rel))
                    #     continue

                    college_key = norm_key(college_raw)
                    entry = pairs.setdefault((college_key, season), {"ind": None, "team": None})
                    if fname_l == "individual_stats_overall.csv":
                        entry["ind"] = p
                    else:
                        entry["team"] = p

                # Validate and save each pair
                saved_rows = []
                errors = []

                for (college_key, season), files in sorted(pairs.items()):
                    ind_p, team_p = files["ind"], files["team"]
                    display_college = COLLEGE_MAP.get(college_key, college_key)

                    if not ind_p or not team_p:
                        missing = []
                        if not ind_p:  missing.append("individual_stats_overall.csv")
                        if not team_p: missing.append("team_stats.csv")
                        errors.append(f"{display_college} / {season}: missing {', '.join(missing)}.")
                        continue

                    # Read + validate columns
                    try:
                        df_ind = read_slug_csv(ind_p)
                        df_team = read_slug_csv(team_p)
                    except Exception as e:
                        errors.append(f"{display_college} / {season}: failed to read CSVs -> {e}")
                        continue

                    miss_ind, miss_team = validate_pair(df_ind, df_team)
                    if miss_ind or miss_team:
                        msg = []
                        if miss_ind:  msg.append("individual_stats_overall.csv: " + "; ".join(miss_ind))
                        if miss_team: msg.append("team_stats.csv: " + "; ".join(miss_team))
                        errors.append(f"{display_college} / {season}: validation failed -> " + " | ".join(msg))
                        continue

                    # Save files to RAW_BASE/college/season
                    try:
                        dst = RAW_BASE / college_key / season
                        dst.mkdir(parents=True, exist_ok=True)
                        # Overwrite intentionally if duplicates exist in ZIP
                        (dst / "individual_stats_overall.csv").write_bytes(ind_p.read_bytes())
                        (dst / "team_stats.csv").write_bytes(team_p.read_bytes())
                        saved_rows.append((COLLEGE_MAP.get(college_key, college_key), season))
                    except Exception as e:
                        errors.append(f"{display_college} / {season}: failed to save -> {e}")

                # Report out-of-spec paths
                if ignored_paths:
                    st.info("Ignored files/folders not matching the strict `college/season/file.csv` layout (extra roots are fine):")
                    st.code("\n".join(sorted(ignored_paths)))

                # Errors & success summary
                if errors:
                    st.warning("Some (college, season) pairs could not be saved:")
                    st.code("\n".join(errors))

                summarise_saved(saved_rows)


    st.markdown("---")
    # Keep your existing pipeline buttons
    if st.button("Run pipeline now on all data"):
        rid = run_pipeline()
        if rid:
            st.success(f"Pipeline completed: {rid}")
            st.info("Click Finish button to update results.")

    if st.button("Finish"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Refreshed. Go to the 'Roster' tab to view new data.")

    