import re
import json
import hashlib
from pathlib import Path
import zipfile

import joblib
import pandas as pd
import streamlit as st

from get_paths import ROOT, DATA_DIR, RAW_BASE, ART_DIR


import streamlit as st
# Hashing
def hash_data_folder() -> str:
    h = hashlib.sha256()
    for p in sorted(DATA_DIR.rglob("*.csv")):
        try:
            h.update(p.name.encode())
            h.update(p.read_bytes())
        except Exception:
            pass
    return h.hexdigest()

# Caches
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

# Artifacts
def latest_artifacts():
    latest = ART_DIR / "latest"
    if not latest.exists():
        st.warning("No latest folder found.")
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

# Pretty column names
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

# Upload validation helpers
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

def validate_pair(df_ind: pd.DataFrame, df_team: pd.DataFrame) -> tuple[list, list]:
    return validate_individual_cols(set(df_ind.columns)), validate_team_cols(set(df_team.columns))

def read_slug_csv(path: Path) -> pd.DataFrame:
    return slugify_columns(pd.read_csv(path))

# ZIP utils
def safe_join(base: Path, *parts: str) -> Path:
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
        if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
            continue
        if Path(name).name.startswith("._"):
            continue
        if any(part.startswith(".") for part in Path(name).parts):
            continue

        if zi.is_dir():
            out_dir = safe_join(to_dir, name)
            out_dir.mkdir(parents=True, exist_ok=True)
            continue

        if not is_csv(name):
            continue

        out_path = safe_join(to_dir, name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        extracted.append(out_path)
    return extracted

def summarise_saved(rows: list[tuple[str, str]]):
    if not rows:
        st.info("No valid (college, season) pairs were found.")
        return
    df_sum = pd.DataFrame(rows, columns=["College", "Season"]).drop_duplicates()
    st.success(f"Saved {len(df_sum)} season(s).")
    st.dataframe(df_sum.sort_values(["College", "Season"]), use_container_width=True)
