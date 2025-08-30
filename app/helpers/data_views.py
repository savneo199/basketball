# helpers/data_views.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------- Common helpers ----------
def _as_str_path(p: Path | str) -> str:
    return str(p if isinstance(p, Path) else Path(p))

def _escape_duckdb_string(s: str) -> str:
    return s.replace("'", "''")

def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa
        return True
    except Exception:
        return False

# ---------- DuckDB connection (fast path) ----------
@st.cache_resource(show_spinner=False)
def _duckdb_conn(processed_path: Path):
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA enable_object_cache=true")
    p = Path(processed_path)
    pstr = _escape_duckdb_string(p.as_posix())
    try:
        con.execute(f"""
            CREATE OR REPLACE VIEW processed AS
            SELECT * FROM parquet_scan('{pstr}', union_by_name=true)
        """)
    except Exception:
        con.execute(f"""
            CREATE OR REPLACE VIEW processed AS
            SELECT * FROM read_parquet('{pstr}', union_by_name=true)
        """)
    return con

def _duckdb_existing_columns(con) -> set[str]:
    try:
        df = con.execute("DESCRIBE processed").df()
        if "column_name" in df.columns:
            return set(df["column_name"].astype(str).tolist())
        return set(con.execute("SELECT * FROM processed LIMIT 0").df().columns)
    except Exception:
        return set()

# ---------- Safe PyArrow fallback ----------
@st.cache_data(ttl=600, show_spinner=False)
def parquet_columns(processed_path: Path) -> list[str]:
    path_str = _as_str_path(processed_path)
    try:
        import pyarrow.dataset as ds
        return list(ds.dataset(path_str).schema.names)
    except Exception:
        try:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(path_str).schema_arrow.names)
        except Exception:
            try:
                return list(pd.read_parquet(path_str).columns)
            except Exception:
                return []

def _safe_read_parquet(processed_path: Path, desired_cols: list[str]) -> pd.DataFrame:
    exist = set(parquet_columns(processed_path))
    use_cols = [c for c in desired_cols if c in exist]
    if not use_cols:
        return pd.DataFrame(columns=[])
    return pd.read_parquet(_as_str_path(processed_path), columns=use_cols)

# ---------- Manifest (team/season index) ----------
def _default_manifest_path(processed_path: Path) -> Path:
    # store next to processed parquet
    p = Path(processed_path)
    return p.with_name("processed_manifest.parquet")

@st.cache_resource(show_spinner=False)
def _build_manifest(processed_path: Path) -> Path | None:
    """Create a tiny manifest with (college_norm, season, college_sample) and save to parquet."""
    out = _default_manifest_path(processed_path)
    try:
        if _has_duckdb():
            con = _duckdb_conn(processed_path)
            df = con.execute("""
                SELECT
                  lower(trim(college))::VARCHAR AS college_norm,
                  cast(season AS VARCHAR)      AS season,
                  any_value(college)::VARCHAR  AS college_sample
                FROM processed
                WHERE college IS NOT NULL AND season IS NOT NULL
                GROUP BY 1,2
            """).df()
        else:
            # Arrow fallback (one-time)
            cols = ["college", "season"]
            df_all = _safe_read_parquet(processed_path, cols)
            if df_all.empty:
                return None
            df = df_all.dropna(subset=["college", "season"]).copy()
            df["college_norm"] = df["college"].astype(str).str.strip().str.lower()
            df["season"] = df["season"].astype(str)
            df["college_sample"] = df["college"].astype(str)
            df = df[["college_norm", "season", "college_sample"]].drop_duplicates()

        if df.empty:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def _load_manifest(processed_path: Path) -> pd.DataFrame:
    p = _default_manifest_path(processed_path)
    if not p.exists():
        p2 = _build_manifest(processed_path)
        if p2 is None or not Path(p2).exists():
            return pd.DataFrame(columns=["college_norm", "season", "college_sample"])
    try:
        df = pd.read_parquet(p)
        # normalize types
        for c in ["college_norm", "season", "college_sample"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        return df
    except Exception:
        return pd.DataFrame(columns=["college_norm", "season", "college_sample"])

# ---------- Public, cached API (selectors use manifest) ----------
@st.cache_data(ttl=600, show_spinner=False)
def list_team_choices(processed_path: Path, college_map: dict[str, str]) -> list[str]:
    man = _load_manifest(processed_path)
    if man.empty:
        # last-resort fallback
        df = _safe_read_parquet(processed_path, ["college"])
        if df.empty:
            return []
        df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
        disp = df[" college_norm"].map(college_map).fillna(df["college"].astype(str))
        return sorted(disp.dropna().unique().tolist())

    disp = man["college_norm"].map(college_map).fillna(man["college_sample"])
    return sorted(disp.dropna().unique().tolist())

@st.cache_data(ttl=600, show_spinner=False)
def list_seasons_for_team(processed_path: Path, college_norm: str) -> list[str]:
    man = _load_manifest(processed_path)
    if not man.empty:
        return (
            man.loc[man["college_norm"] == str(college_norm), "season"]
               .dropna().astype(str).sort_values().unique().tolist()
        )
    # fallback
    df = _safe_read_parquet(processed_path, ["college", "season"])
    if df.empty:
        return []
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    return (
        df.loc[df[" college_norm"] == str(college_norm), "season"]
          .dropna().astype(str).sort_values().unique().tolist()
    )

# ---------- Roster + Compare (DuckDB fast path, Arrow fallback) ----------
@st.cache_data(ttl=600, show_spinner=False)
def team_season_view(processed_path: Path, college_norm: str, season: str) -> pd.DataFrame:
    cols = [
        "college","season","player_ind","player_number_ind","position","Archetype",
        "minutes_tot_ind","mins_per_game","pts_per_game","ast_per_game","reb_per_game",
        "stl_per_game","blk_per_game","eFG_pct","USG_pct","gp_ind"
    ]
    if _has_duckdb():
        try:
            con = _duckdb_conn(processed_path)
            existing = _duckdb_existing_columns(con)
            proj = [c for c in cols if c in existing]
            if not proj:
                return pd.DataFrame(columns=[])
            q = f"""
                SELECT {', '.join(proj)}
                FROM processed
                WHERE lower(trim(college)) = ?
                  AND cast(season as varchar) = ?
            """
            df = con.execute(q, [str(college_norm), str(season)]).df()
            if "college" in df.columns:
                df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
            return df
        except Exception:
            pass
    # fallback
    df = _safe_read_parquet(processed_path, cols)
    if df.empty:
        return df
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower() if "college" in df.columns else ""
    if "season" in df.columns:
        df = df[df["season"].astype(str) == str(season)]
    if " college_norm" in df.columns:
        df = df[df[" college_norm"] == str(college_norm)]
    return df.copy()

@st.cache_data(ttl=600, show_spinner=False)
def rows_for_compare(processed_path: Path, items: list[dict]) -> list[pd.Series]:
    if not items:
        return []
    want = pd.DataFrame(items, columns=["player_ind","season","college"]).dropna()
    if want.empty:
        return []
    cols = [
        "player_ind","season","college","position","Archetype",
        "pts_per_game","reb_per_game","ast_per_game","stl_per_game","blk_per_game",
        "eFG_pct","USG_pct","mins_per_game","minutes_tot_ind","player_number_ind"
    ]
    if _has_duckdb():
        try:
            con = _duckdb_conn(processed_path)
            existing = _duckdb_existing_columns(con)
            proj = [c for c in cols if c in existing]
            if not proj:
                return []
            players = tuple(want["player_ind"].astype(str).unique().tolist())
            seasons = tuple(want["season"].astype(str).unique().tolist())
            colleges = tuple(want["college"].astype(str).unique().tolist())
            ph_players = ",".join(["?"] * len(players)) or "''"
            ph_seasons = ",".join(["?"] * len(seasons)) or "''"
            ph_colleges = ",".join(["?"] * len(colleges)) or "''"
            q = f"""
                SELECT {', '.join(proj)}
                FROM processed
                WHERE player_ind IN ({ph_players})
                  AND cast(season as varchar) IN ({ph_seasons})
                  AND college IN ({ph_colleges})
            """
            params = list(players) + list(seasons) + list(colleges)
            df = con.execute(q, params).df()
            for c in ["player_ind","season","college"]:
                if c in df.columns:
                    df[c] = df[c].astype(str)
            merged = want.merge(df, on=["player_ind","season","college"], how="left", suffixes=("", "_df"))
            out: list[pd.Series] = []
            for _, row in want.iterrows():
                hit = merged[
                    (merged["player_ind"] == str(row["player_ind"])) &
                    (merged["season"] == str(row["season"])) &
                    (merged["college"] == str(row["college"]))
                ]
                if not hit.empty:
                    out.append(hit.iloc[0])
            return out
        except Exception:
            pass
    # fallback
    df = _safe_read_parquet(processed_path, cols)
    if df.empty:
        return []
    for c in ["player_ind","season","college"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    merged = want.merge(df, on=["player_ind","season","college"], how="left", suffixes=("", "_df"))
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
