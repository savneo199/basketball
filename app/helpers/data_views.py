# helpers/data_views.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------------- Common helpers ----------------
def _as_str_path(p: Path | str) -> str:
    return str(p if isinstance(p, Path) else Path(p))

def _escape_duckdb_string(s: str) -> str:
    # Escape single quotes for SQL string literal in DuckDB
    return s.replace("'", "''")


# ---------------- DuckDB connection (fast path) ----------------
def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except Exception:
        return False

@st.cache_resource(show_spinner=False)
def _duckdb_conn(processed_path: Path):
    """
    Persistent DuckDB connection with a VIEW 'processed' over the Parquet file or directory.
    We must inline the file path (no '?' params) inside table functions for CREATE VIEW.
    """
    import duckdb
    con = duckdb.connect()
    try:
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA enable_object_cache=true")
        p = Path(processed_path)
        pstr = _escape_duckdb_string(p.as_posix())

        # Try parquet_scan first (modern); fallback to read_parquet
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
    except Exception:
        # If anything fails during setup, close and re-raise so callers can fallback
        con.close()
        raise
    return con

def _duckdb_existing_columns(con) -> set[str]:
    # DESCRIBE is cheap and avoids scanning data
    try:
        df = con.execute("DESCRIBE processed").df()
        if "column_name" in df.columns:
            return set(df["column_name"].astype(str).tolist())
        # Older DuckDB: fall back to LIMIT 0 projection
        return set(con.execute("SELECT * FROM processed LIMIT 0").df().columns)
    except Exception:
        return set()


# ---------------- Safe PyArrow helpers (fallback path) ----------------
@st.cache_data(ttl=600, show_spinner=False)
def parquet_columns(processed_path: Path) -> list[str]:
    """
    Return the list of column names in the Parquet file or directory.
    """
    path_str = _as_str_path(processed_path)
    try:
        import pyarrow.dataset as ds
        schema = ds.dataset(path_str).schema
        return list(schema.names)
    except Exception:
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path_str)
            return list(pf.schema_arrow.names)
        except Exception:
            try:
                df = pd.read_parquet(path_str)
                return list(df.columns)
            except Exception:
                return []

def _safe_read_parquet(processed_path: Path, desired_cols: list[str]) -> pd.DataFrame:
    exist = set(parquet_columns(processed_path))
    use_cols = [c for c in desired_cols if c in exist]
    if not use_cols:
        return pd.DataFrame(columns=[])
    return pd.read_parquet(_as_str_path(processed_path), columns=use_cols)


# ---------------- Public, cached API (tries DuckDB first, falls back safely) ----------------
@st.cache_data(ttl=600, show_spinner=False)
def list_team_choices(processed_path: Path, college_map: dict[str, str]) -> list[str]:
    # Fast path: DuckDB
    if _has_duckdb():
        try:
            con = _duckdb_conn(processed_path)
            df = con.execute("""
                SELECT DISTINCT college FROM processed
                WHERE college IS NOT NULL
            """).df()
            if df.empty:
                return []
            df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
            disp = df[" college_norm"].map(college_map).fillna(df["college"].astype(str))
            return sorted(disp.dropna().unique().tolist())
        except Exception:
            pass  # fall through to Arrow fallback

    # Fallback: PyArrow/pandas
    df = _safe_read_parquet(processed_path, ["college"])
    if df.empty or "college" not in df.columns:
        return []
    df = df.dropna(subset=["college"]).copy()
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    disp = df[" college_norm"].map(college_map).fillna(df["college"].astype(str))
    return sorted(disp.dropna().unique().tolist())


@st.cache_data(ttl=600, show_spinner=False)
def list_seasons_for_team(processed_path: Path, college_norm: str) -> list[str]:
    if _has_duckdb():
        try:
            con = _duckdb_conn(processed_path)
            df = con.execute("""
                SELECT DISTINCT season
                FROM processed
                WHERE lower(trim(college)) = ?
                  AND season IS NOT NULL
                ORDER BY season
            """, [str(college_norm)]).df()
            return df["season"].astype(str).tolist() if "season" in df.columns else []
        except Exception:
            pass

    df = _safe_read_parquet(processed_path, ["college", "season"])
    if df.empty or not {"college", "season"}.issubset(df.columns):
        return []
    df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
    out = (
        df.loc[df[" college_norm"] == str(college_norm), "season"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return out


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

    # Fallback
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

    want = pd.DataFrame(items, columns=["player_ind", "season", "college"]).dropna()
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

            # Build param placeholders
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
            for c in ["player_ind", "season", "college"]:
                if c in df.columns:
                    df[c] = df[c].astype(str)

            merged = want.merge(df, on=["player_ind", "season", "college"], how="left", suffixes=("", "_df"))

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

    # Fallback
    df = _safe_read_parquet(processed_path, cols)
    if df.empty:
        return []
    for c in ["player_ind", "season", "college"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
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
