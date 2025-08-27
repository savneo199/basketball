import streamlit as st
import pandas as pd
import json

from helpers import latest_artifacts, load_parquet, rename_columns, COLLEGE_MAP, COLLEGE_MAP_INV

def render():
    st.subheader("Roster & Metrics")
    paths = latest_artifacts()
    if not paths or not paths["processed"].exists():
        st.info("Run pipeline to populate processed data.")
        return

    df = load_parquet(paths["processed"]).copy()
    if "college" in df.columns:
        df[" college_norm"] = df["college"].astype(str).str.strip().str.lower()
        df["college_display"] = df[" college_norm"].map(COLLEGE_MAP).fillna(df["college"])
    else:
        st.info("College names not found.")
        return

    if "season" not in df.columns:
        st.warning("Expected column 'season' not found in processed data.")
        return

    TEAM_COL_DISPLAY = "college_display"
    teams = sorted(df[TEAM_COL_DISPLAY].dropna().unique().tolist())
    team_display = st.selectbox("Team", teams, index=0 if teams else None)
    selected_norm = COLLEGE_MAP_INV.get(team_display, str(team_display).strip().lower()) if team_display else None

    seasons_for_team = sorted(
        df.loc[df[" college_norm"] == selected_norm, "season"].dropna().unique().tolist()
    ) if team_display else []

    if not seasons_for_team:
        st.info("No seasons found for the selected team.")
        return

    season = st.selectbox("Season", seasons_for_team, index=0 if seasons_for_team else None)
    show_adv = st.checkbox("Show advanced metrics", value=False)

    if not (team_display and season):
        st.info("Select both Team and Season to view the roster.")
        return

    filt = df[(df[" college_norm"] == selected_norm) & (df["season"] == season)].copy()
    if show_adv:
        drop_cols = [c for c in [" college_norm", "college_display", "college", "season"] if c in filt.columns]
        view = filt.drop(columns=drop_cols).copy()
    else:
        minimal_cols = [
            "player_ind", "player_number_ind", "minutes_tot_ind", "Archetype", "scoring_pts_ind",
            "ast_ind", "rebounds_tot_ind", "stl_ind", "gp_ind"
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
    from pathlib import Path
    from helpers import load_json_file
    notes_file = Path("app_notes.json")
    notes = load_json_file(notes_file) if notes_file.exists() else {}
    key = f"{team_display}__{season}"
    existing_text = notes.get(key, "")
    txt = st.text_area("Write notes for this team/season", value=existing_text, height=160, key="notes_roster")
    if st.button("Save notes", key="save_notes_roster"):
        notes[key] = txt
        notes_file.write_text(json.dumps(notes, indent=2))
        st.success("Notes saved")
