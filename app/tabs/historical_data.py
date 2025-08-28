import streamlit as st
import pandas as pd
import json
from pathlib import Path
from helpers.helpers import latest_artifacts, load_parquet, rename_columns, load_json_file, COLLEGE_MAP, COLLEGE_MAP_INV
from helpers.archetype_positions import normalize_position, positions_for_archetype
from helpers.court_builder import build_lineup_labels, make_lineup_figure


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

    # Possible lineup map
    st.markdown("---")
    st.subheader("Possible lineup")

    minutes_col = "minutes_tot_ind" if "minutes_tot_ind" in filt.columns else None
    if minutes_col is None:
        for alt in ["minutes_tot", "minutes", "mins"]:
            if alt in filt.columns:
                minutes_col = alt
                break

    def _assign_slots_by_position(df_top5: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Return (labels, slots_order) aligned to ['PG','SG','SF','PF','C'] using player positions.
        If multiple players map to the same slot, we place the highest-minutes one in that slot
        and cascade others to the next best available slot.
        """
        slots = ["PG", "SG", "SF", "PF", "C"]
        assigned: dict[str, pd.Series] = {}
        leftovers: list[pd.Series] = []

        # best attempt: direct normalized position, otherwise infer from archetype
        for _, row in df_top5.iterrows():
            raw_pos = str(row.get("position", "") or "")
            pos = normalize_position(raw_pos)
            if not pos:
                # try infer from archetype if present
                arc = str(row.get("Archetype", "") or "")
                prefs = positions_for_archetype(arc)
                pos = prefs[0] if prefs else ""
            if pos in slots and pos not in assigned:
                assigned[pos] = row
            else:
                leftovers.append(row)

        # cascade leftovers into remaining slots by reasonable preferences
        def pref_chain(pos_guess: str) -> list[str]:
            # reasonable fallbacks by family
            if pos_guess in ("PG", "SG"):   # guards
                return ["PG", "SG", "SF", "PF", "C"]
            if pos_guess == "SF":           # wing
                return ["SF", "PF", "SG", "PG", "C"]
            if pos_guess == "PF":           # big forward
                return ["PF", "C", "SF", "SG", "PG"]
            if pos_guess == "C":            # center
                return ["C", "PF", "SF", "SG", "PG"]
            return ["SF", "PF", "PG", "SG", "C"]  # unknown

        for row in leftovers:
            raw_pos = str(row.get("position", "") or "")
            guess = normalize_position(raw_pos)
            if not guess:
                arc = str(row.get("Archetype", "") or "")
                prefs = positions_for_archetype(arc)
                guess = prefs[0] if prefs else ""
            for s in pref_chain(guess):
                if s in slots and s not in assigned:
                    assigned[s] = row
                    break

        # build ordered labels + slots
        labels, slots_order = [], []
        for s in slots:
            if s in assigned:
                r = assigned[s]
                name = (
                    r["player_ind"] if "player_ind" in r and pd.notna(r["player_ind"])
                    else r.get("player", r.get("player_name", "Player"))
                )
                arch = r.get("Archetype", "")
                labels.append(f"{name} ({arch})" if arch else str(name))
                slots_order.append(s)
        return labels, slots_order

    if minutes_col is None or "player_ind" not in filt.columns:
        st.info("Not enough data to build a lineup (need player names and minutes).")
    else:
        top5 = filt.sort_values(minutes_col, ascending=False).head(5).reset_index(drop=True)
        labels, slots_order = _assign_slots_by_position(top5)
        fig = make_lineup_figure(labels, slots_order=slots_order)
        st.plotly_chart(fig, use_container_width=True)



    st.markdown("---")
    st.subheader("Notes & Comments")

    notes_file = Path("app_notes.json")
    notes = load_json_file(notes_file) if notes_file.exists() else {}
    key = f"{team_display}__{season}"
    existing_text = notes.get(key, "")
    txt = st.text_area("Write notes for this team/season", value=existing_text, height=160, key="notes_roster")
    if st.button("Save notes", key="save_notes_roster"):
        notes[key] = txt
        notes_file.write_text(json.dumps(notes, indent=2))
        st.success("Notes saved")
