import streamlit as st
import pandas as pd
import json
from pathlib import Path
from helpers.helpers import latest_artifacts, load_parquet, rename_columns, load_json_file, COLLEGE_MAP, COLLEGE_MAP_INV
from helpers.archetype_positions import normalize_position, positions_for_archetype
from helpers.court_builder import build_lineup_labels, make_lineup_figure
from streamlit_plotly_events import plotly_events

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
            "player_ind", "player_number_ind", "mins_per_game", "Archetype", "pts_per_game",
            "ast_per_game", "reb_per_game", "stl_per_game", "blk_per_game","eFG_pct", "gp_ind"
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
    # Possible lineup
    st.markdown("---")
    st.subheader("Possible lineup")

    minutes_col = "minutes_tot_ind" 

    def assign_slots_by_position(df_top5: pd.DataFrame):
        slots = ["PG", "SG", "SF", "PF", "C"]
        assigned, leftovers = {}, []

        for _, row in df_top5.iterrows():
            raw_pos = str(row.get("position", "") or "")
            pos = normalize_position(raw_pos)
            if not pos:
                arc = str(row.get("Archetype", "") or "")
                prefs = positions_for_archetype(arc)
                pos = prefs[0] if prefs else ""
            if pos in slots and pos not in assigned:
                assigned[pos] = row
            else:
                leftovers.append(row)

        def pref_chain(pos_guess: str) -> list[str]:
            if pos_guess in ("PG", "SG"):   return ["PG", "SG", "SF", "PF", "C"]
            if pos_guess == "SF":           return ["SF", "PF", "SG", "PG", "C"]
            if pos_guess == "PF":           return ["PF", "C", "SF", "SG", "PG"]
            if pos_guess == "C":            return ["C", "PF", "SF", "SG", "PG"]
            return ["SF", "PF", "PG", "SG", "C"]

        for row in leftovers:
            raw_pos = str(row.get("position", "") or "")
            guess = normalize_position(raw_pos)
            if not guess:
                arc = str(row.get("Archetype", "") or "")
                prefs = positions_for_archetype(arc)
                guess = prefs[0] if prefs else ""
            for s in pref_chain(guess):
                if s not in assigned:
                    assigned[s] = row
                    break

        ordered_slots = [s for s in slots if s in assigned]
        ordered_rows  = [assigned[s] for s in ordered_slots]
        return ordered_rows, ordered_slots

    if minutes_col is None:
        st.info("Not enough data to build a lineup (need minutes).")
    else:
        top5 = filt.sort_values(minutes_col, ascending=False).head(5).reset_index(drop=True)
        rows, slots_order = assign_slots_by_position(top5)

        # Build label, number, stats lists in slot order
        labels, numbers, stats_list, names_for_click = [], [], [], []
        for r in rows:
            name = r["player_ind"] 
            arch = r["Archetype"]
            labels.append(f"{name} ({arch})")

            # get player number
            num = ""
            if "player_number_ind":
                try:
                    num = str(int(float(r["player_number_ind"])))
                except Exception:
                    num = str(r["player_number_ind"])
            numbers.append(num)

            stats_list.append({
            "PTS/Game": r["pts_per_game"],
            "AST/Game": r["ast_per_game"],
            "REB/Game": r["reb_per_game"],
            "STL/Game": r["stl_per_game"],
            "BLK/Game": r["blk_per_game"],
            "eFG%": r["eFG_pct"]*100,
            })
            names_for_click.append(str(name))

        fig = make_lineup_figure(labels, title = "Possible lineup for " + team_display + " (" + season + ")", slots_order=slots_order, numbers=numbers, stats=stats_list)

        # Optional click-to-inspect
        try:
            from streamlit_plotly_events import plotly_events
            selected = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="lineup_click")
            if selected:
                idx = selected[0].get("pointIndex", selected[0].get("pointNumber", 0))
                idx = int(idx)
                if 0 <= idx < len(stats_list):
                    s = stats_list[idx]
                    nm = names_for_click[idx]
                    st.markdown(f"**{nm} — quick stats**")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("PTS/Game", f"{s['PTS/Game']:.3f}")
                    c2.metric("AST/Game", f"{s['AST/Game']:.3f}")
                    c3.metric("REB/Game", f"{s['REB/Game']:.3f}")
                    c4.metric("BLK/Game", f"{s['BLK/Game']:.3f}")
                    c5.metric("eFG%", f"{s['eFG%']:.3f}%" if s['eFG%'] is not None else "—")
            else:
                st.caption("Tip: click a circle to pin a stat card; hover for details.")
        except Exception:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Tip: hover a circle to see stats. (Install `streamlit-plotly-events` to enable click.)")



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
