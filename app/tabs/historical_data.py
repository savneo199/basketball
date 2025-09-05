import streamlit as st
import pandas as pd
import json
from pathlib import Path
from helpers.helpers import latest_artifacts, load_duckdb, rename_columns, load_json_file, COLLEGE_MAP, COLLEGE_MAP_INV
from helpers.archetype_positions import normalize_position, positions_for_archetype
from helpers.court_builder import build_lineup_labels, make_lineup_figure


def render():
    st.subheader("Roster & Metrics")
    paths = latest_artifacts()
    if not paths or not paths["processed"].exists():
        st.info("Run pipeline to populate processed data.")
        return

    df = load_duckdb(paths["processed"]).copy()
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
            "player_number_ind", "player_ind", "mins_per_game", "Archetype", "pts_per_game",
            "ast_per_game", "reb_per_game", "stl_per_game", "eFG_pct_ind"
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

    def assign_slots_by_position(df_top5: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Return (labels, slots_order) aligned to ['PG','SG','SF','PF','C'] using player positions.
        If multiple players map to the same slot, we place the highest-minutes one in that slot
        and cascade others to the next best available slot.
        """
        slots = ["PG", "SG", "SF", "PF", "C"]
        assigned: dict[str, pd.Series] = {}
        leftovers: list[pd.Series] = []

        # infer from archetype
        for _, row in df_top5.iterrows():
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
        labels, slots_order = assign_slots_by_position(top5)
        fig = make_lineup_figure(labels, slots_order=slots_order)
        st.plotly_chart(fig, use_container_width=True)


    # Player comparison
    st.markdown("---")
    st.subheader("Player comparison (up to 2)")

    # Toggle scope
    cross_scope = st.checkbox(
        "Compare across different teams and seasons",
        value=False,
        help="When on, you can select players from the entire dataset; otherwise only from the current Team/Season."
    )

    # Decide data source current filter (team+season) or full dataset
    source_df = df.copy() if cross_scope else filt.copy()

    # Guard rails
    if source_df.empty or "player_ind" not in source_df.columns:
        st.info("No players available for comparison.")
    else:
        # Ensure helpful display columns exist
        if "college_display" not in source_df.columns and "college" in source_df.columns:
            source_df[" college_norm"] = source_df["college"].astype(str).str.strip().str.lower()
            source_df["college_display"] = source_df[" college_norm"].map(COLLEGE_MAP).fillna(source_df["college"])

        # Build unique player options, labeling with team + season
        key_cols = []
        if " college_norm" in source_df.columns:
            key_cols.append(" college_norm")
        if "season" in source_df.columns:
            key_cols.append("season")
        key_cols.append("player_ind")

        # A safe minutes column for tie-breaking
        minutes_col_all = None
        if "minutes_tot_ind" in source_df.columns:
            minutes_col_all = "minutes_tot_ind"
        else:
            for alt in ["minutes_tot", "minutes", "mins"]:
                if alt in source_df.columns:
                    minutes_col_all = alt
                    break

        # Remove duplicate rows into one row per key for the selector display 
        if key_cols:
            # sort players
            if minutes_col_all:
                source_sorted = source_df.sort_values(minutes_col_all, ascending=False)
            else:
                source_sorted = source_df.copy()
            dedup = source_sorted.drop_duplicates(subset=key_cols, keep="first")
        else:
            dedup = source_df.copy()

        # Build label for each option
        def label_row(r):
            team_name = r.get("college_display", r.get("college", ""))
            season_name = r.get("season", "")
            player_name = r.get("player_ind", "Player")
            player_archetype = r.get("Archetype", "Unknown")
            if cross_scope:
                return f"{player_name} — {team_name} ({season_name} - {player_archetype})"
            else:
                return f"{player_name} - ({player_archetype})"

        dedup = dedup.copy()
        dedup["__key_tuple__"] = dedup.apply(
            lambda r: (
                r.get(" college_norm", None),
                r.get("season", None),
                r.get("player_ind", None)
            ), axis=1
        )
        dedup["__label__"] = dedup.apply(label_row, axis=1)

        # Numeric metrics available in the chosen scope
        numeric_cols_scope = [
            c for c in source_df.columns
            if pd.api.types.is_numeric_dtype(source_df[c]) and c not in {"season"}  # keep season numeric out by default
        ]

        # Default metric candidates 
        default_metric_candidates = [
            "pts_per_game", "ast_per_game", "reb_per_game", 
            "stl_per_game", "eFG_pct_ind", "to_per_game"
        ]
        default_metrics = [c for c in default_metric_candidates if c in numeric_cols_scope]
        if not default_metrics:
            default_metrics = numeric_cols_scope[:6]

        # Build label mapping for the metrics using your rename_columns
        # Create a dummy frame with those numeric columns to get renamed headers
        dummy_cols_df = pd.DataFrame(columns=numeric_cols_scope)
        pretty_dummy = rename_columns(dummy_cols_df.copy())
        pretty_map = {orig: pretty for orig, pretty in zip(numeric_cols_scope, pretty_dummy.columns)}
        pretty_to_orig = {v: k for k, v in pretty_map.items()}

        # Default stats
        pretty_defaults = [pretty_map[c] for c in default_metrics if c in pretty_map]

        with st.container():
            c1, c2 = st.columns([2, 2])
            with c1:
                selected_labels = st.multiselect(
                    "Choose up to two players",
                    options=sorted(dedup["__label__"].tolist()),
                    max_selections=2
                )
            with c2:
                pretty_metric_choices = st.multiselect(
                    "Metrics to compare",
                    options=[pretty_map[c] for c in numeric_cols_scope],
                    default=pretty_defaults or [pretty_map[c] for c in numeric_cols_scope[:6]],
                    help="Select up to six metrics to compare.",
                    max_selections=6
                )
        
                st.caption("Tip: Select Players with similar archetypes for better comparisons.")

        if not selected_labels:
            st.info("Select one or two players to compare.")
        elif not pretty_metric_choices:
            st.info("Choose at least one metric.")
        else:
            # Map selected labels back to keys
            sel = dedup.loc[dedup["__label__"].isin(selected_labels), ["__key_tuple__"]].drop_duplicates()
            selected_keys = sel["__key_tuple__"].tolist()

            # Map pretty labels back to original column names
            metrics = [pretty_to_orig[p] for p in pretty_metric_choices if p in pretty_to_orig]

            # Pull exact rows for each selected player key from the full df (so we honor cross-scope)
            rows = []
            for (cnorm, seas, pname) in selected_keys:
                sub = df.copy()
                if cnorm is not None and " college_norm" in sub.columns:
                    sub = sub.loc[sub[" college_norm"] == cnorm]
                if seas is not None and "season" in sub.columns:
                    sub = sub.loc[sub["season"] == seas]
                sub = sub.loc[sub["player_ind"] == pname]
                if sub.empty:
                    continue
                if minutes_col_all and minutes_col_all in sub.columns:
                    rows.append(sub.sort_values(minutes_col_all, ascending=False).iloc[0])
                else:
                    rows.append(sub.iloc[0])

            if not rows:
                st.info("Could not locate the selected players in the dataset.")
            else:
                compare_cols = ["player_ind", "college_display", "season"] + metrics
                comp_df = pd.DataFrame(rows)
                # ensure display columns exist
                if "college_display" not in comp_df.columns and "college" in comp_df.columns:
                    comp_df[" college_norm"] = comp_df["college"].astype(str).str.strip().str.lower()
                    comp_df["college_display"] = comp_df[" college_norm"].map(COLLEGE_MAP).fillna(comp_df["college"])
                comp_df = comp_df[[c for c in compare_cols if c in comp_df.columns]].copy()

                # Display with pretty column headers
                comp_display = rename_columns(comp_df.copy())
                st.dataframe(comp_display, use_container_width=True)

                st.download_button(
                    "Download CSV (comparison)",
                    data=comp_display.to_csv(index=False).encode(),
                    file_name=f"comparison_{'cross' if cross_scope else selected_norm}_{season if not cross_scope else 'multi'}.csv",
                    mime="text/csv",
                )

                # Radar chart
                scale_df = df  # whole dataset

                # Compute min-max per metric on the chosen scaling frame
                scale_min = scale_df[metrics].min(numeric_only=True)
                scale_max = scale_df[metrics].max(numeric_only=True)
                scale_denom = (scale_max - scale_min).replace(0, 1)

                # Build scaled values 0..1 for the selected players
                scaled = comp_df.copy()
                for m in metrics:
                    scaled[m] = pd.to_numeric(scaled[m], errors="coerce")
                    scaled[m] = (scaled[m] - scale_min[m]) / scale_denom[m]

                # Pretty theta labels in the original metric order selected by the coach
                pretty_theta = [pretty_map[m] for m in metrics]

                import plotly.graph_objects as go
                fig_radar = go.Figure()
                for _, row in scaled.iterrows():
                    r_vals = row[metrics].fillna(0).tolist()
                    r_vals += [r_vals[0]]
                    theta = pretty_theta + [pretty_theta[0]]
                    legend_name = row.get("player_ind", "Player")
                    # If cross-scope, enrich legend with team/season
                    if cross_scope:
                        legend_name = f"{row.get('player_ind','Player')} — {row.get('college_display','')} ({row.get('season','')})"
                    fig_radar.add_trace(go.Scatterpolar(
                        r=r_vals,
                        theta=theta,
                        fill='toself',
                        name=legend_name
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    margin=dict(l=10, r=10, t=30, b=10),
                    title="Radar comparison"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.write("##")

    # Notes
    st.write("##")
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
