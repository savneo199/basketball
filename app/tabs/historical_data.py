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


    

    # ==================== Inline Compare (replace the old Compare tab) ====================

    # Where to store selected players globally
    CART_KEY = "compare_cart"
    if CART_KEY not in st.session_state:
        st.session_state[CART_KEY] = []  # list of dicts: {"player_ind","season","college"}

    # Metrics used for comparison
    COMPARE_METRICS = [
        ("pts_per_game", "PTS/G"),
        ("reb_per_game", "REB/G"),
        ("ast_per_game", "AST/G"),
        ("stl_per_game", "STL/G"),
        ("blk_per_game", "BLK/G"),
        ("eFG_pct", "eFG%"),   # 0–1 or 0–100 handled
        ("USG_pct", "USG%"),
    ]

    ID_COLS = ["player_ind", "college", "season", "position", "Archetype", "player_number_ind"]

    def _num(v):
        try:
            return float(v) if v is not None and not pd.isna(v) else 0.0
        except Exception:
            return 0.0

    def _option_label(row: pd.Series) -> str:
        return f"{row.get('player_ind','Player')} — {row.get('college','')} ({row.get('season','')})"

    def _unique_key(row: pd.Series) -> tuple:
        # Use (name, season, college) as a stable unique identifier
        return (str(row.get("player_ind","")), str(row.get("season","")), str(row.get("college","")))

    def _fetch_rows_from_cart(df_all: pd.DataFrame, cart: list[dict]) -> list[pd.Series]:
        rows = []
        for item in cart:
            mask = (
                (df_all.get("player_ind","").astype(str) == item.get("player_ind","")) &
                (df_all.get("season","").astype(str) == item.get("season","")) &
                (df_all.get("college","").astype(str) == item.get("college",""))
            )
            sub = df_all[mask]
            if not sub.empty:
                rows.append(sub.iloc[0])
        return rows

    def _radar_figure(rows: list[pd.Series]):
        import plotly.graph_objects as go
        cats = [name for _, name in COMPARE_METRICS]
        fig = go.Figure()

        # per-metric max for 0–1 normalization
        max_by_metric = []
        for key, _ in COMPARE_METRICS:
            vals = []
            for r in rows:
                v = _num(r.get(key))
                if key == "eFG_pct" or key.endswith("_pct"):
                    if v <= 1.0:
                        v *= 100.0
                vals.append(v)
            max_by_metric.append(max(vals) if vals else 1.0)

        for r in rows:
            vals = []
            for (key, _), vmax in zip(COMPARE_METRICS, max_by_metric):
                v = _num(r.get(key))
                if key == "eFG_pct" or key.endswith("_pct"):
                    if v <= 1.0:
                        v *= 100.0
                vals.append(0.0 if vmax == 0 else v / vmax)
            vals.append(vals[0])  # close shape

            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats + [cats[0]],
                fill="toself",
                name=_option_label(r),
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
                showlegend=True,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    st.markdown("### Compare players")
    enable_compare = st.toggle("Enable compare mode", value=False, help="Check up to 3 players to compare.")

    if enable_compare:
        # Build a small, selectable view from the current filter (team+season)
        # Use raw columns; avoid rename_columns() so we keep consistent keys.
        selectable_cols = [
            "player_ind", "player_number_ind", "position", "Archetype",
            "minutes_tot_ind", "mins_per_game",
            "pts_per_game", "reb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
            "eFG_pct", "USG_pct",
            "college", "season",
        ]
        sel_view = filt[[c for c in selectable_cols if c in filt.columns]].copy()
        if sel_view.empty:
            st.info("No players in this selection.")
        else:
            sel_view.insert(0, "Compare?", False)  # editable checkbox column
            # Use data_editor to allow ticking rows
            edited = st.data_editor(
                sel_view,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Compare?": st.column_config.CheckboxColumn(help="Tick to add to comparison"),
                },
            )

            # Collect checked rows and add to cart (max 3 total)
            if st.button("Add checked to compare"):
                checked = edited[edited["Compare?"] == True]
                if checked.empty:
                    st.info("No rows were checked.")
                else:
                    cart = st.session_state[CART_KEY]
                    for _, r in checked.iterrows():
                        key = _unique_key(r)
                        if len(cart) >= 3:
                            break
                        if not any(
                            (c.get("player_ind",""), c.get("season",""), c.get("college","")) == key
                            for c in cart
                        ):
                            cart.append({
                                "player_ind": key[0],
                                "season": key[1],
                                "college": key[2],
                            })
                    st.session_state[CART_KEY] = cart
                    if len(st.session_state[CART_KEY]) >= 3:
                        st.warning("Compare limit is 3 players. Extra selections were ignored.")

    # Show comparison panel if there are players in the cart
    cart = st.session_state[CART_KEY]
    if cart:
        st.markdown("#### Comparison panel")
        # Buttons to clear or remove individuals
        cc1, cc2 = st.columns([1, 4])
        with cc1:
            if st.button("Clear all"):
                st.session_state[CART_KEY] = []
                st.stop()
        with cc2:
            # Show chips with remove buttons
            chip_cols = st.columns(len(cart))
            for i, item in enumerate(list(cart)):
                with chip_cols[i]:
                    label = f"{item['player_ind']} — {item['college']} ({item['season']})"
                    st.caption(label)
                    if st.button("Remove", key=f"rm_{i}"):
                        st.session_state[CART_KEY].pop(i)
                        st.experimental_rerun()

        # Fetch full rows from the global processed DF (not only current team/season)
        # Use the same processed dataset you loaded earlier in this tab `df`
        rows = _fetch_rows_from_cart(df, st.session_state[CART_KEY])
        if not rows:
            st.info("No matching rows found for items in the cart.")
        else:
            # KPI strip for each
            for r in rows:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("PTS/G", f"{_num(r.get('pts_per_game')):.1f}")
                c2.metric("REB/G", f"{_num(r.get('reb_per_game')):.1f}")
                c3.metric("AST/G", f"{_num(r.get('ast_per_game')):.1f}")
                c4.metric("STL/G", f"{_num(r.get('stl_per_game')):.1f}")
                c5.metric("BLK/G", f"{_num(r.get('blk_per_game')):.1f}")
                st.caption(_option_label(r))

            st.markdown("---")
            # Radar plot
            fig = _radar_figure(rows)
            st.plotly_chart(fig, use_container_width=True)

            # Side-by-side table
            show_cols = ID_COLS + [k for k, _ in COMPARE_METRICS]
            table = pd.DataFrame([{c: r.get(c) for c in show_cols} for r in rows])
            if "eFG_pct" in table.columns:
                table["eFG_pct"] = table["eFG_pct"].apply(lambda v: (_num(v) * 100.0 if _num(v) <= 1 else _num(v)))
            table["USG_pct"] = table.get("USG_pct", pd.Series(dtype=float)).apply(_num)
            st.dataframe(table, use_container_width=True)
    else:
        st.caption("Tip: toggle **Compare mode**, tick up to 3 players, then click **Add checked to compare**.")
    # ==================== /Inline Compare ====================



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
