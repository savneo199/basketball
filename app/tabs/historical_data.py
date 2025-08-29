# tabs/historical_data.py
import streamlit as st
import pandas as pd
import json
from pathlib import Path

from helpers.helpers import (
    latest_artifacts,
    load_parquet,
    rename_columns,
    load_json_file,
    COLLEGE_MAP,
    COLLEGE_MAP_INV,
)
from helpers.archetype_positions import normalize_position, positions_for_archetype
from helpers.court_builder import make_lineup_figure
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go

CART_KEY = "compare_cart"  # session key for the compare cart


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

    # Team + season selectors
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

    # Filter for this team+season
    filt = df[(df[" college_norm"] == selected_norm) & (df["season"] == season)].copy()

    # ---------------- Roster table (with inline compare checkbox) ----------------
    if show_adv:
        drop_cols = [c for c in [" college_norm", "college_display", "college", "season"] if c in filt.columns]
        view = filt.drop(columns=drop_cols).copy()
    else:
        minimal_cols = [
            "player_ind", "player_number_ind", "mins_per_game", "Archetype", "pts_per_game",
            "ast_per_game", "reb_per_game", "stl_per_game", "blk_per_game", "eFG_pct", "gp_ind",
            # Optional: "position",
        ]
        cols = [c for c in minimal_cols if c in filt.columns]
        view = filt[cols].copy()

    # Add compare checkbox to THIS table
    view.insert(0, "Compare?", False)

    # Pretty headers for display
    view_display = rename_columns(view.copy())

    # Render with data_editor so the checkbox is editable (keeps single-table UX)
    edited = st.data_editor(
        view_display,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Compare?": st.column_config.CheckboxColumn(help="Tick up to 3 players to compare"),
        },
    )

    # Clean download (no compare column)
    download_display = view_display.drop(columns=[c for c in ["Compare?"] if c in view_display.columns])
    st.download_button(
        "Download CSV (current selection)",
        data=download_display.to_csv(index=False).encode(),
        file_name=f"{team_display}_{season}_players.csv",
        mime="text/csv",
    )

    # ---------------- Possible lineup (render robustly) ----------------
    st.markdown("---")
    st.subheader("Possible lineup")

    # Choose a minutes column fallback chain
    minutes_col = None
    for cand in ["minutes_tot_ind", "minutes_tot", "minutes", "mins_per_game"]:
        if cand in filt.columns:
            minutes_col = cand
            break

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

    if not minutes_col:
        st.info("Not enough data to build a lineup (need minutes).")
    else:
        # Choose top5 by the available minutes metric
        top5 = filt.sort_values(minutes_col, ascending=False).head(5).reset_index(drop=True)
        rows, slots_order = assign_slots_by_position(top5)

        # Build label, jersey numbers, and hover stats
        labels, numbers, stats_list, names_for_click = [], [], [], []
        for r in rows:
            name = r.get("player_ind", "Player")
            arch = r.get("Archetype", "")
            labels.append(f"{name} ({arch})" if arch else str(name))

            # jersey number
            num = ""
            if "player_number_ind" in r.index:
                try:
                    num = str(int(float(r["player_number_ind"])))
                except Exception:
                    num = str(r["player_number_ind"])
            numbers.append(num)

            # eFG% may be 0–1; convert to %
            efg = r.get("eFG_pct")
            if pd.notna(efg):
                try:
                    efg = float(efg)
                    if efg <= 1.0:
                        efg *= 100.0
                except Exception:
                    efg = None
            else:
                efg = None

            stats_list.append({
                "PTS/Game": r.get("pts_per_game", 0.0),
                "AST/Game": r.get("ast_per_game", 0.0),
                "REB/Game": r.get("reb_per_game", 0.0),
                "STL/Game": r.get("stl_per_game", 0.0),
                "BLK/Game": r.get("blk_per_game", 0.0),
                "eFG%": efg,
            })
            names_for_click.append(str(name))

        fig = make_lineup_figure(
            labels,
            title=f"Possible lineup for {team_display} ({season})",
            slots_order=slots_order,
            numbers=numbers,
            stats=stats_list,
        )

        # Use unique keys so the figure doesn't "disappear" when other Plotly figs render
        lineup_plotly_key = f"lineup_click_{selected_norm}_{season}"

        try:
            selected = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                select_event=False,
                key=lineup_plotly_key,
            )
            if selected:
                idx = selected[0].get("pointIndex", selected[0].get("pointNumber", 0))
                idx = int(idx)
                if 0 <= idx < len(stats_list):
                    s = stats_list[idx]
                    nm = names_for_click[idx]
                    st.markdown(f"**{nm} — quick stats**")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("PTS/Game", f"{float(s['PTS/Game']):.3f}")
                    c2.metric("AST/Game", f"{float(s['AST/Game']):.3f}")
                    c3.metric("REB/Game", f"{float(s['REB/Game']):.3f}")
                    c4.metric("BLK/Game", f"{float(s['BLK/Game']):.3f}")
                    efgtxt = f"{float(s['eFG%']):.3f}%" if s['eFG%'] is not None else "—"
                    c5.metric("eFG%", efgtxt)
            else:
                st.caption("Tip: click a circle to pin a stat card; hover for details.")
        except Exception:
            st.plotly_chart(fig, use_container_width=True, key=f"lineup_chart_{selected_norm}_{season}")
            st.caption("Tip: hover a circle to see stats. (Install `streamlit-plotly-events` to enable click.)")

    # ---------------- Inline comparison panel below roster ----------------
    st.markdown("---")
    st.markdown("#### Comparison")

    # Build / init cart
    if CART_KEY not in st.session_state:
        st.session_state[CART_KEY] = []

    # After rename, the display name for player is "Player Name"
    PLAYER_NAME_COL = "Player Name" if "Player Name" in edited.columns else "player_ind"

    # When coach clicks "Add checked", push to cart
    if st.button("Add checked to compare"):
        checked = edited[edited["Compare?"] == True] if "Compare?" in edited.columns else edited.iloc[0:0]
        if checked.empty:
            st.info("No rows were checked.")
        else:
            # College and season are constant in this view
            college_val = str(filt["college"].iloc[0]) if "college" in filt.columns and not filt.empty else ""
            cart = st.session_state[CART_KEY]
            for _, row in checked.iterrows():
                name = str(row.get(PLAYER_NAME_COL, "")).strip()
                if not name:
                    continue
                item = {"player_ind": name, "season": str(season), "college": college_val}
                # Max 3 + unique by (name, season, college)
                if len(cart) >= 3:
                    break
                if not any(
                    (c.get("player_ind",""), c.get("season",""), c.get("college",""))
                    == (item["player_ind"], item["season"], item["college"])
                    for c in cart
                ):
                    cart.append(item)
            st.session_state[CART_KEY] = cart
            if len(cart) >= 3:
                st.warning("Compare limit is 3 players. Extra selections were ignored.")

    cart = st.session_state[CART_KEY]
    if cart:
        cA, cB = st.columns([1, 4])
        with cA:
            if st.button("Clear all"):
                st.session_state[CART_KEY] = []
                st.rerun()
        with cB:
            chip_cols = st.columns(len(cart))
            for i, it in enumerate(list(cart)):
                with chip_cols[i]:
                    st.caption(f"{it['player_ind']} — {it['college']} ({it['season']})")
                    if st.button("Remove", key=f"rm_{i}"):
                        st.session_state[CART_KEY].pop(i)
                        st.rerun()

        # Fetch full rows from global df
        def _num(v):
            try:
                return float(v) if v is not None and not pd.isna(v) else 0.0
            except Exception:
                return 0.0

        rows = []
        for it in st.session_state[CART_KEY]:
            mask = (
                (df["player_ind"].astype(str) == it["player_ind"]) &
                (df["season"].astype(str) == it["season"]) &
                (df["college"].astype(str) == it["college"])
            )
            sub = df[mask]
            if not sub.empty:
                rows.append(sub.iloc[0])

        if rows:
            # KPI strip per player
            for r in rows:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("PTS/G", f"{_num(r.get('pts_per_game')):.1f}")
                c2.metric("REB/G", f"{_num(r.get('reb_per_game')):.1f}")
                c3.metric("AST/G", f"{_num(r.get('ast_per_game')):.1f}")
                c4.metric("STL/G", f"{_num(r.get('stl_per_game')):.1f}")
                c5.metric("BLK/G", f"{_num(r.get('blk_per_game')):.1f}")
                st.caption(f"{r.get('player_ind','')} — {r.get('college','')} ({r.get('season','')})")

            # Radar chart (normalized per-metric across selected)
            COMPARE_METRICS = [
                ("pts_per_game", "PTS/G"),
                ("reb_per_game", "REB/G"),
                ("ast_per_game", "AST/G"),
                ("stl_per_game", "STL/G"),
                ("blk_per_game", "BLK/G"),
                ("eFG_pct", "eFG%"),
                ("USG_pct", "USG%"),
            ]
            cats = [n for _, n in COMPARE_METRICS]
            fig = go.Figure()

            max_by_metric = []
            for k, _ in COMPARE_METRICS:
                vals = []
                for rr in rows:
                    v = _num(rr.get(k))
                    if k == "eFG_pct" or k.endswith("_pct"):
                        if v <= 1.0:
                            v *= 100.0
                    vals.append(v)
                max_by_metric.append(max(vals) if vals else 1.0)

            for rr in rows:
                vals = []
                for (k, _), vmax in zip(COMPARE_METRICS, max_by_metric):
                    v = _num(rr.get(k))
                    if k == "eFG_pct" or k.endswith("_pct"):
                        if v <= 1.0:
                            v *= 100.0
                    vals.append(0.0 if vmax == 0 else v / vmax)
                vals.append(vals[0])
                label = f"{rr.get('player_ind','')} — {rr.get('college','')} ({rr.get('season','')})"
                fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill="toself", name=label))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=520, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"cmp_radar_{selected_norm}_{season}")

            # Spacer between radar and table
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

            # Side-by-side table (pretty headers)
            show_cols = [
                "player_ind","college","season","position","Archetype",
                "pts_per_game","reb_per_game","ast_per_game","stl_per_game","blk_per_game",
                "eFG_pct","USG_pct"
            ]
            table = pd.DataFrame([{c: r.get(c) for c in show_cols if c in r.index} for r in rows])

            if "eFG_pct" in table.columns:
                table["eFG_pct"] = table["eFG_pct"].apply(lambda v: (_num(v)*100.0 if _num(v) <= 1 else _num(v)))
            if "USG_pct" in table.columns:
                table["USG_pct"] = table["USG_pct"].apply(_num)

            pretty_table = rename_columns(table.copy())
            st.dataframe(pretty_table, use_container_width=True)
        else:
            st.info("No matching rows found for items in the compare cart.")
    else:
        st.caption("Tip: tick **Compare?** for up to 3 players and click **Add checked to compare**.")

    # ---------------- Notes ----------------
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
