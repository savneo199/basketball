# tabs/historical_data.py
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go

from helpers.helpers import latest_artifacts, rename_columns, load_json_file, COLLEGE_MAP, COLLEGE_MAP_INV
from helpers.data_views import list_team_choices, list_seasons_for_team, team_season_view, rows_for_compare
from helpers.court_builder import make_lineup_figure
from helpers.perf import timed
from helpers.archetype_positions import normalize_position, positions_for_archetype
from streamlit_plotly_events import plotly_events

CART_KEY = "compare_cart"

def render():
    st.subheader("Roster & Metrics")

    paths = latest_artifacts()
    if not paths or not paths["processed"].exists():
        st.info("Run pipeline to populate processed data.")
        return
    processed_path = paths["processed"]

    # -------------------------------- Controls form --------------------------------
    with st.form("controls", clear_on_submit=False, border=False):
        with timed("team list (manifest)"):
            teams = list_team_choices(processed_path, COLLEGE_MAP)
        team_display = st.selectbox("Team", teams, index=0 if teams else None, key="sel_team")

        selected_norm = COLLEGE_MAP_INV.get(team_display, str(team_display).strip().lower()) if team_display else None

        with timed("seasons for team (manifest)"):
            seasons_for_team = list_seasons_for_team(processed_path, selected_norm) if team_display else []
        season = st.selectbox("Season", seasons_for_team, index=0 if seasons_for_team else None, key="sel_season")

        show_adv = st.checkbox("Show advanced metrics", value=False, key="show_adv")
        interactive = st.checkbox("Interactive charts (click-to-inspect)", value=True, key="interactive")

        applied = st.form_submit_button("Apply")

    # Preserve last applied filters in session to avoid recompute while user is changing inputs
    if "applied_filters" not in st.session_state or applied:
        st.session_state["applied_filters"] = {
            "team_display": team_display,
            "selected_norm": selected_norm,
            "season": season,
            "show_adv": show_adv,
            "interactive": interactive,
        }

    flt = st.session_state["applied_filters"]
    team_display, selected_norm, season, show_adv, interactive = (
        flt["team_display"], flt["selected_norm"], flt["season"], flt["show_adv"], flt["interactive"]
    )

    if not (team_display and season):
        st.info("Select both Team and Season, then press **Apply**.")
        return

    # -------------------------------- Roster table --------------------------------
    with timed("team+season slice"):
        filt = team_season_view(processed_path, selected_norm, season)

    if filt.empty:
        st.warning("No data for this selection.")
        return

    if show_adv:
        view = filt.drop(columns=[c for c in [" college_norm", "college", "season"] if c in filt.columns]).copy()
    else:
        minimal_cols = [
            "player_ind","player_number_ind","mins_per_game","Archetype","pts_per_game",
            "ast_per_game","reb_per_game","stl_per_game","blk_per_game","eFG_pct","gp_ind",
        ]
        view = filt[[c for c in minimal_cols if c in filt.columns]].copy()

    # Compare toggle (editor is heavy; only mount when needed)
    edit_mode = st.toggle("Select players to compare", value=False, help="Turn on to tick players (max 3)")
    if edit_mode:
        view.insert(0, "Compare?", False)
        view_display = rename_columns(view.copy())
        edited = st.data_editor(
            view_display, hide_index=True, use_container_width=True, num_rows="fixed",
            column_config={"Compare?": st.column_config.CheckboxColumn(help="Tick up to 3 players to compare")},
            key=f"editor_{selected_norm}_{season}",
        )
    else:
        view_display = rename_columns(view.copy())
        edited = None
        st.dataframe(view_display, use_container_width=True, height=420, hide_index=True)

    st.download_button(
        "Download CSV (current selection)",
        data=view_display.drop(columns=[c for c in ["Compare?"] if c in view_display.columns]).to_csv(index=False).encode(),
        file_name=f"{team_display}_{season}_players.csv",
        mime="text/csv",
    )

    # -------------------------------- Possible lineup --------------------------------
    st.markdown("---")
    st.subheader("Possible lineup")

    minutes_col = None
    for cand in ["minutes_tot_ind","minutes_tot","minutes","mins_per_game"]:
        if cand in filt.columns:
            minutes_col = cand; break

    def assign_slots_by_position(df_top5: pd.DataFrame):
        slots = ["PG","SG","SF","PF","C"]; assigned, leftovers = {}, []
        for _, row in df_top5.iterrows():
            raw_pos = str(row.get("position","") or "")
            pos = normalize_position(raw_pos) or (positions_for_archetype(str(row.get("Archetype","")) or "")[:1] or [""])[0]
            if pos in slots and pos not in assigned: assigned[pos] = row
            else: leftovers.append(row)
        def pref_chain(pos_guess: str) -> list[str]:
            if pos_guess in ("PG","SG"): return ["PG","SG","SF","PF","C"]
            if pos_guess == "SF":        return ["SF","PF","SG","PG","C"]
            if pos_guess == "PF":        return ["PF","C","SF","SG","PG"]
            if pos_guess == "C":         return ["C","PF","SF","SG","PG"]
            return ["SF","PF","PG","SG","C"]
        for row in leftovers:
            guess = normalize_position(str(row.get("position",""))) or (positions_for_archetype(str(row.get("Archetype","")) or "")[:1] or [""])[0]
            for s in pref_chain(guess):
                if s not in assigned: assigned[s] = row; break
        ordered = [s for s in slots if s in assigned]
        return [assigned[s] for s in ordered], ordered

    if not minutes_col:
        st.info("Not enough data to build a lineup (need minutes).")
    else:
        top5 = filt.sort_values(minutes_col, ascending=False).head(5).reset_index(drop=True)
        rows, slots_order = assign_slots_by_position(top5)

        labels, numbers, stats_list, names_for_click = [], [], [], []
        for r in rows:
            name = r.get("player_ind","Player")
            arch = r.get("Archetype","")
            labels.append(f"{name} ({arch})" if arch else str(name))
            num = ""
            if "player_number_ind" in r.index:
                try: num = str(int(float(r["player_number_ind"])))
                except Exception: num = str(r["player_number_ind"])
            numbers.append(num)
            efg = r.get("eFG_pct")
            try:
                efg = float(efg) if efg is not None else None
                if efg is not None and efg <= 1.0: efg *= 100.0
            except Exception:
                efg = None
            stats_list.append({
                "PTS/Game": r.get("pts_per_game",0.0),
                "AST/Game": r.get("ast_per_game",0.0),
                "REB/Game": r.get("reb_per_game",0.0),
                "STL/Game": r.get("stl_per_game",0.0),
                "BLK/Game": r.get("blk_per_game",0.0),
                "eFG%": efg,
            })
            names_for_click.append(str(name))

        fig = make_lineup_figure(
            labels, title=f"Possible lineup for {team_display} ({season})",
            slots_order=slots_order, numbers=numbers, stats=stats_list,
        )

        key_base = f"lineup_{selected_norm}_{season}"
        if interactive:
            selected = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=key_base+"_click")
            if selected:
                idx = int(selected[0].get("pointIndex", selected[0].get("pointNumber", 0)))
                if 0 <= idx < len(stats_list):
                    s = stats_list[idx]; nm = names_for_click[idx]
                    st.markdown(f"**{nm} — quick stats**")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("PTS/G", f"{float(s['PTS/Game']):.3f}")
                    c2.metric("AST/G", f"{float(s['AST/Game']):.3f}")
                    c3.metric("REB/G", f"{float(s['REB/Game']):.3f}")
                    c4.metric("BLK/G", f"{float(s['BLK/Game']):.3f}")
                    c5.metric("eFG%", f"{float(s['eFG%']):.3f}%" if s['eFG%'] is not None else "—")
            else:
                st.caption("Tip: click a circle to pin a stat card; hover for details.")
        else:
            st.plotly_chart(fig, use_container_width=True, key=key_base+"_static")
            st.caption("Interactive charts are off (faster).")

    # -------------------------------- Comparison --------------------------------
    st.markdown("---")
    with st.expander("Comparison", expanded=False):
        if CART_KEY not in st.session_state:
            st.session_state[CART_KEY] = []

        if edit_mode and st.button("Add checked to compare"):
            checked = edited[edited["Compare?"] == True] if (edited is not None and "Compare?" in edited.columns) else pd.DataFrame()
            if checked.empty:
                st.info("No rows were checked.")
            else:
                college_val = str(filt["college"].iloc[0]) if "college" in filt.columns and not filt.empty else ""
                cart = st.session_state[CART_KEY]
                pname_col = "Player Name" if "Player Name" in checked.columns else "player_ind"
                for _, row in checked.iterrows():
                    name = str(row.get(pname_col, "")).strip()
                    if not name: continue
                    item = {"player_ind": name, "season": str(season), "college": college_val}
                    if len(cart) >= 3: break
                    if not any((c.get("player_ind",""), c.get("season",""), c.get("college","")) == (item["player_ind"], item["season"], item["college"]) for c in cart):
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

            with timed("compare rows fetch"):
                rows = rows_for_compare(processed_path, cart)

            if rows:
                def _num(v): 
                    try: return float(v) if v is not None and not pd.isna(v) else 0.0
                    except Exception: return 0.0

                for r in rows:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("PTS/G", f"{_num(r.get('pts_per_game')):.1f}")
                    c2.metric("REB/G", f"{_num(r.get('reb_per_game')):.1f}")
                    c3.metric("AST/G", f"{_num(r.get('ast_per_game')):.1f}")
                    c4.metric("STL/G", f"{_num(r.get('stl_per_game')):.1f}")
                    c5.metric("BLK/G", f"{_num(r.get('blk_per_game')):.1f}")
                    st.caption(f"{r.get('player_ind','')} — {r.get('college','')} ({r.get('season','')})")

                COMPARE_METRICS = [
                    ("pts_per_game","PTS/G"),("reb_per_game","REB/G"),("ast_per_game","AST/G"),
                    ("stl_per_game","STL/G"),("blk_per_game","BLK/G"),("eFG_pct","eFG%"),("USG_pct","USG%"),
                ]
                cats = [n for _, n in COMPARE_METRICS]
                fig = go.Figure()
                max_by_metric = []
                for k, _ in COMPARE_METRICS:
                    vals = []
                    for rr in rows:
                        v = _num(rr.get(k))
                        if k == "eFG_pct" or k.endswith("_pct"):
                            if v <= 1.0: v *= 100.0
                        vals.append(v)
                    max_by_metric.append(max(vals) if vals else 1.0)
                for rr in rows:
                    vals = []
                    for (k, _), vmax in zip(COMPARE_METRICS, max_by_metric):
                        v = _num(rr.get(k))
                        if k == "eFG_pct" or k.endswith("_pct"):
                            if v <= 1.0: v *= 100.0
                        vals.append(0.0 if vmax == 0 else v / vmax)
                    vals.append(vals[0])
                    label = f"{rr.get('player_ind','')} — {rr.get('college','')} ({rr.get('season','')})"
                    fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill="toself", name=label))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                    showlegend=True, height=520, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"cmp_radar_{selected_norm}_{season}")
                st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

                show_cols = [
                    "player_ind","college","season","position","Archetype",
                    "pts_per_game","reb_per_game","ast_per_game","stl_per_game","blk_per_game","eFG_pct","USG_pct"
                ]
                table = pd.DataFrame([{c: rr.get(c) for c in show_cols} for rr in rows])
                if "eFG_pct" in table.columns:
                    table["eFG_pct"] = table["eFG_pct"].apply(lambda v: (_num(v)*100.0 if _num(v) <= 1 else _num(v)))
                if "USG_pct" in table.columns:
                    table["USG_pct"] = table["USG_pct"].apply(_num)
                pretty_table = rename_columns(table.copy())
                st.dataframe(pretty_table, use_container_width=True)
            else:
                st.info("No matching rows found for items in the compare cart.")
        else:
            st.caption("Tip: toggle **Select players to compare**, tick up to 3 players, then click **Add checked to compare**.")

    # -------------------------------- Notes --------------------------------
    st.markdown("---")
    with st.expander("Notes & Comments", expanded=False):
        notes_file = Path("app_notes.json")
        notes = load_json_file(notes_file) if notes_file.exists() else {}
        key = f"{team_display}__{season}"
        existing_text = notes.get(key, "")
        txt = st.text_area("Write notes for this team/season", value=existing_text, height=160, key="notes_roster")
        if st.button("Save notes", key="save_notes_roster"):
            notes[key] = txt
            notes_file.write_text(json.dumps(notes, indent=2))
            st.success("Notes saved")
