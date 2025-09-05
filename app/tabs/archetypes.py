import json
from pathlib import Path
import streamlit as st
import pandas as pd

from helpers.helpers import rename_columns, BASE_MAP
def render():
    # Load archetypes
    ARCH_JSON_PATH = Path("data/archetypes.json")
    ARCHETYPE_RULES: dict[str, list[list]] = json.loads(ARCH_JSON_PATH.read_text())
    ALL_ARCHETYPES = sorted(ARCHETYPE_RULES.keys())

    # Rename metrics using BASE_MAP
    def pretty_metric_labels(metrics: list[str]) -> list[str]:
        dummy = pd.DataFrame(columns=metrics)
        renamed = rename_columns(dummy)
        return list(renamed.columns)

    def metrics_to_short_sentence(metrics: list[str]) -> str:
        """
        Build a brief, coach-friendly description from the metric names.
        """
        phrases = [BASE_MAP.get(m, m) for m in metrics]
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0]
        return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


    st.title("Archetype Dictionary")

    st.caption(
        "Pick an archetype to see its description, its key indicators and how the model uses them."
    )

    col1, col2 = st.columns([1, 2], vertical_alignment="center")
    with col1:
        archetype = st.selectbox("Archetype", ALL_ARCHETYPES, index=0 if ALL_ARCHETYPES else None)
    with col2:
        st.write("")  # spacing

    if not archetype:
        st.info("Add your archetypes JSON to `data/archetypes.json` to get started.")
        st.stop()

    pairs = ARCHETYPE_RULES.get(archetype, [])
    metrics = [m for m, _w in pairs]
    weights = {m: w for m, w in pairs}

    # Pretty labels
    pretty = pretty_metric_labels(metrics)
    pretty_map = {m: p for m, p in zip(metrics, pretty)}

    # Narrative description
    offense_cues = {"eFG_pct","TS_pct","3pt_3pt_pct_ind","threeA_rate","threeA_per40","three_per100","2pt_pct",
                    "PPP","USG_pct","pts_per40","AST_pct","AST_per_TO","TOV_pct","FTr","Spacing","Gravity",
                    "PPT","BoxCreation","Assist_to_Usage"}
    defense_cues = {"stl_per40","stl_per100","STL_pct","blk_per40","def_stops_per100","DPMR","DRB_pct","ORB_pct"}

    has_off = any(m in offense_cues for m in metrics)
    has_def = any(m in defense_cues for m in metrics)

    focus = []
    if has_off: focus.append("offensive impact")
    if has_def: focus.append("defensive impact")
    focus_txt = " and ".join(focus) if focus else "impact"

    core_txt = metrics_to_short_sentence(metrics)

    st.subheader(archetype)
    st.write(
        f"This archetype emphasizes **{focus_txt}** via {core_txt}. "
        "Indicators and weights below reflect how this profile is identified in the archetyping model."
    )

    # Key indicators table (pretty names + weights)
    disp = pd.DataFrame({
        "Indicator": [pretty_map[m] for m in metrics],
        "Weight": [weights[m] for m in metrics],
        "Raw Column": metrics
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

    with st.expander("How to interpret the indicators"):
        st.markdown(
            "- **Shooting efficiency** (eFG%/TS%) and **PPP** → overall scoring efficiency.\n"
            "- **Three-point accuracy/volume & spacing/gravity** → perimeter threat that widens the floor.\n"
            "- **Usage (USG%) & points per 40** → on-ball shot creation / volume.\n"
            "- **AST% / AST:TO / TOV%** → playmaking quality and ball security.\n"
            "- **Rebounding (ORB%/DRB%) & defensive stops/blocks/steals** → possession control and rim/perimeter defense.\n"
            "- **FTr** → rim pressure and foul-drawing.\n"
            "Weights (+1/−1) show whether higher values **support** (+) or **contradict** (−) the archetype definition."
        )

    st.markdown("---")
    st.subheader("Reference List")

    st.markdown(
        """
    - Kubatko, J., Oliver, D., Pelton, K., & Rosenbaum, D. T. (2007). *A Starting Point for Analyzing Basketball Statistics*. **Journal of Quantitative Analysis in Sports**. (possessions, efficiency, eFG%, TS%, usage).  
    - García, J., Ibáñez, S. J., Martínez de Santos, R., Leite, N., & Sampaio, J. (2013). *Identifying Basketball Performance Indicators in Regular Season and Playoff Games*. **Journal of Human Kinetics**. (offensive/defensive indicators: assists, turnovers, rebounds, steals, blocks).  
    - Mikołajec, K., Maszczyk, A., & Zając, T. (2013). *Game Indicators Determining Sports Performance in the NBA*. **International Journal of Performance Analysis in Sport**. (key indicators associated with performance).  
    - Sun, W., et al. (2022). *Evaluation of differences in the performance strategies of winning and losing female basketball teams*. **Frontiers in Sports and Active Living**. (attack vs. defense indicators including 3P, FT, rebounds, steals, blocks).  
    - Yang, L., et al. (2024). *Noisy condition and three-point shot performance in skilled basketball players*. **Frontiers in Psychology**. (importance and context of 3-point shooting).
    """
    )

    st.caption(
        "These sources justify the indicator families (efficiency, usage, creation, spacing, rebounds, defense). "
        "Specific custom metrics (e.g., DPMR, BoxCreation, Gravity) are model constructs; we display them alongside "
        "peer-reviewed indicator families they relate to."
    )
