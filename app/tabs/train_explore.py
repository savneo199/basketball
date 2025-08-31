import plotly.express as px
import pandas as pd
import streamlit as st

from helpers.helpers import latest_artifacts, load_json_file
from run_pipeline import run_pipeline

def render():
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Run pipeline now"):
            rid = run_pipeline()
            if rid:
                st.success(f"Artifacts updated: {rid}")
    with colB:
        if st.button("Refresh artifacts"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Refreshed")


    
        
    st.subheader("Cluster Composition by Archetype")
    paths = latest_artifacts()
    if not paths:
        st.info("No artifacts yet. Run the pipeline.")
        return

    summary = load_json_file(paths["summary"]) if paths["summary"].exists() else {}
    cluster_sizes = (summary.get("cluster_sizes") or {})
    if not cluster_sizes:
        st.info("No 'cluster_sizes' found in cluster_summary.json.")
        return

    items = sorted(((str(name), int(v)) for name, v in cluster_sizes.items()),
                   key=lambda x: (-x[1], x[0]))
    labels = [name for name, _ in items]
    counts = [cnt for _, cnt in items]
    total = max(sum(counts), 1)
    df_pie = pd.DataFrame({
        "Archetype": labels,
        "Count": counts,
        "Percent": [c * 100.0 / total for c in counts],
    })
    fig = px.pie(df_pie, names="Archetype", values="Count", hole=0.35)
    fig.update_traces(
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>% of total: %{percent}<extra></extra>"
    )
    fig.update_layout(
        legend_title_text="Archetypes",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Counts by archetype")
    st.dataframe(
        df_pie[["Archetype", "Count", "Percent"]].sort_values("Count", ascending=False),
        use_container_width=True
    )
