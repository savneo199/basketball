# helpers/court_builder.py
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

LINE_WHITE = "rgba(255,255,255,0.9)"
LINE_WHITE_SOFT = "rgba(255,255,255,0.35)"

def _resolve_theme():
    try:
        import streamlit as st
        base = st.get_option("theme.base") or "light"
        txt  = st.get_option("theme.textColor") or ("#FAFAFA" if base == "dark" else "#31333F")
        prim = st.get_option("theme.primaryColor") or ("#79B8FF" if base == "dark" else "#2E7CF6")
    except Exception:
        base, txt, prim = "light", "#31333F", "#2E7CF6"
    ann_bg = "rgba(0,0,0,0.55)" if base == "light" else "rgba(255,255,255,0.18)"
    ann_txt = "#FFFFFF" if base == "light" else txt
    return {"base": base, "text": txt, "primary": prim, "ann_bg": ann_bg, "ann_txt": ann_txt}

def default_spots(width=50, length=47):
    cx = width / 2.0
    return {
        "PG": (cx, length * 0.76),
        "SG": (width * 0.16, length * 0.60),
        "SF": (width * 0.84, length * 0.56),
        "PF": (width * 0.18, length * 0.28),
        "C":  (width * 0.82, length * 0.28),
    }

# --- simple, reliable half-court shapes ---
def court_shapes(width=50, length=47, line_color=LINE_WHITE):
    cx, hoop_y = width / 2.0, 5.25
    lane_w, lane_h = 16.0, 19.0
    ft_circle_r, restricted_r = 6.0, 4.0
    backboard_y = 4.0
    three_r = 23.75

    def L(w=2): return dict(width=w, color=line_color)

    shapes = [
        dict(type="rect", x0=0, y0=0, x1=width, y1=length, line=L(2), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=cx - lane_w/2, y0=0, x1=cx + lane_w/2, y1=lane_h, line=L(2), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=cx-0.75, y0=hoop_y-0.75, x1=cx+0.75, y1=hoop_y+0.75, line=L(2), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=cx-restricted_r, y0=hoop_y-restricted_r, x1=cx+restricted_r, y1=hoop_y+restricted_r, line=L(1), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=cx-ft_circle_r, y0=lane_h-ft_circle_r, x1=cx+ft_circle_r, y1=lane_h+ft_circle_r, line=L(1), fillcolor="rgba(0,0,0,0)"),
    ]

    backboard = go.Scatter(x=[cx - 3, cx + 3], y=[backboard_y, backboard_y], mode="lines",
                           line=L(3), hoverinfo="skip", showlegend=False)

    theta = np.linspace(np.deg2rad(30), np.deg2rad(150), 140)
    x_arc = cx + three_r * np.cos(theta)
    y_arc = hoop_y + three_r * np.sin(theta)
    three_arc = go.Scatter(x=x_arc, y=y_arc, mode="lines",
                           line=L(2), hoverinfo="skip", showlegend=False)

    return shapes, [backboard, three_arc]

# ---- cache geometry/spots once per session ----
def _cached_court(width=50, length=47):
    try:
        import streamlit as st
        @st.cache_resource(show_spinner=False)
        def _do(width: float, length: float):
            return court_shapes(width=width, length=length, line_color=LINE_WHITE), default_spots(width=width, length=length)
        return _do(width, length)
    except Exception:
        return (court_shapes(width=width, length=length, line_color=LINE_WHITE), default_spots(width=width, length=length))

def make_lineup_figure(
    labels: list[str],
    title: str = "Possible lineup",
    width: float = 50,
    length: float = 47,
    marker_size: int = 28,
    slots_order: list[str] | None = None,
    numbers: list[str] | None = None,
    stats: list[dict] | None = None,
):
    theme = _resolve_theme()
    (shapes, extra_traces), spots = _cached_court(width=width, length=length)
    default_order = ["PG", "SG", "SF", "PF", "C"]
    order = slots_order if (slots_order and len(slots_order) == len(labels)) else default_order[:len(labels)]
    coords = [spots.get(pos, spots["SF"]) for pos in order]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # hover text
    hovertext = []
    for i, lab in enumerate(labels):
        if stats and i < len(stats):
            s = stats[i] or {}
            jersey = f" #{numbers[i]}" if numbers and i < len(numbers) and numbers[i] else ""
            ef = s.get("eFG%")
            ef_str = f"{float(ef):.3f}%" if ef is not None else "—"
            hovertext.append(
                f"<b>{lab}{jersey}</b><br>"
                f"PTS: {float(s.get('PTS/Game',0)):.3f} | AST: {float(s.get('AST/Game',0)):.3f} | "
                f"REB: {float(s.get('REB/Game',0)):.3f} | BLK: {float(s.get('BLK/Game',0)):.3f}<br>"
                f"eFG%: {ef_str}"
            )
        else:
            jersey = f" #{numbers[i]}" if numbers and i < len(numbers) and numbers[i] else ""
            hovertext.append(f"<b>{lab}{jersey}</b>")

    fig = go.Figure()
    fig.update_layout(shapes=shapes)
    for t in extra_traces:
        if hasattr(t, "line") and t.line is not None:
            t.line.color = LINE_WHITE
        fig.add_trace(t)

    if labels:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=marker_size + 12, opacity=0.15, color=theme["primary"]),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            hovertext=hovertext, hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(size=marker_size, color=theme["primary"], line=dict(width=2, color=LINE_WHITE)),
            showlegend=False, name="player",
        ))
        if numbers:
            nums = [numbers[i] if i < len(numbers) and numbers[i] is not None else "" for i in range(len(labels))]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="text",
                text=nums, textfont=dict(size=13, color="white"),
                hoverinfo="skip", showlegend=False,
            ))
        yshift = int(marker_size * 0.9)
        for x, y, text in zip(xs, ys, labels):
            fig.add_annotation(
                x=x, y=y, text=text, showarrow=False,
                font=dict(size=13, color=theme["ann_txt"]),
                xanchor="center", yanchor="bottom", yshift=yshift, align="center",
                bgcolor=theme["ann_bg"], bordercolor=LINE_WHITE_SOFT, borderwidth=1, borderpad=6,
            )

    fig.update_xaxes(range=[0, width], visible=False)
    fig.update_yaxes(range=[0, length], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=20, r=20, t=36, b=20),
        title=title,
        title_font=dict(size=20, color=theme["text"]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=560,
    )
    return fig
