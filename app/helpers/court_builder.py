from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import pandas as pd

LINE_WHITE = "rgba(255,255,255,0.9)"
LINE_WHITE_SOFT = "rgba(255,255,255,0.35)"

def resolve_theme():
    """Light/dark-aware colors for text/primary; court lines are forced white."""
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

def build_lineup_labels(
    df_top5: pd.DataFrame,
    name_col="player_ind",
    archetype_col: str = "Archetype",
) -> list[str]:
    labels = []
    for _, row in df_top5.iterrows():
        name = str(row.get(name_col, "")).strip()
        arch = str(row.get(archetype_col, "") or "").strip()
        labels.append(f"{name} — {arch}" if arch else name)
    return labels

def make_lineup_figure(
    labels: list[str],
    title: str,
    width: float = 50,
    length: float = 47,
    marker_size: int = 28,
    slots_order: list[str] | None = None,  
    numbers: list[str] | None = None,
    stats: list[dict] | None = None,
):
    """
    Half-court with five labeled spots.
    - Transparent background to blend with Streamlit.
    - Labels appear above the circles.
    - Court lines are white.
    - If slots_order is provided, places labels in those PG/SG/SF/PF/C slots (same length as labels).
    - If numbers provided, renders them inside the circles.
    - If stats provided, they appear in hover tooltips.
    """
    theme = resolve_theme()
    shapes = court_shapes(width=width, length=length, line_color=LINE_WHITE)
    spots = default_spots(width=width, length=length)

    default_order = ["PG", "SG", "SF", "PF", "C"]
    order = slots_order if (slots_order and len(slots_order) == len(labels)) else default_order[:len(labels)]
    coords = [spots.get(pos, spots["SF"]) for pos in order]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Build hover text from labels + stats
    hovertext = []
    for i, lab in enumerate(labels):
        if stats and i < len(stats) and isinstance(stats[i], dict):
            s = stats[i]
            fgm = s.get("FGM", 0); fga = s.get("FGA", 0); fgp = s.get("FG%")
            fg_line = f"{fgm}-{fga}"
            if fgp is not None:
                fg_line += f" ({fgp:.1f}%)"
            jersey = f" #{numbers[i]}" if numbers and i < len(numbers) and numbers[i] else ""
            hovertext.append(
                f"<b>{lab}{jersey}</b><br>"
                f"PTS: {float(s.get('PTS/Game',0)):.3f} <br>"
                f"AST: {float(s.get('AST/Game',0)):.3f} <br>"
                f"REB: {float(s.get('REB/Game',0)):.3f} <br>"
                f"BLK: {float(s.get('BLK/Game',0)):.3f} <br>"
                f"eFG%: {float(s.get('eFG%',0)):.3f}%"
            )
        else:
            jersey = f" #{numbers[i]}" if numbers and i < len(numbers) and numbers[i] else ""
            hovertext.append(f"<b>{lab}{jersey}</b>")

    fig = go.Figure()
    fig.update_layout(shapes=shapes)

    if labels:
        # Soft glow
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=marker_size + 12, opacity=0.15, color=theme["primary"]),
                hoverinfo="skip", showlegend=False,
            )
        )
        # Solid circle with white border
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="markers",
                hovertext=hovertext,
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(size=marker_size, color=theme["primary"], line=dict(width=2, color=LINE_WHITE)),
                showlegend=False,
                name="player",
            )
        )
        # Numbers inside circles 
        if numbers:
            nums = [numbers[i] if i < len(numbers) and numbers[i] is not None else "" for i in range(len(labels))]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="text",
                    text=nums,
                    textfont=dict(size=13, color="white"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Labels above the circles
        yshift = int(marker_size * 0.9)
        for x, y, text in zip(xs, ys, labels):
            fig.add_annotation(
                x=x, y=y,
                text=text,
                showarrow=False,
                font=dict(size=13, color=theme["ann_txt"]),
                xanchor="center",
                yanchor="bottom",
                yshift=yshift,
                align="center",
                bgcolor=theme["ann_bg"],
                bordercolor=LINE_WHITE_SOFT,
                borderwidth=1,
                borderpad=6,
            )
    fig.update_xaxes(range=[0, width], visible=False, constrain="domain", fixedrange=True)
    fig.update_yaxes(range=[0, length], visible=False, scaleanchor="x", scaleratio=1, constrain="domain", fixedrange=True)
    fig.update_layout(
        margin=dict(l=20, r=20, t=36, b=20),
        autosize=True,
        title=title,
        title_font=dict(size=20, color=theme["text"]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=560,
    )
    return fig


def default_spots(width=50, length=47):
    cx = width / 2.0
    return {
        "PG": (cx, length * 0.76),
        "SG": (width * 0.16, length * 0.60),
        "SF": (width * 0.84, length * 0.56),
        "PF": (width * 0.18, length * 0.28),
        "C": (width * 0.82, length * 0.28),
    }

# internal shapes
# --- replace your court_shapes() with this ---
def court_shapes(width=50, length=47, line_color=LINE_WHITE):
    cx, hoop_y = width / 2.0, 5.25
    lane_w, lane_h = 16.0, 19.0
    ft_circle_r, restricted_r = 6.0, 4.0
    backboard_y = 4.0
    three_r = 23.75           # NBA/WNBA 3PT radius (to hoop center)
    corner_x_off = 22.0       # horizontal distance for corner 3
    # angle where arc meets the corner lines
    cos_tb = min(corner_x_off / three_r, 1.0)
    sin_tb = (1.0 - cos_tb**2) ** 0.5
    theta_break = np.arccos(cos_tb)
    y_break = hoop_y + three_r * sin_tb
    x_left_break  = cx - three_r * cos_tb  # = cx - 22.0
    x_right_break = cx + three_r * cos_tb  # = cx + 22.0

    def L(w=2): return dict(width=w, color=line_color)

    # base court (all set below so players render on top)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=width, y1=length, line=L(2),
             fillcolor="rgba(0,0,0,0)", layer="below"),
        dict(type="rect", x0=cx - lane_w/2, y0=0, x1=cx + lane_w/2, y1=lane_h, line=L(2),
             fillcolor="rgba(0,0,0,0)", layer="below"),
        dict(type="circle", x0=cx-0.75, y0=hoop_y-0.75, x1=cx+0.75, y1=hoop_y+0.75, line=L(2),
             fillcolor="rgba(0,0,0,0)", layer="below"),
        dict(type="circle", x0=cx-restricted_r, y0=hoop_y-restricted_r, x1=cx+restricted_r, y1=hoop_y+restricted_r, line=L(1),
             fillcolor="rgba(0,0,0,0)", layer="below"),
        dict(type="circle", x0=cx-ft_circle_r, y0=lane_h-ft_circle_r, x1=cx+ft_circle_r, y1=lane_h+ft_circle_r, line=L(1),
             fillcolor="rgba(0,0,0,0)", layer="below"),
        # backboard
        dict(type="line", x0=cx-3, y0=backboard_y, x1=cx+3, y1=backboard_y, line=L(3), layer="below"),
    ]

    # 3PT arc as a PATH shape
    theta = np.linspace(theta_break, np.pi - theta_break, 180)
    x_arc = cx + three_r * np.cos(theta)
    y_arc = hoop_y + three_r * np.sin(theta)
    arc_path = "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in zip(x_arc, y_arc))
    shapes.append(dict(type="path", path=arc_path, line=L(3), layer="below"))

    # corner threes (straight vertical segments)
    shapes.append(dict(type="line", x0=cx - corner_x_off, y0=0, x1=cx - corner_x_off, y1=y_break,
                       line=L(3), layer="below"))
    shapes.append(dict(type="line", x0=cx + corner_x_off, y0=0, x1=cx + corner_x_off, y1=y_break,
                       line=L(3), layer="below"))


    return shapes
