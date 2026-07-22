import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .loader import figure_path
from .theme import get_theme


MODEL_COLORS = ["primary", "secondary", "success", "warning", "danger"]


def apply_chart_theme(ax, title=None):
    """Apply active dashboard theme tokens to a Matplotlib axis."""
    theme = get_theme()
    ax.figure.patch.set_alpha(0)
    ax.set_facecolor("none")
    if title:
        ax.set_title(title, color=theme["text"], fontweight="bold", pad=14)
    ax.tick_params(colors=theme["muted_text"])
    ax.xaxis.label.set_color(theme["muted_text"])
    ax.yaxis.label.set_color(theme["muted_text"])
    ax.grid(color=theme["grid"], alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color(theme["border"])
    return theme


def show_figure(name, caption=None):
    path = figure_path(name)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found: figures/{name}")


def probability_chart(classes, probabilities):
    """Render a themed probability bar chart."""
    frame = pd.DataFrame({"Class": classes, "Probability": probabilities})
    bar_chart(frame, "Class", "Probability", "Class Probabilities")


def bar_chart(frame, x, y, title=None):
    """Render a theme-aware Matplotlib bar chart."""
    theme = get_theme()
    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    ax.bar(frame[x].astype(str), frame[y], color=theme["primary"], edgecolor=theme["border"])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    apply_chart_theme(ax, title)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def meter(label, value):
    safe_value = max(0.0, min(float(value), 1.0))
    st.write(f"{label}: {safe_value:.1%}")
    st.progress(safe_value)


def radar_chart(index):
    if index.empty:
        st.info("Component scores are not available.")
        return
    component_columns = [
        "Accuracy Score",
        "Noise Robustness Score",
        "Missing Data Score",
        "Calibration Score",
        "Distribution Shift Score",
        "Confidence Score",
    ]
    available = [column for column in component_columns if column in index.columns]
    if not available:
        st.info("Radar chart requires component score columns.")
        return

    labels = [column.replace(" Score", "").replace("Distribution Shift", "Shift") for column in available]
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    theme = get_theme()
    fig, ax = plt.subplots(figsize=(6.4, 5.8), subplot_kw={"polar": True})
    colors = [theme[name] for name in MODEL_COLORS if name in theme]
    for idx, (_, row) in enumerate(index.iterrows()):
        values = [float(row[column]) for column in available]
        values += values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=1.8, label=row["Model"], color=color)
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, color=theme["muted_text"])
    ax.set_ylim(75, 100)
    ax.set_title("Reliability Component Radar", fontweight="bold", pad=18, color=theme["text"])
    ax.grid(color=theme["grid"], alpha=0.32)
    ax.tick_params(colors=theme["muted_text"])
    ax.figure.patch.set_alpha(0)
    ax.set_facecolor("none")
    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    legend.get_frame().set_facecolor(theme["card"])
    legend.get_frame().set_edgecolor(theme["border"])
    for text in legend.get_texts():
        text.set_color(theme["text"])
    st.pyplot(fig, clear_figure=True)


def correlation_heatmap(frame, title="Correlation Matrix"):
    if frame.empty:
        st.info("No dataset loaded.")
        return
    corr = frame.corr(numeric_only=True)
    theme = get_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title, fontweight="bold", pad=12, color=theme["text"])
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7, color=theme["muted_text"])
    ax.set_yticklabels(corr.columns, fontsize=7, color=theme["muted_text"])
    ax.figure.patch.set_alpha(0)
    ax.set_facecolor("none")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=theme["muted_text"])
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
