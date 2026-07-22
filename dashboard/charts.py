import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .loader import figure_path


def show_figure(name, caption=None):
    path = figure_path(name)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found: figures/{name}")


def probability_chart(classes, probabilities):
    frame = pd.DataFrame({"Class": classes, "Probability": probabilities})
    st.bar_chart(frame, x="Class", y="Probability")


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

    fig, ax = plt.subplots(figsize=(6.4, 5.8), subplot_kw={"polar": True})
    for _, row in index.iterrows():
        values = [float(row[column]) for column in available]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=row["Model"])
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(75, 100)
    ax.set_title("Reliability Component Radar", fontweight="bold", pad=18)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    st.pyplot(fig, clear_figure=True)


def correlation_heatmap(frame, title="Correlation Matrix"):
    if frame.empty:
        st.info("No dataset loaded.")
        return
    corr = frame.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
