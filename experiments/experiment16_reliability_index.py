from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
OUTPUT_CSV = EXPERIMENT_DIR / "reliability_index.csv"
OUTPUT_FIGURE = EXPERIMENT_DIR / "reliability_index.png"
ROOT_FIGURE = ROOT_DIR / "figures" / "reliability_index.png"
PAPER_FIGURE = ROOT_DIR / "paper" / "figures" / "reliability_index.png"

MODEL_STATISTICS = EXPERIMENT_DIR / "model_statistics.csv"
FAILURE_MATRIX = EXPERIMENT_DIR / "failure_matrix.csv"
CALIBRATION_METRICS = EXPERIMENT_DIR / "calibration_metrics.csv"
SHIFT_STATISTICS = EXPERIMENT_DIR / "shift_statistics.csv"

WEIGHTS = {
    "Accuracy Score": 0.25,
    "Noise Robustness Score": 0.20,
    "Missing Data Score": 0.15,
    "Calibration Score": 0.15,
    "Distribution Shift Score": 0.15,
    "Confidence Score": 0.10,
}

COLORS = ["#176B5B", "#287E9B", "#D79A28", "#B84A5F"]


def require_columns(frame, columns, source):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} is missing columns: {sorted(missing)}")


def load_components():
    model_stats = pd.read_csv(MODEL_STATISTICS)
    failure_matrix = pd.read_csv(FAILURE_MATRIX)
    calibration = pd.read_csv(CALIBRATION_METRICS)
    shift = pd.read_csv(SHIFT_STATISTICS)

    require_columns(
        model_stats,
        ["Model", "Mean Acc", "Noise Mean Acc"],
        MODEL_STATISTICS,
    )
    require_columns(failure_matrix, ["Model", "Missing"], FAILURE_MATRIX)
    require_columns(
        calibration,
        ["Model", "Expected Calibration Error"],
        CALIBRATION_METRICS,
    )
    require_columns(
        shift,
        [
            "Model",
            "Mean Shift",
            "Accuracy Mean",
            "Confidence-Accuracy Gap",
        ],
        SHIFT_STATISTICS,
    )

    accuracy = model_stats.set_index("Model")["Mean Acc"] * 100
    noise = model_stats.set_index("Model")["Noise Mean Acc"] * 100
    missing = failure_matrix.set_index("Model")["Missing"]

    mean_ece = calibration.groupby("Model")["Expected Calibration Error"].mean()
    calibration_score = (1 - mean_ece).clip(lower=0, upper=1) * 100

    endpoint = shift[shift["Mean Shift"] == shift["Mean Shift"].max()]
    distribution_shift = endpoint.set_index("Model")["Accuracy Mean"] * 100

    mean_confidence_gap = shift.groupby("Model")["Confidence-Accuracy Gap"].mean()
    confidence_score = (1 - mean_confidence_gap).clip(lower=0, upper=1) * 100

    components = pd.concat(
        {
            "Accuracy Score": accuracy,
            "Noise Robustness Score": noise,
            "Missing Data Score": missing,
            "Calibration Score": calibration_score,
            "Distribution Shift Score": distribution_shift,
            "Confidence Score": confidence_score,
        },
        axis=1,
    ).dropna()

    expected_models = set(model_stats["Model"])
    if set(components.index) != expected_models:
        absent = expected_models - set(components.index)
        raise ValueError(f"Missing component data for models: {sorted(absent)}")

    return components


def calculate_index(components):
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError("Reliability Index weights must sum to 1.0")
    if ((components < 0) | (components > 100)).any().any():
        raise ValueError("Reliability Index components must be between 0 and 100")

    weighted = sum(
        components[column] * weight for column, weight in WEIGHTS.items()
    )
    results = components.copy()
    results.insert(0, "Reliability Index", weighted)
    results = results.sort_values("Reliability Index", ascending=False).reset_index()
    return results.rename(columns={"index": "Model"})


def draw_figure(results):
    fig, (rank_ax, component_ax) = plt.subplots(
        1,
        2,
        figsize=(12, 5.4),
        gridspec_kw={"width_ratios": [0.9, 1.55]},
    )

    ranked = results.iloc[::-1]
    bars = rank_ax.barh(
        ranked["Model"],
        ranked["Reliability Index"],
        color=COLORS[::-1],
        height=0.58,
    )
    rank_ax.bar_label(
        bars,
        labels=[f"{value:.2f}" for value in ranked["Reliability Index"]],
        padding=6,
        fontsize=10,
        fontweight="bold",
    )
    rank_ax.set_xlim(0, 100)
    rank_ax.set_xlabel("Reliability Index (0-100)")
    rank_ax.set_title("Overall Ranking", fontweight="bold")
    rank_ax.grid(axis="x", alpha=0.2)
    rank_ax.spines[["top", "right", "left"]].set_visible(False)
    rank_ax.tick_params(axis="y", length=0)

    component_columns = list(WEIGHTS)
    component_matrix = results.set_index("Model")[component_columns]
    component_matrix.columns = [
        "Accuracy",
        "Noise",
        "Missing",
        "Calibration",
        "Shift",
        "Confidence",
    ]
    sns.heatmap(
        component_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=75,
        vmax=100,
        linewidths=0.6,
        cbar_kws={"label": "Component Score"},
        ax=component_ax,
    )
    component_ax.set_title("Component Scores", fontweight="bold")
    component_ax.set_xlabel("")
    component_ax.set_ylabel("")
    component_ax.tick_params(axis="x", rotation=35)
    component_ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Experiment 16: Model Reliability Index", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.92,
        "Cross-experiment synthesis of performance, robustness, calibration, shift resistance, and confidence",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    for path in [OUTPUT_FIGURE, ROOT_FIGURE, PAPER_FIGURE]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    components = load_components()
    results = calculate_index(components)
    results.round(4).to_csv(OUTPUT_CSV, index=False)
    draw_figure(results)

    print("\nModel Reliability Index (0-100):\n")
    print(results.round(2).to_string(index=False))
    print(f"\nSaved index to {OUTPUT_CSV}")
    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
