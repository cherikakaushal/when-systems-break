from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
METRICS_CSV = EXPERIMENT_DIR / "calibration_metrics.csv"
BINS_CSV = EXPERIMENT_DIR / "calibration_bins.csv"
CURVE_FIGURE = EXPERIMENT_DIR / "calibration_curve.png"
DIAGRAM_FIGURE = EXPERIMENT_DIR / "reliability_diagram.png"

N_RUNS = 30
N_BINS = 10
NOISE_LEVEL = 0.35
CONDITIONS = ("Clean", "Noise")
COLORS = {
    "Logistic Regression": "#176B5B",
    "SVM": "#287E9B",
    "Random Forest": "#D79A28",
    "Decision Tree": "#B84A5F",
}


def build_models(seed):
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=seed),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=seed),
        ),
    }


def add_noise(X, rng):
    feature_scale = X.std(axis=0).replace(0, 1)
    noise = rng.normal(0, NOISE_LEVEL, X.shape) * feature_scale.to_numpy()
    return X + noise


def prediction_confidence(model, X, y):
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    class_indices = {label: index for index, label in enumerate(model.classes_)}
    predicted_indices = np.array([class_indices[label] for label in predictions])
    confidence = probabilities[np.arange(len(predictions)), predicted_indices]
    correct = (predictions == np.asarray(y)).astype(int)
    return confidence, correct


def bin_calibration(confidence, correct):
    edges = np.linspace(0, 1, N_BINS + 1)
    bin_ids = np.digitize(confidence, edges[1:-1], right=False)
    rows = []

    for bin_id in range(N_BINS):
        selected = bin_ids == bin_id
        count = int(selected.sum())
        if count == 0:
            continue

        mean_confidence = float(confidence[selected].mean())
        observed_accuracy = float(correct[selected].mean())
        rows.append(
            {
                "Bin": bin_id + 1,
                "Lower Bound": edges[bin_id],
                "Upper Bound": edges[bin_id + 1],
                "Count": count,
                "Mean Confidence": mean_confidence,
                "Observed Accuracy": observed_accuracy,
                "Calibration Gap": abs(mean_confidence - observed_accuracy),
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(bins):
    total = bins["Count"].sum()
    return float((bins["Count"] / total * bins["Calibration Gap"]).sum())


def collect_predictions():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    pooled = {
        model_name: {
            condition: {"confidence": [], "correct": [], "run_ece": []}
            for condition in CONDITIONS
        }
        for model_name in build_models(0)
    }

    for seed in range(N_RUNS):
        rng = np.random.default_rng(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        test_sets = {"Clean": X_test, "Noise": add_noise(X_test, rng)}

        for model_name, model in build_models(seed).items():
            model.fit(X_train, y_train)

            for condition, condition_X in test_sets.items():
                confidence, correct = prediction_confidence(
                    model, condition_X, y_test
                )
                run_bins = bin_calibration(confidence, correct)
                pooled[model_name][condition]["confidence"].append(confidence)
                pooled[model_name][condition]["correct"].append(correct)
                pooled[model_name][condition]["run_ece"].append(
                    expected_calibration_error(run_bins)
                )

    return pooled


def summarize(pooled):
    bin_tables = []
    metric_rows = []

    for model_name, conditions in pooled.items():
        for condition, values in conditions.items():
            confidence = np.concatenate(values["confidence"])
            correct = np.concatenate(values["correct"])
            bins = bin_calibration(confidence, correct)
            bins.insert(0, "Condition", condition)
            bins.insert(0, "Model", model_name)
            bin_tables.append(bins)

            high_confidence = bins[bins["Bin"] == N_BINS]
            high_accuracy = (
                float(high_confidence["Observed Accuracy"].iloc[0])
                if not high_confidence.empty
                else np.nan
            )
            high_mean_confidence = (
                float(high_confidence["Mean Confidence"].iloc[0])
                if not high_confidence.empty
                else np.nan
            )
            high_count = (
                int(high_confidence["Count"].iloc[0])
                if not high_confidence.empty
                else 0
            )
            run_ece = np.asarray(values["run_ece"])

            metric_rows.append(
                {
                    "Model": model_name,
                    "Condition": condition,
                    "Expected Calibration Error": expected_calibration_error(bins),
                    "Mean Run ECE": run_ece.mean(),
                    "Run ECE Std": run_ece.std(ddof=1),
                    "Mean Confidence": confidence.mean(),
                    "Observed Accuracy": correct.mean(),
                    "90-100% Bin Mean Confidence": high_mean_confidence,
                    "90-100% Bin Accuracy": high_accuracy,
                    "90-100% Bin Count": high_count,
                    "Samples": len(correct),
                }
            )

    return pd.DataFrame(metric_rows), pd.concat(bin_tables, ignore_index=True)


def draw_calibration_curve(bins):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)

    for ax, condition in zip(axes, CONDITIONS):
        ax.plot([0, 1], [0, 1], "--", color="#555555", label="Perfect calibration")
        for model_name, color in COLORS.items():
            subset = bins[
                (bins["Model"] == model_name)
                & (bins["Condition"] == condition)
            ]
            ax.plot(
                subset["Mean Confidence"],
                subset["Observed Accuracy"],
                marker="o",
                linewidth=2,
                color=color,
                label=model_name,
            )
        ax.set_title(f"{condition} Inputs", fontweight="bold")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Observed Correctness")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Calibration Curves Across Models", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, CURVE_FIGURE, "calibration_curve.png")


def draw_reliability_diagram(metrics, bins):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True, sharey=True)

    for ax, (model_name, color) in zip(axes.flat, COLORS.items()):
        ax.plot([0, 1], [0, 1], "--", color="#666666", linewidth=1.2)
        for condition, marker in [("Clean", "o"), ("Noise", "s")]:
            subset = bins[
                (bins["Model"] == model_name)
                & (bins["Condition"] == condition)
            ]
            ece = metrics.loc[
                (metrics["Model"] == model_name)
                & (metrics["Condition"] == condition),
                "Expected Calibration Error",
            ].iloc[0]
            line_style = "-" if condition == "Clean" else ":"
            ax.plot(
                subset["Mean Confidence"],
                subset["Observed Accuracy"],
                marker=marker,
                linestyle=line_style,
                linewidth=2,
                color=color,
                alpha=1.0 if condition == "Clean" else 0.72,
                label=f"{condition} (ECE {ece:.3f})",
            )
        ax.set_title(model_name, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Mean Predicted Confidence")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed Correctness")

    fig.suptitle(
        "Reliability Diagrams Under Clean and Noisy Inputs",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, DIAGRAM_FIGURE, "reliability_diagram.png")


def save_figure(fig, experiment_path, filename):
    destinations = [
        experiment_path,
        ROOT_DIR / "figures" / filename,
        ROOT_DIR / "paper" / "figures" / filename,
    ]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    pooled = collect_predictions()
    metrics, bins = summarize(pooled)
    metrics = metrics.sort_values(["Condition", "Expected Calibration Error"])
    metrics.round(6).to_csv(METRICS_CSV, index=False)
    bins.round(6).to_csv(BINS_CSV, index=False)
    draw_calibration_curve(bins)
    draw_reliability_diagram(metrics, bins)

    display = metrics.copy()
    percentage_columns = [
        "Expected Calibration Error",
        "Mean Confidence",
        "Observed Accuracy",
        "90-100% Bin Mean Confidence",
        "90-100% Bin Accuracy",
    ]
    display[percentage_columns] = display[percentage_columns].map(
        lambda value: f"{value:.2%}"
    )
    print("\nCalibration results across 30 seeded splits:\n")
    print(
        display[
            [
                "Model",
                "Condition",
                "Expected Calibration Error",
                "Mean Confidence",
                "Observed Accuracy",
                "90-100% Bin Mean Confidence",
                "90-100% Bin Accuracy",
                "90-100% Bin Count",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved metrics to {METRICS_CSV}")
    print(f"Saved bin-level data to {BINS_CSV}")
    print(f"Saved calibration curve to {CURVE_FIGURE}")
    print(f"Saved reliability diagram to {DIAGRAM_FIGURE}")


if __name__ == "__main__":
    main()
