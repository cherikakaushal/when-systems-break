import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from paths import result_path, save_figure

OUTPUT_CSV = result_path("shift_statistics.csv")
OUTPUT_FIGURE = "distribution_shift.png"

N_RUNS = 30
SHIFT_LEVELS = np.linspace(0.0, 0.5, 6)
PSI_BINS = 10
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


def shift_distribution(X, train_mean, train_std, mean_shift, std_multiplier):
    standardized = (X - train_mean) / train_std
    shifted = mean_shift + std_multiplier * standardized
    return shifted * train_std + train_mean


def prediction_metrics(model, X, y):
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    class_indices = {label: index for index, label in enumerate(model.classes_)}
    predicted_indices = np.array([class_indices[label] for label in predictions])
    confidence = probabilities[np.arange(len(predictions)), predicted_indices]
    return accuracy_score(y, predictions), float(confidence.mean())


def domain_classifier_auc(X_train, X_shifted, seed):
    source = pd.concat([X_train, X_shifted], ignore_index=True)
    domain = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_shifted), dtype=int)]
    )
    X_detector_train, X_detector_test, y_detector_train, y_detector_test = (
        train_test_split(
            source,
            domain,
            test_size=0.3,
            random_state=seed,
            stratify=domain,
        )
    )
    detector = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    detector.fit(X_detector_train, y_detector_train)
    target_probability = detector.predict_proba(X_detector_test)[:, 1]
    return roc_auc_score(y_detector_test, target_probability)


def population_stability_index(reference, shifted):
    epsilon = 1e-6
    feature_scores = []

    for column in reference.columns:
        quantiles = np.linspace(0, 1, PSI_BINS + 1)[1:-1]
        internal_edges = np.unique(reference[column].quantile(quantiles).to_numpy())
        edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))
        if len(edges) <= 2:
            continue

        reference_counts, _ = np.histogram(reference[column], bins=edges)
        shifted_counts, _ = np.histogram(shifted[column], bins=edges)
        reference_ratio = np.clip(reference_counts / len(reference), epsilon, None)
        shifted_ratio = np.clip(shifted_counts / len(shifted), epsilon, None)
        psi = np.sum(
            (shifted_ratio - reference_ratio)
            * np.log(shifted_ratio / reference_ratio)
        )
        feature_scores.append(psi)

    return float(np.mean(feature_scores))


def collect_runs():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model_rows = []
    drift_rows = []

    for seed in range(N_RUNS):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        train_mean = X_train.mean()
        train_std = X_train.std().replace(0, 1)
        models = build_models(seed)
        for model in models.values():
            model.fit(X_train, y_train)

        for shift_level in SHIFT_LEVELS:
            std_multiplier = 1.0 + shift_level
            X_shifted = shift_distribution(
                X_test,
                train_mean,
                train_std,
                mean_shift=shift_level,
                std_multiplier=std_multiplier,
            )
            detector_auc = domain_classifier_auc(
                X_train, X_shifted, seed + int(shift_level * 100)
            )
            mean_psi = population_stability_index(X_train, X_shifted)
            drift_rows.append(
                {
                    "Seed": seed,
                    "Mean Shift": shift_level,
                    "Std Multiplier": std_multiplier,
                    "Domain Classifier AUC": detector_auc,
                    "Mean PSI": mean_psi,
                }
            )

            for model_name, model in models.items():
                accuracy, confidence = prediction_metrics(model, X_shifted, y_test)
                model_rows.append(
                    {
                        "Seed": seed,
                        "Model": model_name,
                        "Mean Shift": shift_level,
                        "Std Multiplier": std_multiplier,
                        "Accuracy": accuracy,
                        "Mean Confidence": confidence,
                    }
                )

    return pd.DataFrame(model_rows), pd.DataFrame(drift_rows)


def summarize(model_runs, drift_runs):
    model_summary = (
        model_runs.groupby(["Model", "Mean Shift", "Std Multiplier"])
        .agg(
            **{
                "Accuracy Mean": ("Accuracy", "mean"),
                "Accuracy Std": ("Accuracy", "std"),
                "Confidence Mean": ("Mean Confidence", "mean"),
                "Confidence Std": ("Mean Confidence", "std"),
            }
        )
        .reset_index()
    )
    drift_summary = (
        drift_runs.groupby(["Mean Shift", "Std Multiplier"])
        .agg(
            **{
                "Domain AUC Mean": ("Domain Classifier AUC", "mean"),
                "Domain AUC Std": ("Domain Classifier AUC", "std"),
                "Mean PSI": ("Mean PSI", "mean"),
                "PSI Std": ("Mean PSI", "std"),
            }
        )
        .reset_index()
    )
    summary = model_summary.merge(
        drift_summary, on=["Mean Shift", "Std Multiplier"], how="left"
    )
    baseline = summary[summary["Mean Shift"] == 0].set_index("Model")[
        "Accuracy Mean"
    ]
    summary["Accuracy Drop"] = summary.apply(
        lambda row: baseline[row["Model"]] - row["Accuracy Mean"], axis=1
    )
    summary["Confidence-Accuracy Gap"] = (
        summary["Confidence Mean"] - summary["Accuracy Mean"]
    ).abs()
    return summary.sort_values(["Mean Shift", "Model"])


def draw_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    accuracy_ax, confidence_ax, auc_ax, psi_ax = axes.flat

    for model_name, color in COLORS.items():
        subset = summary[summary["Model"] == model_name]
        accuracy_ax.plot(
            subset["Mean Shift"],
            subset["Accuracy Mean"] * 100,
            marker="o",
            linewidth=2,
            color=color,
            label=model_name,
        )
        confidence_ax.plot(
            subset["Mean Shift"],
            subset["Confidence Mean"] * 100,
            marker="o",
            linewidth=2,
            color=color,
            label=model_name,
        )

    drift = summary[summary["Model"] == "Logistic Regression"]
    auc_ax.plot(
        drift["Mean Shift"],
        drift["Domain AUC Mean"],
        marker="o",
        linewidth=2.2,
        color="#674B8C",
    )
    auc_ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.2)
    psi_ax.plot(
        drift["Mean Shift"],
        drift["Mean PSI"],
        marker="o",
        linewidth=2.2,
        color="#B44B36",
    )
    psi_ax.axhline(0.1, color="#666666", linestyle="--", linewidth=1.2)
    psi_ax.axhline(0.25, color="#666666", linestyle=":", linewidth=1.2)

    accuracy_ax.set_title("Predictive Accuracy", fontweight="bold")
    confidence_ax.set_title("Mean Prediction Confidence", fontweight="bold")
    auc_ax.set_title("Shift Detectability", fontweight="bold")
    psi_ax.set_title("Population Stability Index", fontweight="bold")
    accuracy_ax.set_ylabel("Accuracy (%)")
    confidence_ax.set_ylabel("Confidence (%)")
    auc_ax.set_ylabel("Domain Classifier ROC AUC")
    psi_ax.set_ylabel("Mean PSI Across Features")
    auc_ax.set_ylim(0.45, 1.02)
    confidence_ax.legend(loc="lower left", fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("Test Mean Shift (training standard deviations)")
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Distribution Shift: Performance and Detectability",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.94,
        "Test scale increases from 1.0 to 1.5 as the mean shifts from 0.0 to 0.5",
        ha="center",
        color="#555555",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    save_figure(fig, OUTPUT_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    model_runs, drift_runs = collect_runs()
    summary = summarize(model_runs, drift_runs)
    summary.round(6).to_csv(OUTPUT_CSV, index=False)
    draw_figure(summary)

    endpoint = summary[summary["Mean Shift"] == SHIFT_LEVELS[-1]].copy()
    for column in ["Accuracy Mean", "Confidence Mean", "Accuracy Drop"]:
        endpoint[column] = endpoint[column].map(lambda value: f"{value:.2%}")
    print("\nDistribution shift endpoint (mean=0.5, std=1.5):\n")
    print(
        endpoint[
            [
                "Model",
                "Accuracy Mean",
                "Accuracy Drop",
                "Confidence Mean",
                "Domain AUC Mean",
                "Mean PSI",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved statistics to {OUTPUT_CSV}")
    print(f"Saved figure to figures/{OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
