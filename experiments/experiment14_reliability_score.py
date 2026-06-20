from pathlib import Path

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


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
OUTPUT_CSV = EXPERIMENT_DIR / "reliability_scores.csv"
RUN_OUTPUT_CSV = EXPERIMENT_DIR / "reliability_run_metrics.csv"
OUTPUT_FIGURE = EXPERIMENT_DIR / "reliability_scores.png"
ROOT_FIGURE = ROOT_DIR / "figures" / "reliability_scores.png"
PAPER_FIGURE = ROOT_DIR / "paper" / "figures" / "reliability_scores.png"

N_RUNS = 30
NOISE_LEVEL = 0.35
MISSING_RATIO = 0.25
REMOVED_FEATURES = 5
VARIANCE_TOLERANCE = 0.05

WEIGHTS = {
    "Accuracy Score": 0.30,
    "Robustness Score": 0.25,
    "Confidence Stability Score": 0.15,
    "Refusal Quality Score": 0.20,
    "Repeatability Score": 0.10,
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


def add_missing(X, rng, train_means):
    degraded = X.copy()
    mask = rng.random(degraded.shape) < MISSING_RATIO
    degraded[mask] = np.nan
    return degraded.fillna(train_means)


def top_features(X_train, y_train, seed):
    selector = RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1
    )
    selector.fit(X_train, y_train)
    importances = pd.Series(selector.feature_importances_, index=X_train.columns)
    return importances.nlargest(REMOVED_FEATURES).index.tolist()


def evaluate(model, X, y):
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    class_indices = {label: index for index, label in enumerate(model.classes_)}
    predicted_indices = np.array([class_indices[label] for label in predictions])
    confidence = probabilities[np.arange(len(predictions)), predicted_indices]
    correct = predictions == np.asarray(y)
    return accuracy_score(y, predictions), confidence, correct


def confidence_discrimination(confidence, correct):
    correctness = np.concatenate(correct).astype(int)
    pooled_confidence = np.concatenate(confidence)
    if np.unique(correctness).size < 2:
        return 0.5
    return roc_auc_score(correctness, pooled_confidence)


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    run_rows = []
    pooled = {
        name: {"confidence": [], "correct": []}
        for name in build_models(0)
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

        X_noise = add_noise(X_test, rng)
        X_missing = add_missing(X_test, rng, X_train.mean())
        removed = top_features(X_train, y_train, seed)
        X_train_removed = X_train.drop(columns=removed)
        X_test_removed = X_test.drop(columns=removed)

        for model_name, model in build_models(seed).items():
            model.fit(X_train, y_train)
            condition_results = {}

            for condition, condition_X in {
                "Clean": X_test,
                "Noise": X_noise,
                "Missing": X_missing,
            }.items():
                accuracy, confidence, correct = evaluate(model, condition_X, y_test)
                condition_results[condition] = (accuracy, confidence.mean())
                pooled[model_name]["confidence"].append(confidence)
                pooled[model_name]["correct"].append(correct)

            removal_model = build_models(seed)[model_name]
            removal_model.fit(X_train_removed, y_train)
            removal_accuracy, removal_confidence, removal_correct = evaluate(
                removal_model, X_test_removed, y_test
            )
            condition_results["Feature Removal"] = (
                removal_accuracy,
                removal_confidence.mean(),
            )
            pooled[model_name]["confidence"].append(removal_confidence)
            pooled[model_name]["correct"].append(removal_correct)

            for condition, (accuracy, mean_confidence) in condition_results.items():
                run_rows.append(
                    {
                        "Seed": seed,
                        "Model": model_name,
                        "Condition": condition,
                        "Accuracy": accuracy,
                        "Mean Confidence": mean_confidence,
                    }
                )

    runs = pd.DataFrame(run_rows)
    runs.round(6).to_csv(RUN_OUTPUT_CSV, index=False)
    score_rows = []

    for model_name, model_runs in runs.groupby("Model"):
        clean = model_runs[model_runs["Condition"] == "Clean"]
        degraded = model_runs[model_runs["Condition"] != "Clean"]

        clean_accuracy = clean["Accuracy"].mean()
        degraded_accuracy = degraded["Accuracy"].mean()
        accuracy_score_component = clean_accuracy * 100
        robustness_score = min(degraded_accuracy / clean_accuracy, 1.0) * 100

        condition_summary = model_runs.groupby("Condition")[["Accuracy", "Mean Confidence"]].mean()
        confidence_gap = (
            condition_summary["Mean Confidence"] - condition_summary["Accuracy"]
        ).abs().mean()
        confidence_stability_score = max(0.0, 1.0 - confidence_gap) * 100

        refusal_quality_score = confidence_discrimination(
            pooled[model_name]["confidence"], pooled[model_name]["correct"]
        ) * 100

        condition_std = model_runs.groupby("Condition")["Accuracy"].std().mean()
        repeatability_score = max(
            0.0, 1.0 - min(condition_std / VARIANCE_TOLERANCE, 1.0)
        ) * 100

        components = {
            "Accuracy Score": accuracy_score_component,
            "Robustness Score": robustness_score,
            "Confidence Stability Score": confidence_stability_score,
            "Refusal Quality Score": refusal_quality_score,
            "Repeatability Score": repeatability_score,
        }
        reliability_score = sum(
            components[name] * weight for name, weight in WEIGHTS.items()
        )

        score_rows.append(
            {
                "Model": model_name,
                "Reliability Score": reliability_score,
                **components,
                "Clean Accuracy Mean": clean_accuracy,
                "Degraded Accuracy Mean": degraded_accuracy,
                "Mean Accuracy Std": condition_std,
            }
        )

    scores = pd.DataFrame(score_rows).sort_values(
        "Reliability Score", ascending=False
    )
    component_columns = ["Reliability Score", *WEIGHTS.keys()]
    diagnostic_columns = [
        "Clean Accuracy Mean",
        "Degraded Accuracy Mean",
        "Mean Accuracy Std",
    ]
    scores[component_columns] = scores[component_columns].round(2)
    scores[diagnostic_columns] = scores[diagnostic_columns].round(4)
    scores.to_csv(OUTPUT_CSV, index=False)

    colors = ["#176B5B", "#278F7A", "#D79A28", "#B84A5F"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(
        scores["Model"][::-1],
        scores["Reliability Score"][::-1],
        color=colors[::-1],
        height=0.58,
    )
    ax.bar_label(
        bars,
        labels=[
            f"{value:.1f}" for value in scores["Reliability Score"][::-1]
        ],
        padding=6,
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Reliability Score (0-100)")
    fig.suptitle(
        "Model Reliability Score",
        x=0.16,
        y=0.96,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.16,
        0.91,
        "Accuracy, robustness, confidence stability, refusal quality, and repeatability",
        fontsize=9.5,
        color="#555555",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.14, top=0.82)

    for path in [OUTPUT_FIGURE, ROOT_FIGURE, PAPER_FIGURE]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")

    display_columns = [
        "Model",
        "Reliability Score",
        "Accuracy Score",
        "Robustness Score",
        "Confidence Stability Score",
        "Refusal Quality Score",
        "Repeatability Score",
    ]
    print("\nModel Reliability Score (0-100):\n")
    print(scores[display_columns].to_string(index=False))
    print(f"\nSaved scores to {OUTPUT_CSV}")
    print(f"Saved run-level metrics to {RUN_OUTPUT_CSV}")
    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
