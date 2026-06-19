from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
OUTPUT_CSV = EXPERIMENT_DIR / "confidence_collapse.csv"
OUTPUT_FIGURE = EXPERIMENT_DIR / "confidence_collapse.png"
ROOT_FIGURE = ROOT_DIR / "figures" / "confidence_collapse.png"

SEED = 42
NOISE_LEVELS = np.linspace(0, 1.0, 11)


def add_noise(X, rng, noise_level):
    feature_scale = X.std(axis=0).replace(0, 1)
    noise = rng.normal(0, noise_level, X.shape) * feature_scale.to_numpy()
    return X + noise


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=SEED),
    )
    model.fit(X_train, y_train)

    rows = []

    for noise_level in NOISE_LEVELS:
        rng = np.random.default_rng(SEED)
        X_noisy = add_noise(X_test, rng, noise_level)

        probabilities = model.predict_proba(X_noisy)
        predictions = model.predict(X_noisy)
        prediction_confidence = probabilities.max(axis=1)
        correct_class_confidence = probabilities[np.arange(len(y_test)), y_test]
        wrong_mask = predictions != y_test

        rows.append(
            {
                "Noise Level": noise_level,
                "Accuracy": accuracy_score(y_test, predictions),
                "Mean Prediction Confidence": prediction_confidence.mean(),
                "Mean Correct-Class Confidence": correct_class_confidence.mean(),
                "Wrong Predictions": int(wrong_mask.sum()),
                "Mean Wrong Confidence": (
                    prediction_confidence[wrong_mask].mean() if wrong_mask.any() else 0
                ),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        results["Noise Level"],
        results["Accuracy"],
        marker="o",
        linewidth=2,
        label="Accuracy",
        color="#2E7D6F",
    )
    ax1.plot(
        results["Noise Level"],
        results["Mean Correct-Class Confidence"],
        marker="s",
        linewidth=2,
        label="Correct-Class Confidence",
        color="#A7445B",
    )
    ax2.bar(
        results["Noise Level"],
        results["Wrong Predictions"],
        width=0.055,
        alpha=0.25,
        label="Wrong Predictions",
        color="#D49A2A",
    )

    ax1.set_title("Confidence Collapse Under Increasing Noise")
    ax1.set_xlabel("Noise Level")
    ax1.set_ylabel("Accuracy / Confidence")
    ax2.set_ylabel("Wrong Predictions")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc="lower left")

    fig.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=200)
    fig.savefig(ROOT_FIGURE, dpi=200)

    print("\nConfidence collapse results:\n")
    print(results.to_string(index=False))
    print(f"\nSaved results to {OUTPUT_CSV}")
    print(f"Saved figure to {OUTPUT_FIGURE}")
    print(f"Saved figure copy to {ROOT_FIGURE}")


if __name__ == "__main__":
    main()
