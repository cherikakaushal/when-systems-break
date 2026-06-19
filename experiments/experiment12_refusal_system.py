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
OUTPUT_CSV = EXPERIMENT_DIR / "refusal_statistics.csv"
OUTPUT_FIGURE = EXPERIMENT_DIR / "accuracy_vs_coverage.png"
ROOT_FIGURE = ROOT_DIR / "figures" / "accuracy_vs_coverage.png"

SEED = 42
NOISE_LEVEL = 0.8
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def add_noise(X, rng, noise_level):
    feature_scale = X.std(axis=0).replace(0, 1)
    noise = rng.normal(0, noise_level, X.shape) * feature_scale.to_numpy()
    return X + noise


def main():
    rng = np.random.default_rng(SEED)

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

    X_noisy = add_noise(X_test, rng, NOISE_LEVEL)
    probabilities = model.predict_proba(X_noisy)
    predictions = model.predict(X_noisy)
    confidence = probabilities.max(axis=1)

    rows = []

    for threshold in THRESHOLDS:
        accepted = confidence >= threshold
        refused = ~accepted
        coverage = accepted.mean()
        refusal_rate = refused.mean()

        if accepted.any():
            selective_accuracy = accuracy_score(y_test[accepted], predictions[accepted])
        else:
            selective_accuracy = np.nan

        rows.append(
            {
                "Threshold": threshold,
                "Coverage": coverage,
                "Accuracy": selective_accuracy,
                "Refusal Rate": refusal_rate,
                "Accepted Predictions": int(accepted.sum()),
                "Refused Predictions": int(refused.sum()),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        results["Coverage"] * 100,
        results["Accuracy"] * 100,
        marker="o",
        linewidth=2.2,
        color="#2E7D6F",
    )

    for _, row in results.iterrows():
        ax.annotate(
            f"{row['Threshold']:.2f}",
            (row["Coverage"] * 100, row["Accuracy"] * 100),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )

    ax.set_title("Accuracy vs Coverage Under Refusal Thresholds")
    ax.set_xlabel("Coverage (% predictions accepted)")
    ax.set_ylabel("Selective Accuracy (% accepted predictions correct)")
    ax.set_xlim(0, 105)
    ax.set_ylim(80, 101)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=200)
    fig.savefig(ROOT_FIGURE, dpi=200)

    display = results.copy()
    for column in ["Coverage", "Accuracy", "Refusal Rate"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")

    print("\nRefusal threshold results:\n")
    print(display.to_string(index=False))
    print(f"\nSaved results to {OUTPUT_CSV}")
    print(f"Saved figure to {OUTPUT_FIGURE}")
    print(f"Saved figure copy to {ROOT_FIGURE}")


if __name__ == "__main__":
    main()
