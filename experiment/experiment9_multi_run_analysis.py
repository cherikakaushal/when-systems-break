from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


OUTPUT_PATH = Path(__file__).resolve().parent / "model_statistics.csv"
N_RUNS = 30
NOISE_LEVEL = 0.2


def build_models(seed):
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=seed),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(),
        ),
    }


def add_noise(X, rng, noise_level=NOISE_LEVEL):
    feature_scale = X.std(axis=0).replace(0, 1)
    noise = rng.normal(0, noise_level, X.shape) * feature_scale.to_numpy()
    return X + noise


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    run_results = []

    for seed in range(N_RUNS):
        rng = np.random.default_rng(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )

        X_test_noisy = add_noise(X_test, rng)

        for model_name, model in build_models(seed).items():
            model.fit(X_train, y_train)

            clean_accuracy = accuracy_score(y_test, model.predict(X_test))
            noise_accuracy = accuracy_score(y_test, model.predict(X_test_noisy))

            run_results.append(
                {
                    "Seed": seed,
                    "Model": model_name,
                    "Accuracy": clean_accuracy,
                    "Noise Accuracy": noise_accuracy,
                }
            )

    results = pd.DataFrame(run_results)

    statistics = (
        results.groupby("Model")
        .agg(
            **{
                "Mean Acc": ("Accuracy", "mean"),
                "Std Acc": ("Accuracy", "std"),
                "Noise Mean Acc": ("Noise Accuracy", "mean"),
                "Noise Std Acc": ("Noise Accuracy", "std"),
            }
        )
        .reset_index()
        .sort_values(by="Noise Mean Acc", ascending=False)
    )

    statistics.to_csv(OUTPUT_PATH, index=False)

    print("\nMulti-run model statistics:\n")
    print(statistics.to_string(index=False))
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
