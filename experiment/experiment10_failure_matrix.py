from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
MATRIX_CSV = EXPERIMENT_DIR / "failure_matrix.csv"
MATRIX_FIGURE = EXPERIMENT_DIR / "failure_matrix.png"
ROOT_FIGURE = ROOT_DIR / "figures" / "failure_matrix.png"

NOISE_LEVEL = 0.35
MISSING_RATIO = 0.25
REMOVED_FEATURES = 5
SEED = 42


def build_models(seed):
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=seed),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(),
        ),
    }


def add_noise(X, rng):
    feature_scale = X.std(axis=0).replace(0, 1)
    noise = rng.normal(0, NOISE_LEVEL, X.shape) * feature_scale.to_numpy()
    return X + noise


def add_missing(X, rng):
    X_missing = X.copy()
    mask = rng.random(X_missing.shape) < MISSING_RATIO
    X_missing[mask] = np.nan
    return X_missing


def top_features_to_remove(X_train, y_train):
    selector = RandomForestClassifier(n_estimators=200, random_state=SEED)
    selector.fit(X_train, y_train)

    importances = pd.Series(selector.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False).head(REMOVED_FEATURES).index.tolist()


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

    train_means = X_train.mean()
    X_test_noise = add_noise(X_test, rng)
    X_test_missing = add_missing(X_test, rng).fillna(train_means)

    removed_features = top_features_to_remove(X_train, y_train)
    X_train_removed = X_train.drop(columns=removed_features)
    X_test_removed = X_test.drop(columns=removed_features)

    rows = []

    for model_name, model in build_models(SEED).items():
        model.fit(X_train, y_train)

        clean = accuracy_score(y_test, model.predict(X_test))
        noise = accuracy_score(y_test, model.predict(X_test_noise))
        missing = accuracy_score(y_test, model.predict(X_test_missing))

        removal_model = build_models(SEED)[model_name]
        removal_model.fit(X_train_removed, y_train)
        feature_removal = accuracy_score(y_test, removal_model.predict(X_test_removed))

        rows.append(
            {
                "Model": model_name,
                "Clean": round(clean * 100, 2),
                "Noise": round(noise * 100, 2),
                "Missing": round(missing * 100, 2),
                "Feature Removal": round(feature_removal * 100, 2),
            }
        )

    matrix = pd.DataFrame(rows).set_index("Model")
    matrix.to_csv(MATRIX_CSV)

    plt.figure(figsize=(9, 4.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=50,
        vmax=100,
        linewidths=0.6,
        cbar_kws={"label": "Accuracy (%)"},
    )
    plt.title("Failure Matrix Across Models and Data Degradation")
    plt.xlabel("Condition")
    plt.ylabel("Model")
    plt.tight_layout()

    MATRIX_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    ROOT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(MATRIX_FIGURE, dpi=200)
    plt.savefig(ROOT_FIGURE, dpi=200)

    print("\nFailure matrix (% accuracy):\n")
    print(matrix)
    print(f"\nSaved matrix to {MATRIX_CSV}")
    print(f"Saved heatmap to {MATRIX_FIGURE}")
    print(f"Saved hero image to {ROOT_FIGURE}")
    print(f"\nRemoved features: {', '.join(removed_features)}")


if __name__ == "__main__":
    main()
