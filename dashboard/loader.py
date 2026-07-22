import re

import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import EXPERIMENTS_DIR, FIGURES_DIR, RESULTS_DIR


DATASETS = {
    "Breast Cancer": load_breast_cancer,
    "Wine": load_wine,
    "Iris": load_iris,
}

DATASET_OPTIONS = [
    "Breast Cancer",
    "Wine",
    "Iris",
    "Heart Disease (placeholder)",
    "Titanic (placeholder)",
    "Custom CSV (placeholder)",
]


EXPERIMENTS = [
    (1, "Baseline Noise Injection", "Compare clean performance with noisy-input performance."),
    (2, "Missing Data Simulation", "Study how incomplete inputs affect predictions."),
    (3, "Feature Importance", "Identify which features contribute most to model behavior."),
    (4, "Model Comparison", "Compare algorithms under clean, noisy, and missing inputs."),
    (5, "Feature Removal", "Measure degradation after removing important signals."),
    (6, "Noise Robustness Curve", "Track accuracy as noise increases."),
    (7, "Threshold Analysis", "Find reliability thresholds under perturbation."),
    (8, "Failure Comparison", "Compare multiple degradation patterns."),
    (9, "Multi-Run Statistics", "Estimate mean and variance across 30 random seeds."),
    (10, "Failure Matrix", "Generate the model-by-condition heatmap."),
    (11, "Confidence Collapse", "Track confidence decline and wrong predictions under noise."),
    (12, "Refusal System", "Measure accuracy, coverage, and refusal rate."),
    (13, "Calibration Analysis", "Test whether predicted confidence matches correctness."),
    (14, "Reliability Score Framework", "Combine accuracy, robustness, confidence, refusal quality, and variance."),
    (15, "Distribution Shift", "Measure performance loss under changed test distributions."),
    (16, "Reliability Index", "Synthesize six reliability dimensions into one score."),
    (17, "Model Ranking", "Rank all models by Reliability Index."),
]


@st.cache_data
def read_csv(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_tables():
    return {path.stem: pd.read_csv(path) for path in sorted(RESULTS_DIR.glob("*.csv"))}


@st.cache_data
def load_dataset(dataset_name):
    data = DATASETS[dataset_name]()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.target_names


@st.cache_resource
def train_model(dataset_name):
    X, y, _ = load_dataset(dataset_name)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42),
    )
    model.fit(X_train, y_train)
    return model, X_train.mean(), X_train.std().replace(0, 1)


def result_path(name):
    return RESULTS_DIR / name


def figure_path(name):
    return FIGURES_DIR / name


def csv_outputs():
    return sorted(RESULTS_DIR.glob("*.csv"))


def figure_outputs():
    return sorted(FIGURES_DIR.glob("*.png"))


def detect_experiment_outputs(script_path):
    text = script_path.read_text(encoding="utf-8")
    csv_names = sorted(set(re.findall(r'["\']([\w\-]+\.csv)["\']', text)))
    figure_names = sorted(set(re.findall(r'["\']([\w\-]+\.png)["\']', text)))
    return {
        "csv": [RESULTS_DIR / name for name in csv_names if (RESULTS_DIR / name).exists()],
        "figures": [FIGURES_DIR / name for name in figure_names if (FIGURES_DIR / name).exists()],
    }


def experiment_catalog():
    catalog = []
    scripts = {extract_number(path.name): path for path in EXPERIMENTS_DIR.glob("experiment*.py")}
    for number, title, purpose in EXPERIMENTS:
        script = scripts.get(number)
        outputs = detect_experiment_outputs(script) if script else {"csv": [], "figures": []}
        catalog.append(
            {
                "number": number,
                "title": title,
                "purpose": purpose,
                "script": script,
                **outputs,
            }
        )
    return catalog


def extract_number(filename):
    match = re.match(r"experiment(\d+)", filename)
    return int(match.group(1)) if match else None
