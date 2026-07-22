from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
FIGURES_DIR = ROOT / "figures"
PAPER_PATH = ROOT / "paper" / "when-systems-break.pdf"


st.set_page_config(
    page_title="When Systems Break",
    page_icon="WSB",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    [data-testid="stSidebar"] {
        background: #F7F8FA;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stMetric"] label {
        color: #475467;
    }
    .section-note {
        color: #475467;
        font-size: 0.98rem;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid #D0D5DD;
        border-radius: 999px;
        padding: 0.2rem 0.62rem;
        color: #344054;
        background: #FFFFFF;
        font-size: 0.85rem;
        margin-right: 0.35rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv(name):
    path = EXPERIMENTS_DIR / name
    if not path.exists():
        st.warning(f"Missing expected result file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_result_tables():
    return {
        "ranking": read_csv("model_ranking.csv"),
        "index": read_csv("reliability_index.csv"),
        "failure": read_csv("failure_matrix.csv"),
        "calibration": read_csv("calibration_metrics.csv"),
        "confidence": read_csv("confidence_collapse.csv"),
        "shift": read_csv("shift_statistics.csv"),
        "statistics": read_csv("model_statistics.csv"),
        "refusal": read_csv("refusal_statistics.csv"),
    }


@st.cache_data
def load_default_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.target_names


@st.cache_resource
def train_demo_model():
    X, y, _ = load_default_data()
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


def image_path(name):
    path = FIGURES_DIR / name
    return path if path.exists() else None


def show_figure(name, caption=None):
    path = image_path(name)
    if path:
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: figures/{name}")


def format_pct(value):
    return f"{value * 100:.1f}%"


def section_header(title, note):
    st.header(title)
    st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def display_dataframe(frame, percent_columns=None):
    if frame.empty:
        return

    percent_columns = percent_columns or []
    styled = frame.copy()
    for column in percent_columns:
        if column in styled.columns:
            styled[column] = styled[column].map(lambda value: f"{value * 100:.2f}%")
    st.dataframe(styled, use_container_width=True, hide_index=True)


def initialize_state(X):
    if "input_data" not in st.session_state:
        st.session_state.input_data = X.head(1).copy()
    if "operations" not in st.session_state:
        st.session_state.operations = []


def reset_input(X):
    st.session_state.input_data = X.head(1).copy()
    st.session_state.operations = []


def inject_noise(noise_level, feature_std):
    rng = np.random.default_rng(42)
    X_current = st.session_state.input_data.copy()
    noise = rng.normal(0, noise_level, X_current.shape)
    noise = noise * feature_std[X_current.columns].to_numpy()
    st.session_state.input_data = X_current + noise
    st.session_state.operations.append(f"Noise level {noise_level:.2f}")


def create_missing_data():
    X_current = st.session_state.input_data.copy()
    columns_to_blank = X_current.columns[: max(1, len(X_current.columns) // 5)]
    X_current.loc[:, columns_to_blank] = np.nan
    st.session_state.input_data = X_current
    st.session_state.operations.append("Missing data")


def remove_features(train_mean):
    X_current = st.session_state.input_data.copy()
    columns_to_remove = X_current.columns[: max(1, len(X_current.columns) // 5)]
    X_current.loc[:, columns_to_remove] = train_mean[columns_to_remove]
    st.session_state.input_data = X_current
    st.session_state.operations.append("Feature removal")


def estimate_failure_risk(confidence, noise_level, missing_ratio, operation_count):
    risk = (1 - confidence) * 0.55
    risk += noise_level * 0.25
    risk += missing_ratio * 0.15
    risk += min(operation_count * 0.05, 0.20)
    return min(max(risk, 0), 1)


def render_overview(tables):
    ranking = tables["ranking"]
    index = tables["index"]
    calibration = tables["calibration"]
    shift = tables["shift"]

    section_header(
        "Overview",
        "A compact research dashboard for model failure, confidence behavior, drift, calibration, and reliability ranking.",
    )

    if not ranking.empty:
        leader = ranking.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Top Model", leader["Model"])
        cols[1].metric("Reliability Index", f"{leader['Reliability Index']:.2f}/100")
        cols[2].metric("Models Ranked", len(ranking))
        cols[3].metric("Leader Tier", leader["Tier"])

    badges = [
        "Noise robustness",
        "Missing data",
        "Feature degradation",
        "Confidence collapse",
        "Calibration",
        "Distribution shift",
        "Refusal thresholds",
        "Model ranking",
    ]
    st.markdown(
        "".join(f"<span class='status-pill'>{badge}</span>" for badge in badges),
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.55, 0.45])
    with left:
        show_figure("model_ranking.png", "Experiment 17: ranked model reliability")
    with right:
        show_figure("reliability_index.png", "Experiment 16: component-level reliability")

    if not index.empty:
        st.subheader("Reliability Index Table")
        display_dataframe(index.round(2))

    if not calibration.empty and not shift.empty:
        st.subheader("Research Snapshot")
        clean_calibration = calibration[calibration["Condition"] == "Clean"]
        max_shift = shift[shift["Mean Shift"] == shift["Mean Shift"].max()]
        cols = st.columns(3)
        cols[0].metric(
            "Best Clean ECE",
            f"{clean_calibration['Expected Calibration Error'].min() * 100:.2f}%",
        )
        cols[1].metric(
            "Worst Shift Accuracy",
            f"{max_shift['Accuracy Mean'].min() * 100:.1f}%",
        )
        cols[2].metric(
            "Largest Confidence Gap",
            f"{shift['Confidence-Accuracy Gap'].max() * 100:.1f}%",
        )


def render_experiments(tables):
    section_header(
        "Experiments",
        "The project moves from simple degradation tests to cross-experiment reliability synthesis.",
    )

    experiments = pd.DataFrame(
        [
            [1, "Baseline noise injection", "Clean vs noisy baseline"],
            [2, "Missing data simulation", "Incomplete input behavior"],
            [3, "Feature importance", "Important feature discovery"],
            [4, "Model comparison", "Algorithm-level differences"],
            [5, "Feature removal", "Performance after removing signals"],
            [6, "Noise robustness curve", "Accuracy across noise levels"],
            [7, "Threshold analysis", "Failure thresholds"],
            [8, "Failure comparison", "Multiple degradation patterns"],
            [9, "Multi-run statistics", "Mean and variance across 30 seeds"],
            [10, "Failure matrix", "Model by condition heatmap"],
            [11, "Confidence collapse", "Confidence and wrong predictions"],
            [12, "Refusal system", "Accuracy and coverage tradeoff"],
            [13, "Calibration", "Confidence vs correctness"],
            [14, "Reliability score", "Five-component reliability score"],
            [15, "Distribution shift", "Changed deployment distribution"],
            [16, "Reliability Index", "Six-dimensional synthesis"],
            [17, "Model ranking", "Final model ordering"],
        ],
        columns=["#", "Experiment", "Purpose"],
    )
    display_dataframe(experiments)

    stats = tables["statistics"]
    if not stats.empty:
        st.subheader("Thirty-Seed Statistical Robustness")
        display_dataframe(stats.round(4))

    refusal = tables["refusal"]
    if not refusal.empty:
        st.subheader("Refusal Threshold Output")
        display_dataframe(refusal.round(4))
        show_figure("accuracy_vs_coverage.png", "Accuracy improves as low-confidence cases are refused.")


def render_failure_matrix(tables):
    section_header(
        "Failure Matrix",
        "Clean, noisy, missing-data, and feature-removal performance compared across models.",
    )
    show_figure("failure_matrix.png", "Experiment 10: model failure matrix")
    display_dataframe(tables["failure"].round(2))


def render_calibration(tables):
    section_header(
        "Calibration",
        "Calibration tests whether stated confidence behaves like observed correctness.",
    )
    left, right = st.columns(2)
    with left:
        show_figure("calibration_curve.png", "Calibration curves by model and condition")
    with right:
        show_figure("reliability_diagram.png", "Reliability diagram with confidence bins")

    calibration = tables["calibration"]
    if not calibration.empty:
        summary = calibration[
            [
                "Model",
                "Condition",
                "Expected Calibration Error",
                "Mean Confidence",
                "Observed Accuracy",
                "90-100% Bin Accuracy",
            ]
        ].copy()
        display_dataframe(
            summary.round(4),
            percent_columns=[
                "Expected Calibration Error",
                "Mean Confidence",
                "Observed Accuracy",
                "90-100% Bin Accuracy",
            ],
        )


def render_confidence_collapse(tables):
    section_header(
        "Confidence Collapse",
        "As noise increases, confidence, correct-class probability, and wrong predictions move in different ways.",
    )
    show_figure("confidence_collapse.png", "Experiment 11: confidence degradation under noise")
    confidence = tables["confidence"]
    if not confidence.empty:
        display_dataframe(
            confidence.round(4),
            percent_columns=[
                "Accuracy",
                "Mean Prediction Confidence",
                "Mean Correct-Class Confidence",
                "Mean Wrong Confidence",
            ],
        )


def render_distribution_shift(tables):
    section_header(
        "Distribution Shift",
        "Train on one distribution, test on another, then compare accuracy, confidence, and drift signals.",
    )
    show_figure("distribution_shift.png", "Experiment 15: accuracy and drift detection under shift")

    shift = tables["shift"]
    if not shift.empty:
        max_shift = shift[shift["Mean Shift"] == shift["Mean Shift"].max()]
        cols = st.columns(4)
        best = max_shift.sort_values("Accuracy Mean", ascending=False).iloc[0]
        worst_gap = shift.sort_values("Confidence-Accuracy Gap", ascending=False).iloc[0]
        cols[0].metric("Best Shift Model", best["Model"])
        cols[1].metric("Best Shift Accuracy", format_pct(best["Accuracy Mean"]))
        cols[2].metric("Max Domain AUC", f"{shift['Domain AUC Mean'].max():.3f}")
        cols[3].metric("Worst Confidence Gap", format_pct(worst_gap["Confidence-Accuracy Gap"]))
        display_dataframe(
            max_shift[
                [
                    "Model",
                    "Accuracy Mean",
                    "Accuracy Drop",
                    "Confidence Mean",
                    "Domain AUC Mean",
                    "Mean PSI",
                ]
            ].round(4),
            percent_columns=["Accuracy Mean", "Accuracy Drop", "Confidence Mean"],
        )


def render_reliability_ranking(tables):
    section_header(
        "Reliability Ranking",
        "The final ranking answers which model is strongest overall under the project's declared reliability priorities.",
    )
    show_figure("model_ranking.png", "Experiment 17: model ranking by Reliability Index")

    ranking = tables["ranking"]
    if not ranking.empty:
        display_dataframe(ranking.round(2))

    st.subheader("Component Audit")
    show_figure("reliability_index.png", "Experiment 16 keeps the component scores visible.")
    index = tables["index"]
    if not index.empty:
        display_dataframe(index.round(2))


def render_interactive_lab(tables):
    section_header(
        "Interactive Failure Lab",
        "Upload or perturb one input row, then inspect prediction, confidence, and estimated failure risk.",
    )

    X_default, _, target_names = load_default_data()
    model, train_mean, train_std = train_demo_model()
    initialize_state(X_default)

    uploaded_file = st.file_uploader(
        "Upload a CSV with breast cancer feature columns",
        type=["csv"],
    )

    if uploaded_file is not None:
        uploaded = pd.read_csv(uploaded_file)
        expected_columns = list(X_default.columns)
        if all(column in uploaded.columns for column in expected_columns):
            st.session_state.input_data = uploaded[expected_columns].head(1).copy()
            st.session_state.operations = ["Uploaded data"]
        else:
            st.error("Uploaded CSV must include the same feature columns used by the breast cancer dataset.")

    left, right = st.columns([0.34, 0.66])

    with left:
        noise_percent = st.slider("Noise Level", 0, 100, 20)
        noise_level = noise_percent / 100

        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Inject Noise", use_container_width=True):
                inject_noise(noise_level, train_std)
            if st.button("Create Missing Data", use_container_width=True):
                create_missing_data()
        with action_cols[1]:
            if st.button("Remove Features", use_container_width=True):
                remove_features(train_mean)
            if st.button("Reset", use_container_width=True):
                reset_input(X_default)

        st.subheader("Applied Conditions")
        if st.session_state.operations:
            for operation in st.session_state.operations:
                st.write(operation)
        else:
            st.write("Clean input")

    with right:
        X_input = st.session_state.input_data.copy()
        missing_ratio = float(X_input.isna().mean().mean())
        X_model = X_input.fillna(train_mean)

        probabilities = model.predict_proba(X_model)
        prediction = model.predict(X_model)[0]
        confidence = float(probabilities.max(axis=1)[0])
        predicted_label = target_names[prediction]
        failure_risk = estimate_failure_risk(
            confidence,
            noise_level,
            missing_ratio,
            len(st.session_state.operations),
        )

        model_reliability = np.nan
        index = tables["index"]
        if not index.empty:
            match = index[index["Model"] == "Logistic Regression"]
            if not match.empty:
                model_reliability = float(match.iloc[0]["Reliability Index"])

        metric_cols = st.columns(4)
        metric_cols[0].metric("Prediction", predicted_label)
        metric_cols[1].metric("Confidence", f"{confidence:.1%}")
        metric_cols[2].metric("Failure Risk", f"{failure_risk:.1%}")
        metric_cols[3].metric(
            "Reliability Index",
            "N/A" if np.isnan(model_reliability) else f"{model_reliability:.1f}/100",
        )

        st.subheader("Current Input")
        st.dataframe(X_input, use_container_width=True, hide_index=True)

        st.subheader("Class Probabilities")
        probability_table = pd.DataFrame(
            {
                "Class": target_names,
                "Probability": probabilities[0],
            }
        )
        st.bar_chart(probability_table, x="Class", y="Probability")


def main():
    tables = load_result_tables()

    st.sidebar.title("When Systems Break")
    st.sidebar.caption("Research dashboard")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Experiments",
            "Failure Matrix",
            "Calibration",
            "Confidence Collapse",
            "Distribution Shift",
            "Reliability Ranking",
            "Interactive Lab",
        ],
    )

    st.sidebar.divider()
    if PAPER_PATH.exists():
        st.sidebar.markdown(f"[Research paper]({PAPER_PATH.as_posix()})")
    st.sidebar.markdown("[GitHub README](README.md)")
    st.sidebar.markdown("[Experiment scripts](experiments)")

    st.title("When Systems Break")
    st.caption("A research dashboard for machine learning reliability under degraded data.")

    if page == "Overview":
        render_overview(tables)
    elif page == "Experiments":
        render_experiments(tables)
    elif page == "Failure Matrix":
        render_failure_matrix(tables)
    elif page == "Calibration":
        render_calibration(tables)
    elif page == "Confidence Collapse":
        render_confidence_collapse(tables)
    elif page == "Distribution Shift":
        render_distribution_shift(tables)
    elif page == "Reliability Ranking":
        render_reliability_ranking(tables)
    else:
        render_interactive_lab(tables)


if __name__ == "__main__":
    main()
