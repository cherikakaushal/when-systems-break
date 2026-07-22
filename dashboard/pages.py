import numpy as np
import pandas as pd
import streamlit as st

from .charts import meter, probability_chart, radar_chart, show_figure
from .components import card, section, step
from .loader import DATASET_OPTIONS, DATASETS, csv_outputs, experiment_catalog, figure_outputs, load_dataset, load_tables, result_path, train_model
from .metrics import confidence_meter_label, estimate_failure_risk, latest_findings, reliability_label, repository_stats
from .tables import csv_download, display_table
from .utils import PAPER_PATH, format_pct


def render_overview():
    tables = load_tables()
    catalog = experiment_catalog()
    ranking = tables.get("model_ranking", pd.DataFrame())
    index = tables.get("reliability_index", pd.DataFrame())

    section(
        "Overview",
        "Research summary for robustness, calibration, confidence collapse, distribution shift, refusal behavior, and model ranking.",
    )
    if not ranking.empty:
        leader = ranking.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Top Model", leader["Model"])
        cols[1].metric("Reliability Index", f"{leader['Reliability Index']:.2f}/100")
        cols[2].metric("Experiments Completed", len(catalog))
        cols[3].metric("Supported Datasets", "3 + 3 planned")
        cols[4].metric("Leader Tier", leader["Tier"])

    left, right = st.columns([0.54, 0.46])
    with left:
        show_figure("model_ranking.png", "Overall Reliability Ranking")
    with right:
        show_figure("reliability_index.png", "Reliability Component Heatmap")

    st.subheader("Reliability Table")
    display_table(index.round(2), height=220)

    st.subheader("Research Summary")
    cols = st.columns(3)
    cols[0].markdown("**Framework Scope**")
    cols[0].write("Noise, missing values, feature degradation, calibration, refusal, variance, and shift.")
    cols[1].markdown("**Decision Question**")
    cols[1].write("Which model remains reliable when deployment conditions stop being clean?")
    cols[2].markdown("**Research Use**")
    cols[2].write("Compare failure modes without reducing reliability to one accuracy number.")

    st.subheader("Latest Findings")
    for finding in latest_findings(tables):
        st.write(f"- {finding}")

    st.subheader("Repository Statistics")
    stats = repository_stats(catalog, tables)
    stat_cols = st.columns(4)
    for col, (label, value) in zip(stat_cols, stats.items()):
        col.metric(label, value)


def render_experiments():
    section(
        "Experiments",
        "Each experiment keeps its source script in experiments/, CSV outputs in results/, and figures in figures/.",
    )
    catalog = experiment_catalog()
    summary = pd.DataFrame(
        {
            "#": item["number"],
            "Title": item["title"],
            "Purpose": item["purpose"],
            "Figures": len(item["figures"]),
            "CSVs": len(item["csv"]),
        }
        for item in catalog
    )
    display_table(summary, height=430)

    selected_number = st.selectbox(
        "Inspect Experiment",
        [item["number"] for item in catalog],
        format_func=lambda number: f"Experiment {number:02d}",
    )
    selected = next(item for item in catalog if item["number"] == selected_number)
    st.subheader(selected["title"])
    st.write(selected["purpose"])
    if selected["script"]:
        st.caption(f"Script: {selected['script'].relative_to(selected['script'].parent.parent)}")

    if selected["figures"]:
        cols = st.columns(min(2, len(selected["figures"])))
        for index, path in enumerate(selected["figures"]):
            with cols[index % len(cols)]:
                st.image(str(path), caption=path.name, width="stretch")
                _download_path(path, "Download Figure", "image/png")
    else:
        st.info("No figure output detected for this experiment.")

    if selected["csv"]:
        for path in selected["csv"]:
            frame = pd.read_csv(path)
            st.markdown(f"**{path.name}**")
            display_table(frame.head(20), height=260)
            _download_path(path, f"Download {path.name}", "text/csv")
    else:
        st.info("No CSV output detected for this experiment.")


def render_failure_matrix():
    tables = load_tables()
    failure = tables.get("failure_matrix", pd.DataFrame())
    section(
        "Failure Matrix",
        "A model-by-condition view of clean, noisy, missing-data, and feature-removal performance.",
    )
    show_figure("failure_matrix.png", "Failure Matrix Heatmap")
    csv_download(failure, "failure_matrix.csv", "Download Failure Matrix CSV")
    display_table(failure.round(2), height=230)
    card(
        "Research Interpretation",
        "Robustness is condition-specific. Models that perform well on clean inputs can rank differently when information is noisy, missing, or degraded.",
    )


def render_calibration():
    tables = load_tables()
    calibration = tables.get("calibration_metrics", pd.DataFrame())
    section(
        "Calibration",
        "Calibration asks whether stated confidence matches observed correctness.",
    )
    left, right = st.columns(2)
    with left:
        show_figure("calibration_curve.png", "Calibration Curve")
    with right:
        show_figure("reliability_diagram.png", "Reliability Diagram")
    summary = calibration.copy()
    if not summary.empty:
        summary["Brier Score Proxy"] = (
            summary["Mean Confidence"] - summary["Observed Accuracy"]
        ).abs().round(4)
    display_table(
        summary.round(4),
        percent_columns=[
            "Expected Calibration Error",
            "Mean Confidence",
            "Observed Accuracy",
            "90-100% Bin Accuracy",
            "Brier Score Proxy",
        ],
        height=320,
    )
    csv_download(calibration, "calibration_metrics.csv", "Download Calibration Metrics")
    card(
        "Calibration Quality",
        "Lower ECE indicates better agreement between confidence and correctness. The diagrams reveal where scalar calibration metrics hide local errors.",
    )


def render_confidence_collapse():
    tables = load_tables()
    confidence = tables.get("confidence_collapse", pd.DataFrame())
    section(
        "Confidence Collapse",
        "Tracks how confidence and wrong predictions change as input noise increases.",
    )
    show_figure("confidence_collapse.png", "Confidence Collapse Under Noise")
    display_table(
        confidence.round(4),
        percent_columns=[
            "Accuracy",
            "Mean Prediction Confidence",
            "Mean Correct-Class Confidence",
            "Mean Wrong Confidence",
        ],
        height=320,
    )
    csv_download(confidence, "confidence_collapse.csv", "Download Confidence Collapse CSV")
    if not confidence.empty:
        st.subheader("Misclassification Analysis")
        worst = confidence.sort_values("Wrong Predictions", ascending=False).head(5)
        display_table(worst, height=220)
        st.subheader("Confidence Histogram")
        st.bar_chart(confidence, x="Noise Level", y="Mean Prediction Confidence")
    card(
        "Failure Explanation",
        "Confidence can remain high even as wrong predictions increase. That gap is why confidence needs calibration, refusal thresholds, and degradation testing.",
    )


def render_distribution_shift():
    tables = load_tables()
    shift = tables.get("shift_statistics", pd.DataFrame())
    section(
        "Distribution Shift",
        "Train on one distribution, test on another, and compare accuracy, confidence, domain AUC, and PSI.",
    )
    show_figure("distribution_shift.png", "Distribution Shift Analysis")
    if not shift.empty:
        endpoint = shift[shift["Mean Shift"] == shift["Mean Shift"].max()]
        cols = st.columns(4)
        best = endpoint.sort_values("Accuracy Mean", ascending=False).iloc[0]
        cols[0].metric("Best Shift Model", best["Model"])
        cols[1].metric("Shift Accuracy", format_pct(best["Accuracy Mean"]))
        cols[2].metric("Shift Score", f"{shift['Domain AUC Mean'].max():.3f}")
        cols[3].metric("Mean PSI", f"{shift['Mean PSI'].max():.3f}")
        display_table(endpoint.round(4), percent_columns=["Accuracy Mean", "Accuracy Drop", "Confidence Mean"], height=260)
    csv_download(shift, "shift_statistics.csv", "Download Shift Statistics")
    card(
        "Research Interpretation",
        "Prediction confidence and drift detection answer different questions. A model can remain confident while the input population becomes detectably different.",
    )


def render_reliability_ranking():
    tables = load_tables()
    ranking = tables.get("model_ranking", pd.DataFrame())
    index = tables.get("reliability_index", pd.DataFrame())
    section(
        "Reliability Ranking",
        "Ranks models using the cross-experiment Reliability Index while preserving component-level evidence.",
    )
    left, right = st.columns([0.48, 0.52])
    with left:
        show_figure("model_ranking.png", "Overall Model Ranking")
    with right:
        show_figure("reliability_index.png", "Component Heatmap")
    display_table(ranking.round(2), height=220)
    csv_download(ranking, "model_ranking.csv", "Download Ranking CSV")
    st.subheader("Component Scores")
    display_table(index.round(2), height=260)
    st.subheader("Radar Chart")
    radar_chart(index)
    card(
        "Overall Interpretation",
        "The most reliable model is the one with the strongest profile across robustness, calibration, distribution shift, confidence alignment, and missing-data behavior.",
    )


def render_interactive_lab():
    tables = load_tables()
    index = tables.get("reliability_index", pd.DataFrame())
    section(
        "Interactive Reliability Lab",
        "A generic simulator for testing how a model responds when inputs become imperfect.",
    )

    left, right = st.columns([0.35, 0.65])
    with left:
        step("Step 1", "Choose Dataset")
        dataset_name = st.selectbox("Dataset", DATASET_OPTIONS, label_visibility="collapsed")
        if dataset_name not in DATASETS:
            st.info("This dataset connector is planned. Select Breast Cancer, Wine, or Iris to run the current lab.")
            return
        X_default, _, target_names = load_dataset(dataset_name)
        model, train_mean, train_std = train_model(dataset_name)
        _initialize_state(X_default, dataset_name)

        step("Step 2", "Load Sample or Upload CSV")
        sample_number = st.number_input("Sample row", 0, len(X_default) - 1, 0)
        if st.button("Load Sample", width="stretch"):
            st.session_state.input_data = X_default.iloc[[sample_number]].copy()
            st.session_state.operations = [f"Loaded sample {sample_number}"]
            st.session_state.analysis_ready = False
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            _load_uploaded(uploaded, X_default)

        step("Step 3", "Apply Data Degradation")
        noise_percent = st.slider("Noise Level", 0, 100, 20)
        noise_level = noise_percent / 100
        col_a, col_b = st.columns(2)
        if col_a.button("Apply Noise", width="stretch"):
            _inject_noise(noise_level, train_std)
            st.session_state.analysis_ready = False
        if col_b.button("Missing Values", width="stretch"):
            _create_missing_data()
            st.session_state.analysis_ready = False
        if col_a.button("Feature Removal", width="stretch"):
            _remove_features(train_mean)
            st.session_state.analysis_ready = False
        if col_b.button("Reset", width="stretch"):
            st.session_state.input_data = X_default.iloc[[sample_number]].copy()
            st.session_state.operations = []
            st.session_state.analysis_ready = False

        step("Step 4", "Run Reliability Analysis")
        run_analysis = st.button("Run Reliability Analysis", width="stretch", type="primary")
        if run_analysis:
            st.session_state.analysis_ready = True

    with right:
        _render_lab_results(
            model,
            target_names,
            train_mean,
            index,
            noise_level,
        )


def render_resources():
    section("Resources", "Download paper, figures, and CSV outputs.")
    if PAPER_PATH.exists():
        _download_path(PAPER_PATH, "Download Research Paper", "application/pdf")
    st.subheader("CSV Outputs")
    for path in csv_outputs():
        _download_path(path, f"Download {path.name}", "text/csv")
    st.subheader("Figures")
    for path in figure_outputs():
        _download_path(path, f"Download {path.name}", "image/png")


def _render_lab_results(model, target_names, train_mean, index, noise_level):
    if not st.session_state.get("analysis_ready", False):
        card(
            "Reliability Analysis Pending",
            "Load or perturb a sample, then run reliability analysis to generate prediction, confidence, risk, and probability outputs.",
        )
        st.subheader("Current Sample")
        display_table(st.session_state.input_data, height=180)
        return

    X_input = st.session_state.input_data.copy()
    missing_ratio = float(X_input.isna().mean().mean())
    X_model = X_input.fillna(train_mean)
    probabilities = model.predict_proba(X_model)
    prediction = model.predict(X_model)[0]
    confidence = float(probabilities.max(axis=1)[0])
    prediction_label = str(target_names[prediction])
    failure_risk = estimate_failure_risk(confidence, noise_level, missing_ratio, len(st.session_state.operations))
    reliability = _model_reliability(index)

    cols = st.columns(4)
    cols[0].metric("Prediction", prediction_label)
    cols[1].metric("Confidence", f"{confidence:.1%}", confidence_meter_label(confidence))
    cols[2].metric("Failure Risk", f"{failure_risk:.1%}")
    cols[3].metric("Reliability Index", "N/A" if np.isnan(reliability) else f"{reliability:.1f}/100", reliability_label(reliability))

    st.subheader("Meters")
    meter("Confidence Meter", confidence)
    meter("Reliability Gauge", 0 if np.isnan(reliability) else reliability / 100)

    st.subheader("Model Used")
    st.write("Logistic Regression with standard scaling")

    st.subheader("Applied Conditions")
    operations = st.session_state.operations or ["Clean input"]
    st.write(", ".join(operations))

    st.subheader("Probability Chart")
    probability_chart(target_names, probabilities[0])

    st.subheader("Current Sample")
    display_table(X_input, height=180)


def _initialize_state(X, dataset_name):
    if st.session_state.get("active_dataset") != dataset_name:
        st.session_state.active_dataset = dataset_name
        st.session_state.input_data = X.head(1).copy()
        st.session_state.operations = []
        st.session_state.analysis_ready = False
    st.session_state.setdefault("input_data", X.head(1).copy())
    st.session_state.setdefault("operations", [])
    st.session_state.setdefault("analysis_ready", False)


def _load_uploaded(uploaded, X_default):
    frame = pd.read_csv(uploaded)
    expected = list(X_default.columns)
    if all(column in frame.columns for column in expected):
        st.session_state.input_data = frame[expected].head(1).copy()
        st.session_state.operations = ["Uploaded CSV sample"]
        st.session_state.analysis_ready = False
    else:
        st.error("Uploaded CSV must include the feature columns for the selected dataset.")


def _inject_noise(noise_level, feature_std):
    rng = np.random.default_rng(42)
    current = st.session_state.input_data.copy()
    noise = rng.normal(0, noise_level, current.shape) * feature_std[current.columns].to_numpy()
    st.session_state.input_data = current + noise
    st.session_state.operations.append(f"Noise {noise_level:.2f}")


def _create_missing_data():
    current = st.session_state.input_data.copy()
    columns = current.columns[: max(1, len(current.columns) // 5)]
    current.loc[:, columns] = np.nan
    st.session_state.input_data = current
    st.session_state.operations.append("Missing values")


def _remove_features(train_mean):
    current = st.session_state.input_data.copy()
    columns = current.columns[: max(1, len(current.columns) // 5)]
    current.loc[:, columns] = train_mean[columns]
    st.session_state.input_data = current
    st.session_state.operations.append("Feature removal")


def _model_reliability(index):
    if index.empty:
        return np.nan
    match = index[index["Model"] == "Logistic Regression"]
    return float(match.iloc[0]["Reliability Index"]) if not match.empty else np.nan


def _download_path(path, label, mime):
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        width="stretch",
    )
