import subprocess
import zipfile
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from .charts import bar_chart, correlation_heatmap, meter, probability_chart, radar_chart, show_figure
from .components import card, landing_hero, section, step
from .loader import DATASET_OPTIONS, DATASETS, EXPERIMENT_DETAILS, csv_outputs, experiment_catalog, figure_outputs, load_dataset, load_tables, train_model
from .metrics import confidence_meter_label, estimate_failure_risk, latest_findings, reliability_label, repository_stats
from .tables import csv_download, display_table
from .utils import GITHUB_BASE_URL, PAPER_PATH, PAPER_URL, README_URL, ROOT, format_pct


RESEARCH_QUESTIONS = [
    "How does increasing noise affect model accuracy?",
    "Which features contribute most to model performance?",
    "Do different algorithms fail differently?",
    "Can confidence scores indicate failure before accuracy drops?",
    "Is missing information more harmful than noisy information?",
    "When should a model refuse to make a prediction?",
    "Can multiple reliability signals be combined without hiding their tradeoffs?",
    "Does 90% confidence actually correspond to 90% correctness?",
    "Can a system detect when the deployment distribution has changed?",
    "Which model is most reliable when all dimensions are ranked together?",
]


TIMELINE = [
    "Dataset",
    "Training",
    "Noise",
    "Missing Data",
    "Calibration",
    "Distribution Shift",
    "Reliability Index",
    "Ranking",
    "Research Report",
]


def render_landing():
    tables = load_tables()
    catalog = experiment_catalog()
    ranking = tables.get("model_ranking", pd.DataFrame())
    score = tables.get("reliability_scores", pd.DataFrame())

    landing_hero()
    c1, c2, c3 = st.columns(3)
    c1.link_button("Open Dashboard", "#overview", width="stretch")
    c2.link_button("Read Research Paper", PAPER_URL, width="stretch")
    c3.link_button("View GitHub", GITHUB_BASE_URL, width="stretch")

    st.subheader("Research Summary")
    cols = st.columns(3)
    with cols[0]:
        card("What is ML reliability?", "It measures whether a model keeps behaving well when real-world data becomes noisy, incomplete, shifted, or uncertain.")
    with cols[1]:
        card("Why does it matter?", "Production models are rarely tested only on clean textbook data. Reliability shows where confidence and accuracy begin to separate.")
    with cols[2]:
        card("What does this solve?", "The platform turns failure testing, calibration, refusal, shift detection, and ranking into a repeatable research workflow.")

    st.subheader("Research Questions")
    rq_cols = st.columns(2)
    for idx, question in enumerate(RESEARCH_QUESTIONS, 1):
        with rq_cols[(idx - 1) % 2]:
            card(f"RQ{idx}", question)

    st.subheader("Methodology")
    timeline_markup = '<div class="wsb-timeline">'
    for idx, item in enumerate(TIMELINE):
        timeline_markup += f'<span class="wsb-timeline-item">{item}</span>'
        if idx < len(TIMELINE) - 1:
            timeline_markup += '<span class="wsb-timeline-arrow">&rarr;</span>'
    timeline_markup += "</div>"
    st.markdown(timeline_markup, unsafe_allow_html=True)

    st.subheader("Featured Results")
    leader = ranking.iloc[0] if not ranking.empty else None
    top_score = score.iloc[0] if not score.empty else None
    metrics = st.columns(4)
    metrics[0].metric("Top Model", leader["Model"] if leader is not None else "N/A")
    metrics[1].metric("Reliability Score", f"{top_score['Reliability Score']:.2f}/100" if top_score is not None else "N/A")
    metrics[2].metric("Experiments", len(catalog))
    metrics[3].metric("Datasets", "3 active")

    st.divider()
    f1, f2, f3 = st.columns(3)
    f1.markdown(f"[Paper]({PAPER_URL})")
    f2.markdown(f"[GitHub]({GITHUB_BASE_URL})")
    f3.markdown("[Portfolio](https://github.com/cherikakaushal)")


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
    commit, commit_date = _git_metadata()
    stats["Last Commit"] = commit
    stats["Research Paper"] = "v1.0"
    stat_cols = st.columns(3)
    for col, (label, value) in zip(stat_cols, stats.items()):
        col.metric(label, value)
    more_cols = st.columns(3)
    more_cols[0].metric("Last Commit Date", commit_date)
    more_cols[1].metric("Version", "1.0.0")
    more_cols[2].metric("Repository", "GitHub connected")


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
    method, finding = EXPERIMENT_DETAILS.get(
        selected["number"],
        ("Run the experiment script and inspect generated artifacts.", "Generated outputs document the observed behavior."),
    )
    st.subheader(f"Experiment {selected['number']:02d}: {selected['title']}")
    detail_cols = st.columns(2)
    with detail_cols[0]:
        card("Purpose", selected["purpose"])
        card("Method", method)
    with detail_cols[1]:
        card("Interpretation", "Use the figure and CSV output together; the image communicates the trend while the CSV preserves the measured values.")
        card("Key Finding", finding)
    if selected["script"]:
        st.caption(f"Script: {selected['script'].relative_to(selected['script'].parent.parent)}")

    st.subheader("Figure")
    if selected["figures"]:
        cols = st.columns(min(2, len(selected["figures"])))
        for index, path in enumerate(selected["figures"]):
            with cols[index % len(cols)]:
                st.image(str(path), caption=path.name, width="stretch")
                _download_path(path, "Download Figure", "image/png")
    else:
        st.info("No figure output detected for this experiment.")

    st.subheader("CSV")
    if selected["csv"]:
        for path in selected["csv"]:
            frame = pd.read_csv(path)
            st.markdown(f"**{path.name}**")
            display_table(frame.head(20), height=260)
            _download_path(path, f"Download {path.name}", "text/csv")
    else:
        st.info("No CSV output detected for this experiment.")

    st.subheader("Code Used")
    if selected["script"]:
        with st.expander("Show experiment source"):
            st.code(selected["script"].read_text(encoding="utf-8"), language="python")


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
        bar_chart(confidence, "Noise Level", "Mean Prediction Confidence", "Confidence Across Noise Levels")
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


def render_dataset_explorer():
    section(
        "Dataset Explorer",
        "Inspect supported datasets before running reliability simulations.",
    )
    dataset_name = st.selectbox("Dataset", list(DATASETS), key="dataset_explorer")
    X, y, target_names = load_dataset(dataset_name)
    cols = st.columns(4)
    cols[0].metric("Samples", len(X))
    cols[1].metric("Features", X.shape[1])
    cols[2].metric("Classes", len(target_names))
    cols[3].metric("Missing Values", int(X.isna().sum().sum()))

    st.subheader("Class Distribution")
    class_counts = pd.Series(y).map(lambda idx: target_names[idx]).value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    bar_chart(class_counts, "Class", "Count", "Class Distribution")

    st.subheader("Feature Statistics")
    display_table(X.describe().T.reset_index().rename(columns={"index": "Feature"}).round(3), height=360)

    st.subheader("Correlation Matrix")
    correlation_heatmap(X)


def render_model_explorer():
    tables = load_tables()
    index = tables.get("reliability_index", pd.DataFrame())
    ranking = tables.get("model_ranking", pd.DataFrame())
    stats = tables.get("model_statistics", pd.DataFrame())
    calibration = tables.get("calibration_metrics", pd.DataFrame())
    shift = tables.get("shift_statistics", pd.DataFrame())
    failure = tables.get("failure_matrix", pd.DataFrame())

    section(
        "Model Explorer",
        "Click a model to compare accuracy, robustness, calibration, confidence, distribution shift, and final ranking.",
    )
    models = list(index["Model"]) if not index.empty else ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
    model_name = st.selectbox("Model", models)

    row = index[index["Model"] == model_name].iloc[0] if not index.empty else pd.Series(dtype=float)
    rank_row = ranking[ranking["Model"] == model_name].iloc[0] if not ranking.empty else pd.Series(dtype=float)
    stat_row = stats[stats["Model"] == model_name].iloc[0] if not stats.empty else pd.Series(dtype=float)
    cal_rows = calibration[calibration["Model"] == model_name] if not calibration.empty else pd.DataFrame()
    shift_rows = shift[shift["Model"] == model_name] if not shift.empty else pd.DataFrame()

    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{row.get('Accuracy Score', np.nan):.2f}")
    cols[1].metric("Noise", f"{row.get('Noise Robustness Score', np.nan):.2f}")
    cols[2].metric("Calibration", f"{row.get('Calibration Score', np.nan):.2f}")
    cols[3].metric("Confidence", f"{row.get('Confidence Score', np.nan):.2f}")
    cols[4].metric("Shift", f"{row.get('Distribution Shift Score', np.nan):.2f}")
    cols[5].metric("Rank", f"#{int(rank_row.get('Rank', 0))}" if not rank_row.empty else "N/A")

    st.subheader("Model Reliability Components")
    if not row.empty:
        component_frame = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Noise Robustness",
                    "Missing Data",
                    "Calibration",
                    "Distribution Shift",
                    "Confidence",
                ],
                "Score": [
                    row.get("Accuracy Score"),
                    row.get("Noise Robustness Score"),
                    row.get("Missing Data Score"),
                    row.get("Calibration Score"),
                    row.get("Distribution Shift Score"),
                    row.get("Confidence Score"),
                ],
            }
        )
        bar_chart(component_frame, "Metric", "Score", "Reliability Component Scores")
        display_table(component_frame.round(2), height=250)

    st.subheader("Supporting Evidence")
    evidence_cols = st.columns(2)
    with evidence_cols[0]:
        st.markdown("**Multi-run statistics**")
        display_table(pd.DataFrame([stat_row]).round(4), height=180)
        st.markdown("**Calibration metrics**")
        display_table(cal_rows.round(4), height=220)
    with evidence_cols[1]:
        st.markdown("**Failure matrix row**")
        display_table(failure[failure["Model"] == model_name].round(2), height=180)
        st.markdown("**Distribution shift endpoint**")
        if not shift_rows.empty:
            endpoint = shift_rows[shift_rows["Mean Shift"] == shift_rows["Mean Shift"].max()]
            display_table(endpoint.round(4), height=220)


def render_research_progress():
    section(
        "Research Progress",
        "A timeline of the project evolution from baseline experiments to paper and dashboard.",
    )
    catalog = experiment_catalog()
    rows = [
        {"Milestone": f"Experiment {item['number']}", "Title": item["title"], "Status": "Complete"}
        for item in catalog
    ]
    rows.extend(
        [
            {"Milestone": "Research Paper", "Title": "when-systems-break.pdf", "Status": "Complete"},
            {"Milestone": "Dashboard", "Title": "Research platform UI", "Status": "Complete"},
            {"Milestone": "Citation", "Title": "CITATION.cff", "Status": "Complete"},
        ]
    )
    display_table(pd.DataFrame(rows), height=520)


def render_about():
    section(
        "About the Research",
        "Motivation, questions, methodology, limitations, future work, and references in one readable place.",
    )
    card("Motivation", "Most ML projects optimize clean-data accuracy. This project asks what happens when deployment data becomes imperfect.")
    st.subheader("Research Questions")
    for idx, question in enumerate(RESEARCH_QUESTIONS, 1):
        st.write(f"RQ{idx}: {question}")
    st.subheader("Methodology")
    st.write("The project uses controlled degradation experiments across noise, missingness, feature removal, calibration, confidence, refusal thresholds, and distribution shift.")
    st.subheader("Limitations")
    st.write("Experiments use controlled tabular datasets and simplified degradation mechanisms. Results should be interpreted as a research benchmark, not a production safety certification.")
    st.subheader("Future Work")
    st.write("Future work includes SHAP explanations, semantic noise, adaptive calibration, external datasets, and human-in-the-loop evaluation.")
    st.subheader("References")
    st.write("Scikit-learn, NumPy, Pandas, Matplotlib, selective classification, and calibration literature support the implementation and framing.")


def render_downloads():
    section("Downloads", "Export paper, all figures, CSV results, and generated research summaries.")
    if PAPER_PATH.exists():
        _download_path(PAPER_PATH, "Download Research Paper", "application/pdf")
    st.subheader("Download Bundles")
    st.download_button("All Figures (ZIP)", _zip_paths(figure_outputs()), "figures.zip", "application/zip", width="stretch")
    st.download_button("CSV Results (ZIP)", _zip_paths(csv_outputs()), "results.zip", "application/zip", width="stretch")
    st.download_button("Experiment Results (ZIP)", _zip_paths([*csv_outputs(), *figure_outputs()]), "experiment-results.zip", "application/zip", width="stretch")
    st.download_button("Generated Report (PDF)", _generated_report_pdf(), "generated-research-report.pdf", "application/pdf", width="stretch")
    st.download_button("Research Report Summary", _generated_report().encode("utf-8"), "generated-research-summary.md", "text/markdown", width="stretch")
    st.download_button("Experiment Logs", _experiment_log().encode("utf-8"), "experiment-log.txt", "text/plain", width="stretch")

    st.subheader("Dataset Export")
    dataset_name = st.selectbox("Dataset to export", list(DATASETS), key="download_dataset")
    X, y, _ = load_dataset(dataset_name)
    dataset = X.copy()
    dataset["target"] = y
    st.download_button(
        "Complete Dataset CSV",
        dataset.to_csv(index=False).encode("utf-8"),
        f"{dataset_name.lower().replace(' ', '-')}.csv",
        "text/csv",
        width="stretch",
    )

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


def _zip_paths(paths):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    buffer.seek(0)
    return buffer.getvalue()


def _generated_report():
    tables = load_tables()
    ranking = tables.get("model_ranking", pd.DataFrame())
    index = tables.get("reliability_index", pd.DataFrame())
    leader = ranking.iloc[0] if not ranking.empty else None
    lines = [
        "# When Systems Break: Generated Research Summary",
        "",
        "## Summary",
        "This report summarizes the current machine learning reliability benchmark outputs.",
        "",
        "## Headline Result",
        f"Top ranked model: {leader['Model'] if leader is not None else 'N/A'}",
        f"Reliability Index: {leader['Reliability Index']:.2f}" if leader is not None else "Reliability Index: N/A",
        "",
        "## Metrics",
    ]
    if not index.empty:
        lines.append(index.round(2).to_string(index=False))
    lines.extend(
        [
            "",
            "## Conclusion",
            "Reliability is multi-dimensional. Models should be compared across accuracy, robustness, calibration, confidence behavior, distribution shift, and ranking evidence.",
        ]
    )
    return "\n".join(lines)


def _generated_report_pdf():
    report = _generated_report().splitlines()
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.94
        for line in report:
            if y < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.94
            size = 15 if line.startswith("# ") else 11
            weight = "bold" if line.startswith("#") else "normal"
            ax.text(0.08, y, line.replace("#", "").strip(), ha="left", va="top", fontsize=size, fontweight=weight)
            y -= 0.035 if line else 0.022
        pdf.savefig(fig)
        plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _experiment_log():
    catalog = experiment_catalog()
    lines = ["When Systems Break Experiment Log", ""]
    for item in catalog:
        lines.append(f"Experiment {item['number']:02d}: {item['title']}")
        lines.append(f"Purpose: {item['purpose']}")
        lines.append(f"Figures: {', '.join(path.name for path in item['figures']) or 'none'}")
        lines.append(f"CSVs: {', '.join(path.name for path in item['csv']) or 'none'}")
        lines.append("")
    return "\n".join(lines)


def _git_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        date = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit, date
    except Exception:
        return "N/A", "N/A"
