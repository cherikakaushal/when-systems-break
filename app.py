import streamlit as st

import dashboard.pages as pages
from dashboard.components import header
from dashboard.sidebar import render_sidebar


PAGES = {
    "Landing": "render_landing",
    "Overview": "render_overview",
    "Experiments": "render_experiments",
    "Dataset Explorer": "render_dataset_explorer",
    "Interactive Lab": "render_interactive_lab",
    "Research Progress": "render_research_progress",
    "About": "render_about",
    "Failure Matrix": "render_failure_matrix",
    "Calibration": "render_calibration",
    "Confidence Collapse": "render_confidence_collapse",
    "Distribution Shift": "render_distribution_shift",
    "Reliability Ranking": "render_reliability_ranking",
    "Model Explorer": "render_model_explorer",
    "Downloads": "render_downloads",
}


def main():
    st.set_page_config(
        page_title="When Systems Break",
        page_icon="WSB",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    page = render_sidebar()
    if page != "Landing":
        header()
    renderer = getattr(pages, PAGES.get(page, "render_overview"), pages.render_overview)
    renderer()


if __name__ == "__main__":
    main()
