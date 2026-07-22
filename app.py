import streamlit as st

import dashboard.pages as pages
from dashboard.components import header
from dashboard.sidebar import render_sidebar


PAGES = {
    "Landing": pages.render_landing,
    "Overview": pages.render_overview,
    "Experiments": pages.render_experiments,
    "Dataset Explorer": pages.render_dataset_explorer,
    "Interactive Lab": pages.render_interactive_lab,
    "Research Progress": pages.render_research_progress,
    "About": pages.render_about,
    "Failure Matrix": pages.render_failure_matrix,
    "Calibration": pages.render_calibration,
    "Confidence Collapse": pages.render_confidence_collapse,
    "Distribution Shift": pages.render_distribution_shift,
    "Reliability Ranking": pages.render_reliability_ranking,
    "Model Explorer": pages.render_model_explorer,
    "Downloads": pages.render_downloads,
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
    PAGES[page]()


if __name__ == "__main__":
    main()
