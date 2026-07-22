import streamlit as st

from .components import download_file
from .theme import apply_theme
from .utils import EXPERIMENTS_URL, GITHUB_BASE_URL, NAVIGATION, PAPER_PATH, PAPER_URL, README_URL, display_page_name, strip_page_icon


def render_sidebar():
    st.sidebar.title("When Systems Break")
    st.sidebar.caption("Machine Learning Reliability Laboratory")

    appearance = st.sidebar.segmented_control(
        "Appearance",
        ["Dark", "Light"],
        default="Dark",
    )
    apply_theme(appearance)

    options = []
    for group, pages in NAVIGATION.items():
        st.sidebar.markdown(f"**{group}**")
        options.extend(display_page_name(page) for page in pages)

    selected = st.sidebar.radio(
        "Navigation",
        options,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Resources**")
    download_file(PAPER_PATH, "Download Research Paper", "application/pdf")
    st.sidebar.markdown(f"[Paper]({PAPER_URL})")
    st.sidebar.markdown(f"[README]({README_URL})")
    st.sidebar.markdown(f"[GitHub]({GITHUB_BASE_URL})")
    st.sidebar.markdown(f"[Experiments]({EXPERIMENTS_URL})")

    return strip_page_icon(selected)

