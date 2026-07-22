import streamlit as st

from .utils import BADGES


def header():
    """Render the shared top hero used across dashboard pages."""
    hero("Research Dashboard", "When Systems Break", "General Framework for Machine Learning Reliability Research", BADGES)


def landing_hero():
    """Render the landing page hero."""
    hero(
        "Machine Learning Reliability Laboratory",
        "When Systems Break",
        "A General Framework for Studying Machine Learning Reliability Under Imperfect Data",
    )


def hero(eyebrow, title, subtitle, badges=None):
    """Render a reusable hero with optional research badges."""
    badge_markup = ""
    if badges:
        badge_markup = (
            '<div class="wsb-badge-row">'
            + "".join(f'<span class="wsb-pill">{badge}</span>' for badge in badges)
            + "</div>"
        )
    st.markdown(
        f"""
        <div class="wsb-hero">
            <div class="wsb-eyebrow">{eyebrow}</div>
            <h1 class="wsb-title-reset">{title}</h1>
            <div class="wsb-subtitle">{subtitle}</div>
            {badge_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, note=None):
    """Render a consistent section heading and optional explanatory note."""
    st.header(title)
    if note:
        st.markdown(f'<div class="wsb-note">{note}</div>', unsafe_allow_html=True)


def card(title, body):
    """Render a themed card."""
    st.markdown(
        f"""
        <div class="wsb-card">
            <div class="wsb-card-title">{title}</div>
            <div class="wsb-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step(label, title):
    """Render a workflow step label and title."""
    st.markdown(
        f'<div class="wsb-step">{label}</div><strong>{title}</strong>',
        unsafe_allow_html=True,
    )


def download_file(path, label, mime):
    """Render a full-width download button if the file exists."""
    if path.exists():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            width="stretch",
        )
