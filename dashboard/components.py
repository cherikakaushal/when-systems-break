import streamlit as st

from .utils import BADGES


def header():
    st.markdown(
        """
        <div class="wsb-hero">
            <div class="wsb-eyebrow">Research Dashboard</div>
            <h1 style="margin: 0;">When Systems Break</h1>
            <div class="wsb-subtitle">
                General Framework for Machine Learning Reliability Research
            </div>
            <div style="margin-top: 16px;">
        """
        + "".join(f'<span class="wsb-pill">{badge}</span>' for badge in BADGES)
        + """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def landing_hero():
    st.markdown(
        """
        <div class="wsb-hero">
            <div class="wsb-eyebrow">Machine Learning Reliability Laboratory</div>
            <h1 style="margin: 0;">When Systems Break</h1>
            <div class="wsb-subtitle">
                A General Framework for Studying Machine Learning Reliability Under Imperfect Data
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, note=None):
    st.header(title)
    if note:
        st.markdown(f'<div class="wsb-note">{note}</div>', unsafe_allow_html=True)


def card(title, body):
    st.markdown(
        f"""
        <div class="wsb-card">
            <strong>{title}</strong>
            <p class="wsb-muted" style="margin-bottom: 0;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step(label, title):
    st.markdown(
        f'<div class="wsb-step">{label}</div><strong>{title}</strong>',
        unsafe_allow_html=True,
    )


def download_file(path, label, mime):
    if path.exists():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            width="stretch",
        )
