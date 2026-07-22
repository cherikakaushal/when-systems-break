"""Global theme system and reusable CSS for the research dashboard."""

from string import Template

import streamlit as st


RADIUS = "8px"
MAX_WIDTH = "1400px"
FONT_FAMILY = (
    '"Inter", "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
)

THEMES = {
    "Dark": {
        "background": "#0B0F17",
        "surface": "#101620",
        "card": "#141B26",
        "primary": "#7DD3FC",
        "secondary": "#A78BFA",
        "border": "#2A3445",
        "text": "#F5F7FA",
        "muted_text": "#9AA7B8",
        "success": "#34D399",
        "warning": "#FBBF24",
        "danger": "#FB7185",
        "shadow": "rgba(0, 0, 0, 0.24)",
        "hover": "#1F2A3A",
        "active": "#223047",
        "table_alt": "#111A27",
        "grid": "#2A3445",
    },
    "Light": {
        "background": "#F7F9FC",
        "surface": "#FFFFFF",
        "card": "#FFFFFF",
        "primary": "#0369A1",
        "secondary": "#6D28D9",
        "border": "#D8E0EA",
        "text": "#111827",
        "muted_text": "#526071",
        "success": "#047857",
        "warning": "#B45309",
        "danger": "#BE123C",
        "shadow": "rgba(15, 23, 42, 0.09)",
        "hover": "#EEF4FF",
        "active": "#E0F2FE",
        "table_alt": "#F2F5F9",
        "grid": "#D8E0EA",
    },
}


def get_theme():
    """Return the active theme token dictionary."""
    return THEMES[st.session_state.get("theme_mode", "Dark")]


def set_theme(mode):
    """Store the active theme mode for all dashboard modules."""
    st.session_state.theme_mode = mode


def apply_theme(mode):
    """Apply global CSS generated from the active theme tokens."""
    set_theme(mode)
    theme = THEMES[mode]
    css = Template(
        """
        <style>
        :root {
            --wsb-background: $background;
            --wsb-surface: $surface;
            --wsb-card: $card;
            --wsb-primary: $primary;
            --wsb-secondary: $secondary;
            --wsb-border: $border;
            --wsb-text: $text;
            --wsb-muted-text: $muted_text;
            --wsb-success: $success;
            --wsb-warning: $warning;
            --wsb-danger: $danger;
            --wsb-shadow: $shadow;
            --wsb-hover: $hover;
            --wsb-active: $active;
            --wsb-table-alt: $table_alt;
            --wsb-grid: $grid;
            --wsb-radius: $radius;
            --wsb-max-width: $max_width;
            --wsb-font: $font_family;
        }

        html, body, .stApp {
            background: var(--wsb-background);
            color: var(--wsb-text);
            font-family: var(--wsb-font);
            line-height: 1.55;
        }

        .block-container {
            max-width: var(--wsb-max-width);
            padding: 1.25rem 2rem 2.75rem;
        }

        h1, h2, h3, h4, h5, h6, p, li, label, span {
            color: var(--wsb-text);
            font-family: var(--wsb-font);
            letter-spacing: 0;
        }

        h1 { font-size: clamp(2.2rem, 4vw, 4.2rem); line-height: 1.04; }
        h2 { font-size: clamp(1.55rem, 2.4vw, 2.3rem); line-height: 1.16; }
        h3 { font-size: clamp(1.15rem, 1.6vw, 1.45rem); line-height: 1.22; }

        a {
            color: var(--wsb-primary);
            text-decoration-thickness: 1px;
            text-underline-offset: 3px;
        }

        a:hover { color: var(--wsb-secondary); }

        [data-testid="stSidebar"] {
            background: var(--wsb-surface);
            border-right: 1px solid var(--wsb-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--wsb-text) !important;
            font-family: var(--wsb-font);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            min-height: 38px;
            border-radius: var(--wsb-radius);
            padding: 0.2rem 0.35rem;
            transition: background 140ms ease, box-shadow 140ms ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: var(--wsb-hover);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: var(--wsb-active);
            box-shadow: inset 3px 0 0 var(--wsb-primary);
        }

        [data-testid="stSidebar"] a {
            color: var(--wsb-primary) !important;
            font-weight: 650;
            text-decoration: none;
        }

        [data-testid="stSidebar"] a:hover {
            color: var(--wsb-secondary) !important;
            text-decoration: underline;
        }

        div[data-testid="stMetric"] {
            min-height: 122px;
            background: var(--wsb-card);
            border: 1px solid var(--wsb-border);
            border-radius: var(--wsb-radius);
            padding: 16px;
            box-shadow: 0 12px 28px var(--wsb-shadow);
        }

        div[data-testid="stMetric"] label {
            color: var(--wsb-muted-text) !important;
            min-height: 2.15rem;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--wsb-text);
            font-size: 1.58rem;
            line-height: 1.15;
            white-space: normal;
        }

        div[data-testid="stMetricDelta"] {
            color: var(--wsb-muted-text) !important;
        }

        div[data-testid="stMetric"]::before {
            background: var(--wsb-primary);
            border-radius: 999px;
            content: "";
            display: block;
            height: 8px;
            margin-bottom: 10px;
            width: 34px;
        }

        .wsb-hero {
            background: linear-gradient(135deg, var(--wsb-card), var(--wsb-surface));
            border: 1px solid var(--wsb-border);
            border-radius: var(--wsb-radius);
            box-shadow: 0 16px 42px var(--wsb-shadow);
            padding: clamp(20px, 3vw, 34px);
            margin-bottom: 24px;
        }

        .wsb-eyebrow {
            color: var(--wsb-primary);
            font-size: 0.78rem;
            font-weight: 780;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .wsb-title-reset { margin: 0; }

        .wsb-subtitle {
            color: var(--wsb-muted-text);
            font-size: 1.02rem;
            max-width: 900px;
            margin-top: 0.5rem;
        }

        .wsb-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 18px;
        }

        .wsb-pill {
            background: var(--wsb-surface);
            border: 1px solid var(--wsb-border);
            border-radius: 999px;
            color: var(--wsb-text);
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0.18rem 0.68rem;
            font-size: 0.84rem;
            font-weight: 650;
        }

        .wsb-card {
            background: var(--wsb-card);
            border: 1px solid var(--wsb-border);
            border-radius: var(--wsb-radius);
            box-shadow: 0 12px 28px var(--wsb-shadow);
            min-height: 100%;
            padding: 16px;
            transition: border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
        }

        .wsb-card:hover {
            border-color: var(--wsb-primary);
            transform: translateY(-1px);
        }

        .wsb-card-title {
            color: var(--wsb-text);
            font-weight: 760;
            margin-bottom: 0.42rem;
        }

        .wsb-card-body,
        .wsb-note,
        .wsb-muted {
            color: var(--wsb-muted-text);
        }

        .wsb-note {
            font-size: 0.98rem;
            margin-top: -0.25rem;
            margin-bottom: 1rem;
        }

        .wsb-step {
            color: var(--wsb-primary);
            font-size: 0.82rem;
            font-weight: 780;
            margin-bottom: 0.15rem;
        }

        .wsb-timeline {
            align-items: center;
            background: var(--wsb-card);
            border: 1px solid var(--wsb-border);
            border-radius: var(--wsb-radius);
            box-shadow: 0 12px 28px var(--wsb-shadow);
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 16px;
        }

        .wsb-timeline-item {
            background: var(--wsb-surface);
            border: 1px solid var(--wsb-border);
            border-radius: 999px;
            color: var(--wsb-text);
            font-weight: 700;
            padding: 0.32rem 0.68rem;
        }

        .wsb-timeline-arrow {
            color: var(--wsb-muted-text);
            font-weight: 800;
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stLinkButton"] a {
            align-items: center;
            background: var(--wsb-card);
            border: 1px solid var(--wsb-border);
            border-radius: var(--wsb-radius);
            color: var(--wsb-text);
            display: inline-flex;
            font-weight: 720;
            justify-content: center;
            min-height: 42px;
            transition: background 140ms ease, border-color 140ms ease, color 140ms ease, transform 140ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stLinkButton"] a:hover {
            background: var(--wsb-hover);
            border-color: var(--wsb-primary);
            color: var(--wsb-primary);
            transform: translateY(-1px);
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus,
        [data-testid="stLinkButton"] a:focus {
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--wsb-primary) 28%, transparent);
            outline: none;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: var(--wsb-card);
            border: 1px solid var(--wsb-border);
            border-radius: var(--wsb-radius);
            overflow: hidden;
        }

        [data-testid="stDataFrame"] div[role="columnheader"],
        [data-testid="stTable"] thead tr {
            background: var(--wsb-surface);
            position: sticky;
            top: 0;
            z-index: 1;
        }

        [data-testid="stTable"] tbody tr:nth-child(even) {
            background: var(--wsb-table-alt);
        }

        [data-testid="stFileUploader"] section {
            background: var(--wsb-card);
            border-color: var(--wsb-border);
            border-radius: var(--wsb-radius);
        }

        [data-testid="stProgress"] > div > div {
            background-color: var(--wsb-primary);
        }

        [data-testid="stAlert"] {
            background: var(--wsb-card);
            color: var(--wsb-text);
            border-color: var(--wsb-border);
        }

        hr {
            border-color: var(--wsb-border);
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """
    ).substitute({**theme, "radius": RADIUS, "max_width": MAX_WIDTH, "font_family": FONT_FAMILY})
    st.markdown(css, unsafe_allow_html=True)
