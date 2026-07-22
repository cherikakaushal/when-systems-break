from string import Template

import streamlit as st


PALETTES = {
    "Dark": {
        "app_bg": "#0B0F17",
        "sidebar_bg": "#101620",
        "panel_bg": "#141B26",
        "panel_soft": "#182131",
        "panel_border": "#2A3445",
        "text": "#F5F7FA",
        "muted": "#9AA7B8",
        "subtle": "#C8D1DC",
        "hover": "#1F2A3A",
        "active": "#233246",
        "link": "#8BD3FF",
        "link_hover": "#B9E6FF",
        "accent": "#7DD3FC",
        "accent_soft": "#123044",
        "shadow": "rgba(0, 0, 0, 0.22)",
    },
    "Light": {
        "app_bg": "#F7F9FC",
        "sidebar_bg": "#FFFFFF",
        "panel_bg": "#FFFFFF",
        "panel_soft": "#F2F5F9",
        "panel_border": "#DCE3EC",
        "text": "#111827",
        "muted": "#526071",
        "subtle": "#334155",
        "hover": "#EEF4FF",
        "active": "#E6F4FF",
        "link": "#075985",
        "link_hover": "#0C4A6E",
        "accent": "#0369A1",
        "accent_soft": "#E0F2FE",
        "shadow": "rgba(15, 23, 42, 0.08)",
    },
}


def apply_theme(mode):
    colors = PALETTES[mode]
    css = Template(
        """
        <style>
        :root {
            --wsb-bg: $app_bg;
            --wsb-sidebar: $sidebar_bg;
            --wsb-panel: $panel_bg;
            --wsb-panel-soft: $panel_soft;
            --wsb-border: $panel_border;
            --wsb-text: $text;
            --wsb-muted: $muted;
            --wsb-subtle: $subtle;
            --wsb-hover: $hover;
            --wsb-active: $active;
            --wsb-link: $link;
            --wsb-link-hover: $link_hover;
            --wsb-accent: $accent;
            --wsb-accent-soft: $accent_soft;
            --wsb-shadow: $shadow;
        }
        .stApp {
            background: var(--wsb-bg);
            color: var(--wsb-text);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: var(--wsb-sidebar);
            border-right: 1px solid var(--wsb-border);
        }
        [data-testid="stSidebar"] * {
            color: var(--wsb-text) !important;
        }
        [data-testid="stSidebar"] a {
            color: var(--wsb-link) !important;
            font-weight: 650;
            text-decoration: none;
        }
        [data-testid="stSidebar"] a:hover {
            color: var(--wsb-link-hover) !important;
            text-decoration: underline;
        }
        h1, h2, h3 {
            color: var(--wsb-text);
            letter-spacing: 0;
        }
        p, li, label, span {
            color: var(--wsb-text);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.12rem 0.25rem;
            transition: background 120ms ease, border-color 120ms ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: var(--wsb-hover);
        }
        div[data-testid="stMetric"] {
            min-height: 118px;
            background: var(--wsb-panel);
            border: 1px solid var(--wsb-border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 10px 24px var(--wsb-shadow);
        }
        div[data-testid="stMetric"] label {
            color: var(--wsb-subtle) !important;
            min-height: 2.2rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--wsb-text);
            font-size: 1.65rem;
            white-space: normal;
            line-height: 1.15;
        }
        .wsb-hero {
            border: 1px solid var(--wsb-border);
            background: linear-gradient(135deg, var(--wsb-panel), var(--wsb-panel-soft));
            border-radius: 8px;
            padding: 22px 24px;
            margin-bottom: 20px;
            box-shadow: 0 14px 36px var(--wsb-shadow);
        }
        .wsb-eyebrow {
            color: var(--wsb-accent);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }
        .wsb-subtitle {
            color: var(--wsb-muted);
            font-size: 1rem;
            margin-top: 0.35rem;
            max-width: 860px;
        }
        .wsb-card {
            background: var(--wsb-panel);
            border: 1px solid var(--wsb-border);
            border-radius: 8px;
            padding: 16px;
            min-height: 100%;
            box-shadow: 0 10px 24px var(--wsb-shadow);
            transition: border-color 120ms ease, transform 120ms ease;
        }
        .wsb-card:hover {
            border-color: var(--wsb-accent);
        }
        .wsb-note {
            color: var(--wsb-muted);
            font-size: 0.98rem;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
        }
        .wsb-pill {
            display: inline-block;
            border: 1px solid var(--wsb-border);
            border-radius: 999px;
            padding: 0.22rem 0.64rem;
            margin-right: 0.35rem;
            margin-bottom: 0.42rem;
            color: var(--wsb-text);
            background: var(--wsb-panel);
            font-size: 0.84rem;
            font-weight: 620;
        }
        .wsb-step {
            color: var(--wsb-accent);
            font-weight: 750;
            font-size: 0.82rem;
            margin-bottom: 0.15rem;
        }
        .wsb-muted {
            color: var(--wsb-muted);
        }
        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid var(--wsb-border);
            background: var(--wsb-panel);
            color: var(--wsb-text);
            border-radius: 8px;
            transition: border-color 120ms ease, color 120ms ease, background 120ms ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--wsb-accent);
            color: var(--wsb-link-hover);
            background: var(--wsb-hover);
        }
        a {
            color: var(--wsb-link);
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--wsb-border);
            border-radius: 8px;
            overflow: hidden;
        }
        [data-testid="stFileUploader"] section {
            background: var(--wsb-panel);
            border-color: var(--wsb-border);
            border-radius: 8px;
        }
        [data-testid="stAlert"] {
            color: var(--wsb-text);
        }
        </style>
        """
    ).substitute(colors)
    st.markdown(css, unsafe_allow_html=True)

