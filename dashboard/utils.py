from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
EXPERIMENTS_DIR = ROOT / "experiments"
PAPER_PATH = ROOT / "paper" / "when-systems-break.pdf"

GITHUB_BASE_URL = "https://github.com/cherikakaushal/when-systems-break"
PAPER_URL = f"{GITHUB_BASE_URL}/blob/main/paper/when-systems-break.pdf"
README_URL = f"{GITHUB_BASE_URL}#readme"
EXPERIMENTS_URL = f"{GITHUB_BASE_URL}/tree/main/experiments"

BADGES = [
    "Noise Robustness",
    "Calibration",
    "Confidence Collapse",
    "Distribution Shift",
    "Reliability Ranking",
    "Refusal Thresholds",
]

NAVIGATION = {
    "Research": [
        "Landing",
        "Overview",
        "Experiments",
        "Dataset Explorer",
        "Interactive Lab",
        "Research Progress",
        "About",
    ],
    "Analysis": [
        "Failure Matrix",
        "Calibration",
        "Confidence Collapse",
        "Distribution Shift",
        "Reliability Ranking",
        "Model Explorer",
    ],
    "Resources": [
        "Downloads",
    ],
}

PAGE_ICONS = {
    "Landing": "H",
    "Overview": "O",
    "Experiments": "E",
    "Dataset Explorer": "D",
    "Interactive Lab": "L",
    "Research Progress": "P",
    "About": "A",
    "Failure Matrix": "M",
    "Calibration": "C",
    "Confidence Collapse": "X",
    "Distribution Shift": "S",
    "Reliability Ranking": "R",
    "Model Explorer": "V",
    "Downloads": "Z",
}


def format_pct(value, digits=1):
    return f"{value * 100:.{digits}f}%"


def display_page_name(label):
    return f"{PAGE_ICONS.get(label, '-') }  {label}"


def strip_page_icon(label):
    return label.split("  ", 1)[-1]
