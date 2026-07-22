from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
PAPER_FIGURES_DIR = ROOT_DIR / "paper" / "figures"


def figure_path(filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


def result_path(filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename


def paper_figure_path(filename):
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return PAPER_FIGURES_DIR / filename


def save_figure(fig, filename, include_paper=True, **savefig_kwargs):
    figure = figure_path(filename)
    fig.savefig(figure, **savefig_kwargs)
    if include_paper:
        fig.savefig(paper_figure_path(filename), **savefig_kwargs)
    return figure
