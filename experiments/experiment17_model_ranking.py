import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import result_path, save_figure

INPUT_CSV = result_path("reliability_index.csv")
OUTPUT_CSV = result_path("model_ranking.csv")
OUTPUT_FIGURE = "model_ranking.png"

COLORS = ["#176B5B", "#287E9B", "#D79A28", "#B84A5F"]


def load_reliability_index():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} was not found. Run experiment16_reliability_index.py first."
        )

    frame = pd.read_csv(INPUT_CSV)
    required = {"Model", "Reliability Index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{INPUT_CSV.name} is missing columns: {sorted(missing)}")

    return frame.sort_values("Reliability Index", ascending=False).reset_index(drop=True)


def build_ranking(frame):
    leader_score = frame.loc[0, "Reliability Index"]
    ranking = frame[["Model", "Reliability Index"]].copy()
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking["Gap to Leader"] = leader_score - ranking["Reliability Index"]
    ranking["Relative Reliability"] = ranking["Reliability Index"] / leader_score * 100
    ranking["Tier"] = ranking["Gap to Leader"].apply(assign_tier)
    return ranking


def assign_tier(gap):
    if gap <= 0.5:
        return "Leader"
    if gap <= 2.0:
        return "Competitive"
    if gap <= 5.0:
        return "Usable"
    return "Needs Review"


def draw_figure(ranking):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    ordered = ranking.iloc[::-1]
    colors = COLORS[: len(ranking)][::-1]
    bars = ax.barh(
        ordered["Model"],
        ordered["Reliability Index"],
        color=colors,
        height=0.58,
    )

    labels = [
        f"#{rank}  {score:.2f}"
        for rank, score in zip(ordered["Rank"], ordered["Reliability Index"])
    ]
    ax.bar_label(bars, labels=labels, padding=7, fontsize=10, fontweight="bold")

    leader = ranking.iloc[0]
    ax.axvline(
        leader["Reliability Index"],
        color="#222222",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
    )
    ax.text(
        leader["Reliability Index"] - 0.15,
        len(ranking) - 0.35,
        "leader",
        ha="right",
        va="center",
        fontsize=9,
        color="#333333",
    )

    ax.set_xlim(80, 100)
    ax.set_xlabel("Reliability Index (0-100)")
    ax.set_title(
        "Experiment 17: Model Ranking by Reliability Index",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0,
        1.01,
        "Ranking uses the cross-experiment Reliability Index from Experiment 16",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    save_figure(fig, OUTPUT_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    index = load_reliability_index()
    ranking = build_ranking(index)
    ranking.round(4).to_csv(OUTPUT_CSV, index=False)
    draw_figure(ranking)

    print("\nModel Ranking by Reliability Index:\n")
    print(ranking.round(2).to_string(index=False))
    print(f"\nSaved ranking to {OUTPUT_CSV}")
    print(f"Saved figure to figures/{OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
