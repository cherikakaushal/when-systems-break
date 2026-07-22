import matplotlib.pyplot as plt
from paths import save_figure

conditions = [
    "Clean",
    "Noise",
    "Missing",
    "Feature Removal",
    "Combined"
]

accuracies = [
    0.95,
    0.89,
    0.83,
    0.76,
    0.68
]

plt.figure(figsize=(8,5))

plt.plot(
    conditions,
    accuracies,
    marker='o',
    linewidth=2
)

plt.title("Failure Comparison Across Conditions")
plt.xlabel("Condition")
plt.ylabel("Accuracy")

plt.grid(True)

save_figure(plt.gcf(), "failure_comparison.png")
plt.show()
