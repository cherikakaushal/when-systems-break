import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train baseline model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Noise levels
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []

print("\nThreshold Analysis:\n")

for noise in noise_levels:

    # Add Gaussian noise
    noise_matrix = np.random.normal(0, noise, X_test.shape)
    X_noisy = X_test + noise_matrix

    # Predict
    predictions = model.predict(X_noisy)

    # Accuracy
    acc = accuracy_score(y_test, predictions)
    accuracies.append(acc)

    print(f"Noise Level: {noise:.1f} → Accuracy: {acc:.4f}")

# Plot
plt.figure(figsize=(8,5))

plt.plot(
    noise_levels,
    accuracies,
    marker='o',
    linewidth=2
)

plt.title("System Reliability Threshold")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig("experiment/threshold_analysis.png")
plt.show()