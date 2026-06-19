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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train baseline
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, model.predict(X_test))

noise_levels = np.linspace(0, 0.5, 6)  # 0% → 50%
accuracies = []

for noise in noise_levels:
    X_noisy = X_test.copy()

    noise_matrix = np.random.normal(0, noise, X_noisy.shape)
    X_noisy = X_noisy + noise_matrix

    acc = accuracy_score(y_test, model.predict(X_noisy))
    accuracies.append(acc)

    print(f"Noise Level: {noise:.2f} → Accuracy: {acc:.4f}")

# Plot
plt.figure()
plt.plot(noise_levels, accuracies, marker='o')
plt.title("Model Robustness: Accuracy vs Noise")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.grid()

plt.savefig("experiments/noise_curve.png")
plt.show()
