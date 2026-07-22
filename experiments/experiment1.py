import numpy as np
import pandas as pd
from paths import save_figure
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Baseline accuracy
y_pred = model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

# Add noise
noise = np.random.normal(0, 0.5, X_test.shape)
X_test_noisy = X_test + noise

# Accuracy after noise
y_pred_noisy = model.predict(X_test_noisy)
noisy_acc = accuracy_score(y_test, y_pred_noisy)

print("Baseline Accuracy:", baseline_acc)
print("Noisy Accuracy:", noisy_acc)
print("Drop:", baseline_acc - noisy_acc)

import matplotlib.pyplot as plt

labels = ["Baseline", "Noisy"]
values = [baseline_acc, noisy_acc]

plt.figure()
plt.bar(labels, values)
plt.title("Effect of Noise on Model Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
save_figure(plt.gcf(), "result.png")
plt.show()
