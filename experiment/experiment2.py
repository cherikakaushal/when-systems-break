import numpy as np
import pandas as pd
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

# Train baseline model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

baseline = accuracy_score(y_test, model.predict(X_test))

# Introduce missing data (10%)
X_missing = X_test.copy()
mask = np.random.rand(*X_missing.shape) < 0.1
X_missing[mask] = np.nan

# Fill missing values (simple strategy)
X_filled = X_missing.fillna(X_missing.mean())

# Test again
missing_acc = accuracy_score(y_test, model.predict(X_filled))

print("Baseline Accuracy:", baseline)
print("With Missing Data:", missing_acc)
print("Drop:", baseline - missing_acc)