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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train baseline model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

baseline_acc = accuracy_score(y_test, model.predict(X_test))

# Get feature importance (coefficients)
importance = pd.Series(model.coef_[0], index=X.columns)
importance = importance.abs().sort_values(ascending=False)

print("\nTop Features:\n", importance.head(5))

# Remove top 3 important features
top_features = importance.head(3).index

X_train_reduced = X_train.drop(columns=top_features)
X_test_reduced = X_test.drop(columns=top_features)

# Retrain model
model_reduced = LogisticRegression(max_iter=5000)
model_reduced.fit(X_train_reduced, y_train)

reduced_acc = accuracy_score(y_test, model_reduced.predict(X_test_reduced))

print("\nBaseline Accuracy:", baseline_acc)
print("After Removing Top Features:", reduced_acc)
print("Drop:", baseline_acc - reduced_acc)