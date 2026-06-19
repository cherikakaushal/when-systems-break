import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Get coefficients (importance)
importance = model.coef_[0]

# Create dataframe
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
})

# Sort by absolute importance
feature_importance["abs_importance"] = feature_importance["importance"].abs()
feature_importance = feature_importance.sort_values(by="abs_importance", ascending=False)

# Print top 10
print("\nTop 10 Important Features:\n")
print(feature_importance[["feature", "importance"]].head(10))

# Plot top 10
plt.figure(figsize=(8,5))
top_features = feature_importance.head(10)

plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Most Important Features")
plt.gca().invert_yaxis()

# Save image
plt.savefig("experiments/feature_importance.png")
plt.show()
