import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Random Forest": RandomForestClassifier()
}

results = []

# Function to add noise
def add_noise(X, noise_level=0.2):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Function to add missing data
def add_missing(X, missing_ratio=0.1):
    X_missing = X.copy()
    mask = np.random.rand(*X.shape) < missing_ratio
    X_missing[mask] = np.nan
    return X_missing.fillna(X.mean())

# Evaluate models
for name, model in models.items():
    
    # Train on clean data
    model.fit(X_train, y_train)
    clean_acc = accuracy_score(y_test, model.predict(X_test))
    
    # Noisy data
    X_test_noisy = add_noise(X_test)
    noisy_acc = accuracy_score(y_test, model.predict(X_test_noisy))
    
    # Missing data
    X_test_missing = add_missing(X_test)
    missing_acc = accuracy_score(y_test, model.predict(X_test_missing))
    
    results.append({
        "Model": name,
        "Clean Accuracy": clean_acc,
        "Noisy Accuracy": noisy_acc,
        "Missing Accuracy": missing_acc
    })

# Display results
results_df = pd.DataFrame(results)
print("\nModel Comparison Results:\n")
print(results_df)