import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="When Systems Break",
    page_icon="📉",
    layout="wide",
)


@st.cache_data
def load_default_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, data.target_names


@st.cache_resource
def train_model():
    X, y, _ = load_default_data()
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42),
    )
    model.fit(X_train, y_train)
    return model, X_train.mean(), X_train.std().replace(0, 1)


def initialize_state(X):
    if "input_data" not in st.session_state:
        st.session_state.input_data = X.head(1).copy()
    if "operations" not in st.session_state:
        st.session_state.operations = []


def reset_input(X):
    st.session_state.input_data = X.head(1).copy()
    st.session_state.operations = []


def inject_noise(noise_level, feature_std):
    rng = np.random.default_rng(42)
    X_current = st.session_state.input_data.copy()
    noise = rng.normal(0, noise_level, X_current.shape) * feature_std[X_current.columns].to_numpy()
    st.session_state.input_data = X_current + noise
    st.session_state.operations.append(f"Noise {noise_level:.2f}")


def create_missing_data():
    X_current = st.session_state.input_data.copy()
    columns_to_blank = X_current.columns[: max(1, len(X_current.columns) // 5)]
    X_current.loc[:, columns_to_blank] = np.nan
    st.session_state.input_data = X_current
    st.session_state.operations.append("Missing data")


def remove_features(train_mean):
    X_current = st.session_state.input_data.copy()
    columns_to_remove = X_current.columns[: max(1, len(X_current.columns) // 5)]
    X_current.loc[:, columns_to_remove] = train_mean[columns_to_remove]
    st.session_state.input_data = X_current
    st.session_state.operations.append("Feature removal")


def estimate_failure_risk(confidence, noise_level, missing_ratio, operation_count):
    risk = (1 - confidence) * 0.55
    risk += noise_level * 0.25
    risk += missing_ratio * 0.15
    risk += min(operation_count * 0.05, 0.20)
    return min(max(risk, 0), 1)


X_default, y_default, target_names = load_default_data()
model, train_mean, train_std = train_model()
initialize_state(X_default)

st.title("When Systems Break")

uploaded_file = st.file_uploader("Upload a CSV with breast cancer feature columns", type=["csv"])

if uploaded_file is not None:
    uploaded = pd.read_csv(uploaded_file)
    expected_columns = list(X_default.columns)
    if all(column in uploaded.columns for column in expected_columns):
        st.session_state.input_data = uploaded[expected_columns].head(1).copy()
        st.session_state.operations = ["Uploaded data"]
    else:
        st.error("Uploaded CSV must include the same feature columns used by the breast cancer dataset.")

left, right = st.columns([0.36, 0.64])

with left:
    noise_percent = st.slider("Noise Level", 0, 100, 20)
    noise_level = noise_percent / 100

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Inject Noise", use_container_width=True):
            inject_noise(noise_level, train_std)
        if st.button("Create Missing Data", use_container_width=True):
            create_missing_data()
    with action_cols[1]:
        if st.button("Remove Features", use_container_width=True):
            remove_features(train_mean)
        if st.button("Reset", use_container_width=True):
            reset_input(X_default)

    st.subheader("Applied Conditions")
    if st.session_state.operations:
        for operation in st.session_state.operations:
            st.write(operation)
    else:
        st.write("Clean input")

with right:
    X_input = st.session_state.input_data.copy()
    missing_ratio = float(X_input.isna().mean().mean())
    X_model = X_input.fillna(train_mean)

    probabilities = model.predict_proba(X_model)
    prediction = model.predict(X_model)[0]
    confidence = float(probabilities.max(axis=1)[0])
    predicted_label = target_names[prediction]
    failure_risk = estimate_failure_risk(
        confidence,
        noise_level,
        missing_ratio,
        len(st.session_state.operations),
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Prediction", predicted_label)
    metric_cols[1].metric("Confidence", f"{confidence:.1%}")
    metric_cols[2].metric("Failure Risk", f"{failure_risk:.1%}")

    st.subheader("Current Input")
    st.dataframe(X_input, use_container_width=True)

    st.subheader("Class Probabilities")
    probability_table = pd.DataFrame(
        {
            "Class": target_names,
            "Probability": probabilities[0],
        }
    )
    st.bar_chart(probability_table, x="Class", y="Probability")
