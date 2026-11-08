"""Streamlit front-end: evaluate pre-trained classifiers on an uploaded test CSV."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "diagnosis"
CLASS_NAMES = {0: "Malignant", 1: "Benign"}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Tumour Classifier Lab", page_icon="🩺",
                   layout="wide")


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }
    return pred, prob, metrics


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv",
                                    help="Must contain all 30 feature columns "
                                         "and the 'diagnosis' target column.")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=uploaded is None)
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))

st.title("🩺 Breast Tumour Classifier Lab")
st.caption("Five pre-trained classifiers on the Breast Cancer Wisconsin (Diagnostic) "
           "dataset. Models are loaded from disk - nothing is retrained here.")

# -------------------------------------------------------------------- data
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(ROOT / "test_data.csv")
else:
    st.info("Upload a CSV or tick the bundled-sample checkbox to begin.")
    st.stop()

model = load_model(model_name)
needed = list(model.feature_names_in_)
missing = [c for c in needed + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

X, y = df[needed], df[TARGET].astype(int)
if y.nunique() < 2:
    st.error("Target column needs both classes to compute AUC.")
    st.stop()

pred, prob, metrics = evaluate(model, X, y)

tab_single, tab_compare, tab_data = st.tabs(
    ["Selected model", "Compare all models", "Data preview"])

# ----------------------------------------------------------- selected model
with tab_single:
    st.subheader(f"{model_name} - results on {len(df)} rows")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=list(CLASS_NAMES.values()),
                    yticklabels=list(CLASS_NAMES.values()))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.markdown("**Classification report**")
    report = classification_report(y, pred, target_names=list(CLASS_NAMES.values()),
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ------------------------------------------------------------ all models
with tab_compare:
    st.subheader("All five models on the same data")
    rows = []
    for name in MODEL_FILES:
        _, _, m = evaluate(load_model(name), X, y)
        rows.append({"ML Model Name": name, **m})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                 width="stretch")
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])

# -------------------------------------------------------------- preview
with tab_data:
    st.write(f"{df.shape[0]} rows x {df.shape[1]} columns")
    st.dataframe(df.head(50), width="stretch")
    out = df.copy()
    out["predicted"] = pred
    out["prob_benign"] = prob.round(4)
    st.download_button("Download predictions (CSV)", out.to_csv(index=False),
                       file_name=f"predictions_{model_name.replace(' ', '_')}.csv")
