"""Train five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.

Outputs
-------
model/<name>.joblib   : one fitted sklearn Pipeline per classifier (scaler + model)
model/metrics.csv     : evaluation metrics on the held-out split
full_dataset.csv      : the complete dataset (for reference)
test_data.csv         : held-out test split (features + target) for the Streamlit app
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 42
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "diagnosis"  # 1 = benign, 0 = malignant (sklearn's encoding)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True,
                                  random_state=SEED),
}


def file_name(name: str) -> str:
    return name.lower().replace(" ", "_") + ".joblib"


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": TARGET})
    df.to_csv(ROOT / "full_dataset.csv", index=False)

    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, clf in MODELS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({"ML Model Name": name, **compute_metrics(y_test, pred, prob)})
        joblib.dump(pipe, MODEL_DIR / file_name(name))

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(f"Rows: {len(df)}, features: {X.shape[1]}, train/test: "
          f"{len(X_train)}/{len(X_test)}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
