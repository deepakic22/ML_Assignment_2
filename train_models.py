from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "full_dataset.csv"
MODEL_DIR = ROOT / "model"
ARTIFACT_DIR = ROOT / "artifacts"

def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["target", "target_name"], errors="ignore")
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pd.concat([X_test.reset_index(drop=True),
               y_test.reset_index(drop=True).rename("target")], axis=1).to_csv(
        ROOT / "test_data.csv", index=False
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=4, random_state=42
        ),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    safe = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "kNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest (Ensemble)": "random_forest",
    }

    rows, reports, cms = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        })

        reports[name] = classification_report(
            y_test, pred, target_names=["benign", "malignant"],
            output_dict=True, zero_division=0
        )
        cms[name] = confusion_matrix(y_test, pred).tolist()
        joblib.dump(model, MODEL_DIR / f"{safe[name]}.joblib")

    metrics = pd.DataFrame(rows).sort_values("MCC", ascending=False)
    metrics.to_csv(ARTIFACT_DIR / "metrics.csv", index=False)
    (ARTIFACT_DIR / "classification_reports.json").write_text(
        json.dumps(reports, indent=2)
    )
    (ARTIFACT_DIR / "confusion_matrices.json").write_text(
        json.dumps(cms, indent=2)
    )

    print("\nModel comparison on held-out test set:")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
