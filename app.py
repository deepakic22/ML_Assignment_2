from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_TEST = ROOT / "test_data.csv"

st.set_page_config(
    page_title="ML Classification Model Evaluator",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.assignment-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
}
.small-note {font-size: .9rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

@st.cache_resource
def load_models():
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }

@st.cache_data
def load_reference_metrics():
    return pd.read_csv(ARTIFACT_DIR / "metrics.csv")

@st.cache_data
def load_metadata():
    with open(ARTIFACT_DIR / "metadata.json", "r") as f:
        return json.load(f)

def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "pred": pred,
        "prob": prob,
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

models = load_models()
metadata = load_metadata()
reference_metrics = load_reference_metrics()

st.title("Machine Learning Classification Model Evaluator")
st.caption("BITS Pilani WILP - Machine Learning Assignment 2")

st.markdown(
    f"""
    <div class="assignment-card">
    <b>Dataset:</b> {metadata['dataset']} &nbsp; | &nbsp;
    <b>Instances:</b> {metadata['instances']} &nbsp; | &nbsp;
    <b>Features:</b> {metadata['features']} &nbsp; | &nbsp;
    <b>Positive class:</b> malignant (1)
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("Reference model comparison on the fixed held-out test set", expanded=True):
    show = reference_metrics.copy()
    for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        show[c] = show[c].map(lambda x: f"{x:.4f}")
    st.dataframe(show, use_container_width=True, hide_index=True)
    winner = reference_metrics.sort_values(["MCC", "F1", "AUC"], ascending=False).iloc[0]
    st.success(
        f"Overall winner on the reference test set: {winner['ML Model Name']} "
        f"(MCC={winner['MCC']:.4f}, Accuracy={winner['Accuracy']:.4f})"
    )

st.subheader("1. Upload test data")
uploaded = st.file_uploader(
    "Upload a CSV containing the 30 feature columns. Include a 'target' column "
    "to calculate evaluation metrics. You may directly upload the provided test_data.csv.",
    type=["csv"]
)

if uploaded is None:
    df = pd.read_csv(DEFAULT_TEST)
    st.info("No file uploaded. The included test_data.csv is being used.")
else:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded uploaded CSV with {len(df)} rows.")
    except Exception as exc:
        st.error(f"Unable to read the CSV: {exc}")
        st.stop()

st.dataframe(df.head(10), use_container_width=True)

required_features = metadata["feature_names"]
missing = [c for c in required_features if c not in df.columns]
if missing:
    st.error(
        "The uploaded file is missing required feature columns: "
        + ", ".join(missing[:10])
        + (" ..." if len(missing) > 10 else "")
    )
    st.stop()

X = df[required_features].copy()
has_target = "target" in df.columns
y = df["target"].astype(int) if has_target else None

st.subheader("2. Select model")
selected_name = st.selectbox("Classification model", list(MODEL_FILES.keys()))
selected_model = models[selected_name]

if st.button("Run evaluation / prediction", type="primary", use_container_width=True):
    pred = selected_model.predict(X)
    prob = selected_model.predict_proba(X)[:, 1]

    prediction_df = df.copy()
    prediction_df["predicted_target"] = pred
    prediction_df["predicted_class"] = np.where(pred == 1, "malignant", "benign")
    prediction_df["malignant_probability"] = prob

    if has_target:
        result = evaluate(selected_model, X, y)
        st.subheader("3. Evaluation metrics")
        cols = st.columns(6)
        for col, metric in zip(cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
            value = result[metric]
            col.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Confusion matrix")
            cm = confusion_matrix(y, result["pred"])
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm)
            ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
            ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(selected_name)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("#### Classification report")
            report = classification_report(
                y, result["pred"],
                target_names=["benign", "malignant"],
                output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

    else:
        st.warning(
            "No 'target' column was found, so evaluation metrics cannot be calculated. "
            "Predictions are shown below."
        )

    st.subheader("4. Prediction results")
    st.dataframe(
        prediction_df[
            ["predicted_target", "predicted_class", "malignant_probability"]
        ].head(50),
        use_container_width=True
    )
    st.download_button(
        "Download predictions as CSV",
        data=prediction_df.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()
st.markdown(
    "<span class='small-note'>Target encoding: benign = 0, malignant = 1. "
    "The app uses only test data for evaluation.</span>",
    unsafe_allow_html=True
)
