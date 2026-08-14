# Machine Learning Assignment 2 - Classification & Streamlit Deployment

## Student details
- **Name:** Deepak Joshi
- **BITS ID:**  2025da04146
- **Programme:** M.Tech (AIML / DSE)
- **Course:** Machine Learning
- **Assignment:** Assignment 2

## a. Problem statement

The objective is to implement multiple supervised machine-learning classification models on the same public dataset, compare them using the required evaluation measures, build an interactive Streamlit application, and deploy the application on Streamlit Community Cloud.

The implemented models are:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

The required evaluation metrics are:
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## b. Dataset description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Public source:** UCI Machine Learning Repository  
**UCI dataset page:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- **Number of instances:** 569
- **Number of predictive features:** 30
- **Task type:** Binary classification
- **Classes:** Benign and Malignant
- **Target encoding used in this project:** benign = 0, malignant = 1
- **Train/test split:** 80% / 20%
- **Split strategy:** Stratified
- **Random state:** 42

The dataset satisfies the assignment constraints of at least 500 instances and at least 12 features.

## c.Live Streamlit Application

**Streamlit App:** https://mlassignment2-x6a4apu6z4n9y9g87q4q6g.streamlit.app/

## d. Models used and comparison table

The table below reports model performance on the same held-out test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9676 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9649 | 0.9970 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

## Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs strongly after feature standardization. It provides a competitive balance of accuracy, recall and MCC and serves as a strong linear baseline. |
| Decision Tree | Easy to interpret and captures non-linear decision rules, but its single-tree structure can generalize less consistently than the strongest ensemble model. |
| kNN | Standardization is necessary because distance-based learning is sensitive to feature scale. It performs well, but predictions depend on the local neighborhood structure. |
| Naive Bayes | Fast and simple. Its conditional-independence assumption is restrictive because several diagnostic features are correlated, yet it remains a useful probabilistic baseline. |
| Random Forest (Ensemble) | Combines many trees and reduces the variance associated with a single decision tree. On this split it provides the strongest overall MCC-led performance. |
| **Overall Winner** | **Random Forest (Ensemble)**, selected primarily using MCC, with F1 and AUC as secondary checks. MCC is useful because it summarizes all four cells of the confusion matrix. |

## Repository structure

```text
project-folder/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- data/
|   |-- full_dataset.csv
|   |-- training_data.csv
|-- model/
|   |-- __init__.py
|   |-- logistic_regression.py
|   |-- decision_tree.py
|   |-- knn.py
|   |-- naive_bayes.py
|   |-- random_forest.py
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- random_forest.joblib
|-- artifacts/
|   |-- metrics.csv
|   |-- classification_reports.json
|   |-- confusion_matrices.json
|   |-- metadata.json
```
## Reproducibility

The project uses:
- stratified 80/20 train-test split
- `random_state=42` wherever randomness is involved
- saved trained model files in `model/`
- fixed `test_data.csv`
- generated evaluation artifacts under `artifacts/`

To regenerate the saved models and metrics, run:

```bash
python train_models.py
```
