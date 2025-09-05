import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE

# Create folder for saving charts & files
os.makedirs("my_charts", exist_ok=True)

st.title("💳 Credit Card Fraud Detection")
st.write("Baseline vs Augmented (SMOTE) Model Comparison")

# Upload CSV
uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded! Shape: {data.shape}")

    # ---------- Graph 1: Class Distribution ----------
    st.subheader("Fraud vs Non-Fraud Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Class", data=data, ax=ax)
    st.pyplot(fig)

    # ---------- Graph 2: Transaction Amount Distribution ----------
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["Amount"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    # ---------- Graph 3: Correlation Heatmap ----------
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), cmap="coolwarm", ax=ax, vmax=0.8)
    st.pyplot(fig)

    # Train-Test Split
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Baseline Model
    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_baseline.fit(X_train, y_train)
    y_pred = clf_baseline.predict(X_test)
    baseline_auc = roc_auc_score(y_test, clf_baseline.predict_proba(X_test)[:, 1])

    st.subheader("📊 Baseline Model Results")
    st.text(classification_report(y_test, y_pred, digits=4))
    st.write("Baseline AUC:", baseline_auc)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(clf_baseline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve – Baseline")
    st.pyplot(fig)

    # ---------- Augmentation with SMOTE ----------
    st.subheader("🔄 Augmentation with SMOTE")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    st.write("Original data shape:", X.shape, y.value_counts().to_dict())
    st.write("Resampled data shape:", X_res.shape, pd.Series(y_res).value_counts().to_dict())

    X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(
        X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
    )

    clf_aug = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_aug.fit(X_train_aug, y_train_aug)
    y_pred_aug = clf_aug.predict(X_test_aug)
    augmented_auc = roc_auc_score(y_test_aug, clf_aug.predict_proba(X_test_aug)[:, 1])

    st.subheader("📊 Augmented Model Results (SMOTE)")
    st.text(classification_report(y_test_aug, y_pred_aug, digits=4))
    st.write("Augmented AUC:", augmented_auc)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(clf_aug, X_test_aug, y_test_aug, ax=ax)
    ax.set_title("ROC Curve – Augmented (SMOTE)")
    st.pyplot(fig)

    # ---------- Save Results ----------
    results = pd.DataFrame(
        {
            "Model": ["Baseline (Real Data)", "Augmented (SMOTE)"],
            "Precision": [
                precision_score(y_test, y_pred),
                precision_score(y_test_aug, y_pred_aug),
            ],
            "Recall": [
                recall_score(y_test, y_pred),
                recall_score(y_test_aug, y_pred_aug),
            ],
            "F1-Score": [
                f1_score(y_test, y_pred),
                f1_score(y_test_aug, y_pred_aug),
            ],
            "AUC": [baseline_auc, augmented_auc],
        }
    )

    st.subheader("📊 Model Comparison")
    st.dataframe(results)

    results.to_csv("my_charts/model_results.csv", index=False)
    joblib.dump(clf_aug, "fraud_model.pkl")
    st.success("✅ Final model saved as fraud_model.pkl and results saved in my_charts/")

else:
    st.warning("Please upload creditcard.csv to start.")
