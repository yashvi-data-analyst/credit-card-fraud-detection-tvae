import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
from sklearn.manifold import TSNE

# Try importing SDV (TVAE Synthesizer)
try:
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except Exception as e:
    SDV_AVAILABLE = False

st.set_page_config(page_title="Fraud Detection with Synthetic Data", layout="wide")

st.title("💳 Credit Card Fraud Detection with Data Augmentation (TVAE)")

# File uploader
uploaded_file = st.file_uploader("📂 Upload creditcard.csv", type=["csv"])

if uploaded_file:
    # Option to load smaller dataset
    use_sample = st.checkbox("🔹 Use 20,000 sample rows (faster, less RAM)", value=True)

    if use_sample:
        data = pd.read_csv(uploaded_file).sample(20000, random_state=42)
    else:
        data = pd.read_csv(uploaded_file)

    st.success(f"Dataset Loaded! Shape: {data.shape}")
    st.write(data.head())

    # Class distribution
    st.subheader("📊 Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Class", data=data, ax=ax)
    st.pyplot(fig)

    # Transaction amount
    st.subheader("💰 Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["Amount"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), cmap="coolwarm", vmax=0.8, ax=ax)
    st.pyplot(fig)

    # Train-Test split
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

    st.subheader("⚖️ Baseline Model Performance")
    st.text(classification_report(y_test, y_pred, digits=4))
    st.metric("Baseline AUC", f"{baseline_auc:.4f}")

    # ROC Curve Baseline
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(clf_baseline, X_test, y_test, ax=ax)
    st.pyplot(fig)

    # Augmentation with TVAE (if available)
    synth_fraud = pd.DataFrame()
    if SDV_AVAILABLE:
        try:
            fraud_data = data[data["Class"] == 1].copy()

            # ⚠️ Drop columns that break SDV metadata (Time is NOT unique)
            fraud_data = fraud_data.drop(["Class", "Time"], axis=1, errors="ignore")

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(fraud_data)

            synthesizer = TVAESynthesizer(metadata)
            st.info("⏳ Training TVAE on 500 fraud samples...")
            synthesizer.fit(fraud_data.sample(500, random_state=42, replace=True))
            st.success("✅ TVAE training complete!")

            synth_fraud = synthesizer.sample(num_rows=5000)
            synth_fraud["Class"] = 1
            st.write("Synthetic fraud samples generated:", synth_fraud.shape)

        except Exception as e:
            st.error(f"❌ TVAE training failed: {e}")
    else:
        st.warning("⚠️ SDV / TVAE not installed. Skipping synthetic data generation.")

    if not synth_fraud.empty:
        # Augmented Data
        augmented_data = pd.concat([data, synth_fraud], ignore_index=True)
        X_aug = augmented_data.drop("Class", axis=1)
        y_aug = augmented_data["Class"]
        X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(
            X_aug, y_aug, test_size=0.3, stratify=y_aug, random_state=42
        )

        clf_aug = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_aug.fit(X_train_aug, y_train_aug)
        y_pred_aug = clf_aug.predict(X_test_aug)
        augmented_auc = roc_auc_score(y_test_aug, clf_aug.predict_proba(X_test_aug)[:, 1])

        st.subheader("🚀 Augmented Model Performance")
        st.text(classification_report(y_test_aug, y_pred_aug, digits=4))
        st.metric("Augmented AUC", f"{augmented_auc:.4f}")

        # ROC Curve Augmented
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(clf_aug, X_test_aug, y_test_aug, ax=ax)
        st.pyplot(fig)

        # t-SNE
        st.subheader("🌐 t-SNE: Real vs Synthetic Fraud")
        try:
            real_fraud_500 = fraud_data.sample(n=500, random_state=42, replace=True)
            synthetic_500 = synth_fraud.drop("Class", axis=1).sample(
                n=500, random_state=42, replace=True
            )
            combined = pd.concat(
                [real_fraud_500.assign(Source="Real"), synthetic_500.assign(Source="Synthetic")]
            )
            combined_targets = combined.pop("Source")

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_repr = tsne.fit_transform(combined)
            tsne_df = pd.DataFrame(tsne_repr, columns=["TSNE1", "TSNE2"])
            tsne_df["Source"] = combined_targets.values

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="Source", alpha=0.6, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"t-SNE visualization failed: {e}")

        # Comparison table
        st.subheader("📑 Results Comparison")
        results = pd.DataFrame(
            {
                "Model": ["Baseline (Real Data)", "Augmented (Real + Synthetic)"],
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
        st.dataframe(results, use_container_width=True)

        # Save model
        joblib.dump(clf_aug, "fraud_model.pkl")
        st.success("✅ Final Augmented Model saved as fraud_model.pkl")

else:
    st.warning("👆 Please upload `creditcard.csv` to start.")
