import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# Create folder for saving charts & files
os.makedirs("my_charts", exist_ok=True)

# Load Dataset
data = pd.read_csv("creditcard.csv")
print("Dataset shape:", data.shape)
print(data['Class'].value_counts())

# ---------- Graph 1: Class Distribution ----------
plt.figure()
sns.countplot(x='Class', data=data)
plt.title("Fraud (1) vs Non-Fraud (0)")
plt.savefig("my_charts/class_distribution.png")
plt.show()

# ---------- Graph 2: Transaction Amount Distribution ----------
plt.figure()
sns.histplot(data['Amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig("my_charts/amount_distribution.png")
plt.show()

# ---------- Graph 3: Correlation Heatmap ----------
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), cmap="coolwarm", vmax=0.8)
plt.title("Correlation Heatmap")
plt.savefig("my_charts/correlation_heatmap.png")
plt.show()

# Train-Test Split
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Baseline Model
clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_baseline.fit(X_train, y_train)
y_pred = clf_baseline.predict(X_test)
baseline_auc = roc_auc_score(y_test, clf_baseline.predict_proba(X_test)[:,1])

print("\nBaseline Classification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Baseline AUC:", baseline_auc)

# ---------- Graph 4: ROC Curve (Baseline) ----------
plt.figure()
RocCurveDisplay.from_estimator(clf_baseline, X_test, y_test)
plt.title("ROC Curve – Baseline")
plt.savefig("my_charts/roc_baseline.png")
plt.show()

# Train TVAE on Fraud Data (FAST VERSION - 500 samples only)
fraud_data = data[data['Class'] == 1].copy().drop('Class', axis=1)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(fraud_data)

synthesizer = TVAESynthesizer(metadata)
print("\nTraining TVAE on 500 fraud samples... ⏳")
synthesizer.fit(fraud_data.sample(500, random_state=42, replace=True))
print("✅ TVAE training complete!")

# Generate Synthetic Fraud Data
synth_fraud = synthesizer.sample(num_rows=5000)
synth_fraud['Class'] = 1
print("Synthetic fraud samples generated:", synth_fraud.shape)

# Augmented Data
augmented_data = pd.concat([data, synth_fraud], ignore_index=True)
X_aug = augmented_data.drop('Class', axis=1)
y_aug = augmented_data['Class']
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(X_aug, y_aug, test_size=0.3, stratify=y_aug, random_state=42)

# Augmented Model
clf_aug = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_aug.fit(X_train_aug, y_train_aug)
y_pred_aug = clf_aug.predict(X_test_aug)
augmented_auc = roc_auc_score(y_test_aug, clf_aug.predict_proba(X_test_aug)[:,1])

print("\nAugmented Classification Report:\n", classification_report(y_test_aug, y_pred_aug, digits=4))
print("Augmented AUC:", augmented_auc)

# ---------- Graph 5: ROC Curve (Augmented) ----------
plt.figure()
RocCurveDisplay.from_estimator(clf_aug, X_test_aug, y_test_aug)
plt.title("ROC Curve – Augmented")
plt.savefig("my_charts/roc_augmented.png")
plt.show()

# ---------- Graph 6: t-SNE Visualization ----------
real_fraud_1000 = fraud_data.sample(n=500, random_state=42, replace=True)
synthetic_1000 = synth_fraud.drop('Class', axis=1).sample(n=500, random_state=42, replace=True)

combined = pd.concat([
    real_fraud_1000.assign(Source='Real'),
    synthetic_1000.assign(Source='Synthetic')
])
combined_targets = combined.pop('Source')

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_repr = tsne.fit_transform(combined)
tsne_df = pd.DataFrame(tsne_repr, columns=['TSNE1', 'TSNE2'])
tsne_df['Source'] = combined_targets.values

plt.figure(figsize=(8,6))
sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Source', alpha=0.6)
plt.title("t-SNE: Real vs Synthetic Fraud Distribution")
plt.savefig("my_charts/tsne_visualization.png")
plt.show()

# ---------- Save Results Comparison ----------
results = pd.DataFrame({
    "Model": ["Baseline (Real Data)", "Augmented (Real + Synthetic)"],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test_aug, y_pred_aug)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test_aug, y_pred_aug)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test_aug, y_pred_aug)
    ],
    "AUC": [
        baseline_auc,
        augmented_auc
    ]
})

results.to_csv("my_charts/model_results.csv", index=False)

# ---------- Save Reports & Data ----------
with open("my_charts/classification_reports.txt", "w") as f:
    f.write("Baseline Model Report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))
    f.write("\nBaseline AUC: " + str(baseline_auc) + "\n\n")

    f.write("Augmented Model Report:\n")
    f.write(classification_report(y_test_aug, y_pred_aug, digits=4))
    f.write("\nAugmented AUC: " + str(augmented_auc) + "\n")

fraud_data.to_csv("my_charts/fraud_data.csv", index=False)
synth_fraud.to_csv("my_charts/synthetic_fraud.csv", index=False)
augmented_data.to_csv("my_charts/augmented_dataset.csv", index=False)

print("✅ All datasets (fraud, synthetic, augmented) saved in 'my_charts/' folder!")
print("✅ Model comparison report saved in 'my_charts/model_results.csv'")
print("✅ Classification reports saved in 'my_charts/classification_reports.txt'")

# Save final model
joblib.dump(clf_aug, "fraud_model.pkl")
print("✅ Final model saved as fraud_model.pkl")
