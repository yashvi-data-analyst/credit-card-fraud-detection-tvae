# 🚀 Fraud Detection using Generative AI (TVAE + RandomForest)

## 📌 Project Overview
This project demonstrates how **Generative AI** can be used to solve the **class imbalance problem** in credit card fraud detection.  
Using **TVAE (Tabular Variational Autoencoder)**, we generate synthetic fraud transactions and augment them with real data to improve model performance.



## 🎯 Objectives
- Perform **EDA** on real transaction dataset.  
- Train **baseline fraud detection model** on real data.  
- Generate **synthetic fraud samples** using **TVAE**.  
- Create an **augmented dataset (real + synthetic)**.  
- Train and compare models (Baseline vs Augmented).  
- Visualize real vs synthetic data distributions.  
- Save final datasets, results, and model for deployment.  



## 🛠️ Tech Stack
- **Language**: Python 3.x  
- **Libraries**:  
  - Data Processing → `pandas`, `numpy`  
  - Visualization → `matplotlib`, `seaborn`  
  - ML Models → `scikit-learn` (RandomForest)  
  - Generative AI → `sdv` (TVAESynthesizer)  
  - Model Persistence → `joblib`  



## 📂 Project Structure

📁 Fraud-Detection-GenAI/
│── creditcard.csv
│── fraud_model.pkl
│── my_charts/
│ ├── class_distribution.png
│ ├── amount_distribution.png
│ ├── correlation_heatmap.png
│ ├── roc_baseline.png
│ ├── roc_augmented.png
│ ├── tsne_visualization.png
│ ├── model_results.csv
│ ├── classification_reports.txt
│ ├── fraud_data.csv
│ ├── synthetic_fraud.csv
│ └── augmented_dataset.csv
│── main.py
│── README.md
│── presentation.pptx (or include link)




## ▶ Quick Access to Presentation
You can view the full project presentation using the following link:  
**[Project Presentation (Google Drive)](https://drive.google.com/file/d/1y0bT1ZBivlnSpm3rZtzmZpLIddMYpBgF/view?usp=drivesdk)**



## 📊 Workflow
1. **EDA (Exploratory Data Analysis)**  
   - Class distribution (Fraud vs Non-Fraud)  
   - Transaction amount distribution  
   - Correlation heatmap  

2. **Baseline Model**  
   - Train RandomForest on real dataset  
   - Evaluate Precision, Recall, F1, AUC  
   - Plot ROC Curve  

3. **Synthetic Data Generation (TVAE)**  
   - Train TVAE on fraud transactions  
   - Generate 5,000 synthetic fraud samples  

4. **Augmentation + Model Training**  
   - Combine real + synthetic data  
   - Retrain RandomForest  
   - Evaluate & compare  

5. **Visualization**  
   - ROC Curve (Baseline vs Augmented)  
   - t-SNE plot (Real vs Synthetic Fraud)  

6. **Outputs & Saving**  
   - Classification reports  
   - Model results CSV  
   - Final model (`fraud_model.pkl`)  



## 📈 Results
| Model                       | Precision | Recall | F1-Score | AUC   |
|-----------------------------|-----------|--------|----------|-------|
| Baseline (Real Only)        | ~0.80     | ~0.40  | ~0.54    | ~0.76 |
| Augmented (Real + Synthetic)| ~0.84     | ~0.67  | ~0.74    | ~0.89 |

- Synthetic data significantly improved recall and AUC.  
- Augmentation reduced class imbalance impact.  



## 💡 Key Learnings
- Generative AI (TVAE) can effectively create synthetic fraud data.  
- Synthetic augmentation boosts **recall** (catching more fraud cases).  
- Evaluation via **ROC, AUC, t-SNE** shows improved model performance and realistic synthetic data distribution.  



## 📌 References
- Dataset: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Library: [SDV (Synthetic Data Vault)](https://sdv.dev/)  



## ✅ Deliverables
- Source Code (`main.py`)  
- Trained Model (`fraud_model.pkl`)  
- Charts & Visualizations (`my_charts/`)  
- Reports & Results (`classification_reports.txt`, `model_results.csv`)  
- Presentation → [Google Drive link above]  
- Final Documentation (`README.md`)


## 🌐 Deployment
This project has been successfully **deployed** as part of the internship deliverables.  

- **Deployed Platform**: [Streamlit]  
- **Live Demo**: [https://credit-card-fraud-detection-tvae-eexjaoyxq4euxwikpf3r63.streamlit.app]  
- **Model Used**: `fraud_model.pkl` (RandomForest trained on augmented dataset)  

👉 The deployment allows users to input transaction data and get a prediction (Fraud / Non-Fraud) in real-time.



👨‍💻 *Completed as part of a 6-Week AI/ML Internship on Generative AI for Data Augmentation.*
