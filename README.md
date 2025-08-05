# Credit Default Prediction and Fraud Detection

This project develops end-to-end machine learning pipelines for two high-impact classification problems in finance:

1. **Credit Default Prediction**  
2. **Fraud Detection**

Both tasks involve significant class imbalance and demand high recall to minimize financial risk. The pipelines are built using Python and include preprocessing, resampling, threshold tuning, model training, SHAP-based interpretability, and final post-processing.

---

## 🚀 Key Highlights

- Achieved **~78% precision and recall** on the credit default task with Gradient Boosting and engineered features.
- Final fraud pipeline reached **82% precision and 80% recall**, using a two-stage architecture with XGBoost and a post-prediction filtering model.
- Used SHAP to identify and resolve model blind spots via targeted feature creation.
- Explored TabNet for tabular deep learning, but ensemble methods proved more effective on structured data.

---

## 🧠 Techniques Used

- Resampling (SMOTE, SMOTEENN, undersampling)
- Threshold calibration
- SHAP-based interpretability
- Tabular deep learning (TabNet)
- Error-informed feature refinement
- Two-stage filtering for fraud

---

## 📁 Folder Structure

├── Credit_Default_Prediction_Irene_Murua.ipynb      # Notebook for credit default modeling
├── Fraud_Detection_Irene_Murua.ipynb                # Notebook for fraud detection pipeline
├── cs-training.csv                                  # Dataset for credit default
├── predicciones_flags.csv                           # Output used in fraud filter model
├── Credit Default Prediction and Fraud Detection.docs  # Full project report
├── README.md                                        # Project overview and instructions

## 📂 Dataset

Due to file size constraints, datasets are not hosted in this repository.

To run the project locally:

1. **Fraud Detection Data**  
   Download the following files from the [Kaggle IEEE-CIS Fraud Detection Competition]:  
   - `train_transaction.csv`  
   - `train_identity.csv`  


2. **Credit Default Prediction Data**  
   Add the file `cs-training.csv`, used for the credit risk modeling part.

---

## 📄 About `predicciones_flags.csv`

This file is generated during the fraud detection pipeline (specifically in `Fraud_Detection_Irene_Murua.ipynb`). It includes:

- The original model probabilities
- Flags for false negatives
- Additional engineered features for use in the filtering model

It’s used to train the **secondary filter model** that corrects missed fraud cases and helps improve precision while preserving recall.
