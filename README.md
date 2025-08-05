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


