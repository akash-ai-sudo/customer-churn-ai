# AI-Powered Customer Churn Prediction & Retention Recommendation System

## 1. Project Title
AI-Powered Customer Churn Prediction & Retention Recommendation System

## 2. Project Overview
This project is an end-to-end Machine Learning pipeline that predicts whether a telecommunications customer is at a high risk of churning (canceling their service). It provides a full analysis of the underlying data, trains robust predictive models, extracts insights using Explainable AI (SHAP), and wraps everything into an interactive, real-time Streamlit dashboard.

## 3. Problem Statement
Customer churn is a critical issue for telecommunication companies. Acquiring a new customer is drastically more expensive than retaining an existing one. Currently, retention strategies are often reactive rather than proactive. This project aims to identify at-risk customers *before* they leave and provide actionable retention recommendations.

## 4. Objectives
- Build a reproducible end-to-end ML pipeline avoiding data leakage.
- Address imbalanced classes to ensure high Recall (preventing churners from slipping through the cracks).
- Train, evaluate, and compare Logistic Regression, Random Forest, and XGBoost models.
- Provide business interpretability using Feature Importance and SHAP.
- Deliver an interactive Dashboard with real-time UI validations.

## 5. Key Features
- **Data-Leakless Pipeline:** Train/Test split enforced *before* feature scaling and encoding.
- **Class Balancing:** Models initialized with `class_weight='balanced'` explicitly to favor Recall.
- **Strict UI Validations:** The dashboard seamlessly locks out conflicting features automatically (e.g., streaming services disabled if the user has "No internet service").
- **Explainability:** Global Feature Importance and Local SHAP visual comparisons out of the box.

## 6. Dataset Information
- **Source:** IBM Telco Customer Churn dataset
- **Size:** 7,043 rows, 21 columns
- **Features:** Demographics (Gender, SeniorCitizen), Services (Internet, Phone, Streaming, TechSup), and Account information (Tenure, Contract, Monthly/Total Charges).
- **Target Variable:** `Churn` (Yes/No) - Highly imbalanced (~73.4% No / ~26.6% Yes)

## 7. Technologies and Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP
- **Visualizations:** Matplotlib, Seaborn
- **Web App/UI:** Streamlit

## 8. ML Models Used
1. **Logistic Regression** (max_iter=1000, class_weight='balanced')
2. **Random Forest Classifier** (n_estimators=100, class_weight='balanced')
3. **XGBoost Classifier** (scale_pos_weight adjusted dynamically)

## 9. Data Preprocessing
- Removed globally unique but statistically irrelevant IDs (`customerID`).
- Fixed formatting constraints, notably coercing empty strings `" "` in `TotalCharges` (typically for tenure=0 customers) into `NaN`, followed by dropping those 11 anomalous rows to preserve pipeline integrity.

## 10. Feature Engineering
- **Train/Test Splitting:** Conducted initially at an 80/20 ratio using stratified sampling.
- **Categorical Columns:** Applied `OneHotEncoder` resulting in 45 distinct features post-transformation.
- **Numerical Columns:** Scaled via `StandardScaler` to ensure normalized regression gradient descent.
- **Pipeline:** Bundled into a strict `ColumnTransformer` (saved as `preprocessor.joblib`).

## 11. Model Evaluation Metrics
*Sample of actual logged metrics from testing (Logistic Regression performed remarkably well on resolving target classification imbalance):*
- **Accuracy:** 0.7257
- **Precision:** 0.4901 (Trade-off intentionally accepted: approx. 50% false-alarm rate ensures maximum risk coverage)
- **Recall:** 0.7968 (Significantly improved to capture ~80% of actual churners)
- **F1-Score:** 0.6069
- **ROC-AUC:** 0.8351 (Selected Best Model)

## 12. Explainable AI / SHAP
- **Global Overview:** Generates top 20 Absolute Coefficients mapping directly back to explicit features natively encoded in the ML pipeline.
- **SHAP Summary:** Leveraged `shap.LinearExplainer` against the globally best Logistic Regression model to pinpoint which specific localized services strictly trigger churn behaviors.

## 13. Prediction and Risk Classification
- **Classification Engine:** Outputs probability bounds via `.predict_proba()`.
- **Low Risk:** Probability < 0.33
- **Medium Risk:** 0.33 <= Probability < 0.66
- **High Risk:** Probability >= 0.66

## 14. Retention Recommendation System
Built directly into `src/recommendations.py`, the system correlates predictions against input demographics to offer localized rules:
- **High charges detected?** Recommend targeted, custom loyalty discounts.
- **No Tech Support combined with Fiber?** Recommend a free trial.
- **Month-to-Month?** Offer immediate discounts to securely shift them to a 1-year contract.

## 15. Streamlit Dashboard
The culmination of the project is housed in `app.py`. The robust dashboard features three pages:
1. **Dataset Overview & EDA:** Displays actual static raw distribution metric charts directly injected from `reports/`.
2. **Model Evaluation:** Highlights comparison matrices, ROC curves, and SHAP analyses.
3. **Real-time Predictor:** Allows interactive customer manipulation while enforcing edge-case locks (e.g. users without internet strictly have "No internet service" applied dynamically to dependencies).

## 16. Project Folder Structure
```text
customer-churn-ai/
│
├── data/                    # Raw & preprocessed CSVs
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── explainability.py
│   ├── predict.py
│   └── recommendations.py
│
├── models/                  # .joblib persistent pipelines
├── reports/                 # Output PNG charts & CSV metrics
├── app.py                   # Streamlit Frontend
├── requirements.txt         # Strictly Pinned Requirements
├── README.md                # Documentation
└── .gitignore               # Model and Report friendly git-tracking
```

## 17. Installation Requirements
Use Python 3.10 or 3.11. It's strictly advised to create a new Virtual Environment to avoid dependency conflicts.
Requirements strictly pinned in `requirements.txt` to guarantee cross-compatibility (ensure Streamlit Cloud deployments match Python versions).

## 18. Exact Steps to Run the Project
1. Clone this repository internally.
2. `cd customer-churn-ai`
3. `pip install -r requirements.txt`
4. Run ML Pipeline in strict order to generate fresh reports/models:
   - `python src/data_preprocessing.py`
   - `python src/feature_engineering.py`
   - `python src/train_models.py`
   - `python src/explainability.py`
5. Boot up the Dashboard:
   - `streamlit run app.py`

## 19. Example Prediction Workflow
- User navigates to the "Real-time Prediction Engine" tab.
- Sets Contract to `Month-to-month` and Internet to `Fiber optic`, adjusting charges to `$95`.
- Hits `Predict Churn Risk`.
- `app.py` passes the dictionary into `predict.py`.
- `predict_churn()` isolates Risk mapping to *Medium/High*.
- Contextual Recommendations trigger dynamically advising proactive discount measures.

## 20. Results
- **Model Output:** Logistic Regression selected dynamically natively mapping *0.8351 ROC-AUC*. 
- **Recall Target Hit:** Achieved ~79.6% True Positive rate safely securing a business-forward identification pattern catching ~4-out-of-5 real churners natively. Although this introduces a subset of False Positives (represented by a ~49% Precision), offering marginal retention incentives to these falsely flagged customers remains a vastly superior business trade-off than permanently losing active customers entirely.

## 21. Limitations
- **Imbalance Drift:** While `class_weight='balanced'` handles training properly, live drift metrics can degrade model performance natively over years of shifting telecommunication environments without proper re-training strategies in production.
- **Static Dataset:** Predictions revolve largely around fixed demographic data assuming constant behaviors, not dynamic user activities logs.

## 22. Future Scope
- Implementation of a generalized pipeline architecture allowing daily cron-job re-training.
- Integration tests targeting REST APIs offloading standard UI Streamlit components onto dedicated Next.js builds.

## 23. Conclusion
The comprehensive customer-churn modeling successfully achieved end-to-end integration mapping. Not only were accurate and recall-biased inferences outputted predictably across the board, but explicit steps were taken to securely package the modeling logic behind a verified interactive user application designed to minimize Garbage In logic out-of-the-box natively.
