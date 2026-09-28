import os
import joblib
import pandas as pd
import numpy as np

# Load artifacts globally for fast inference
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_PATH, 'models')

# Try to load models. We use Try-Except to allow importing the file in contexts where models might not exist yet
try:
    best_model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
except Exception as e:
    print(f"Warning: Model load failed: {e}")
    best_model = None
    preprocessor = None
    feature_names = None


def predict_churn(customer_data_dict):
    """
    Predicts churn probability and risk level for a single customer.
    customer_data_dict should be a dictionary mapping feature names to values.
    """
    df = pd.DataFrame([customer_data_dict])
    
    # Transform
    # To avoid shape issues, ensure categorical and numerical columns are same as training
    # The preprocessor expects all columns that were present in X
    
    try:
        df_transformed = preprocessor.transform(df)
    except Exception as e:
        return {"error": f"Preprocessing error: {str(e)}"}
        
    # Predict
    prob = best_model.predict_proba(df_transformed)[0][1] # Probability of Churn=Yes
    
    # Risk Classification
    if prob < 0.33:
        risk = "Low Risk"
    elif prob < 0.66:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
        
    return {
        "churn_probability": float(prob),
        "risk_level": risk,
        "transformed_data": df_transformed
    }


