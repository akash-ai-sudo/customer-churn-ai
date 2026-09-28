import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from PIL import Image
import sys

from src.predict import predict_churn
from src.recommendations import get_recommendation

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Page Config
st.set_page_config(page_title="AI Customer Churn & Retention System", layout="wide", page_icon="📊")

# --- Header ---
st.title("📊 AI-Powered Customer Churn Prediction & Retention Recommendation System")
st.markdown("This dashboard provides an end-to-end view of our Machine Learning pipeline for predicting Customer Churn.")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Dataset Overview & EDA", "Model Evaluation & Explainability", "Real-time Prediction Engine"])

def load_data():
    try:
        return pd.read_csv(os.path.join(DATA_DIR, 'cleaned_churn_data.csv'))
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# --- Page 1: Dataset Overview & EDA ---
if page == "Dataset Overview & EDA":
    st.header("1. Dataset Summary")
    if not df.empty:
        total_customers = len(df)
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
        col3.metric("Features", f"{df.shape[1] - 1}")
        
        with st.expander("View Raw Dataset Sample"):
            st.dataframe(df.head(10))
            
        st.header("2. Exploratory Data Analysis (EDA)")
        st.markdown("Insights from the IBM Telco Dataset.")
        
        c1, c2 = st.columns(2)
        try:
            c1.image(Image.open(os.path.join(REPORTS_DIR, 'churn_distribution.png')), use_container_width=True)
            c2.image(Image.open(os.path.join(REPORTS_DIR, 'churn_by_contract.png')), use_container_width=True)
            st.image(Image.open(os.path.join(REPORTS_DIR, 'monthly_charges_churn.png')), use_container_width=True)
            st.image(Image.open(os.path.join(REPORTS_DIR, 'correlation_matrix.png')), use_container_width=True)
        except Exception as e:
            st.warning("EDA images not found. Please run the pipeline first.")
            
    else:
        st.error("Cleaned data not found. Please run data_preprocessing.py.")

# --- Page 2: Model Evaluation & Explainability ---
elif page == "Model Evaluation & Explainability":
    st.header("1. Machine Learning Model Comparison")
    
    try:
        metrics_df = pd.read_csv(os.path.join(REPORTS_DIR, 'model_comparison.csv'), index_col=0)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
        
        with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'r') as f:
            best_info = json.load(f)
        st.success(f"🏆 **Best Model:** {best_info['best_model_name']} (ROC-AUC: {best_info['roc_auc']:.4f})")
    except Exception as e:
        st.warning("Model metrics not found. Run model training first.")
        
    st.header("2. ROC & Confusion Matrices")
    c1, c2 = st.columns(2)
    try:
        c1.image(Image.open(os.path.join(REPORTS_DIR, 'roc_curves.png')), use_container_width=True)
        st.image(Image.open(os.path.join(REPORTS_DIR, 'confusion_matrices.png')), use_container_width=True)
    except:
        st.warning("Evaluation charts not found.")
        
    st.header("3. AI Feature Explainability (SHAP & Importance)")
    try:
        c1, c2 = st.columns(2)
        c1.image(Image.open(os.path.join(REPORTS_DIR, 'feature_importance.png')), use_container_width=True, caption="Global Feature Importance")
        c2.image(Image.open(os.path.join(REPORTS_DIR, 'shap_summary.png')), use_container_width=True, caption="SHAP Summary Plot")
    except:
        st.warning("Explainability charts not found.")

# --- Page 3: Real-time Prediction Engine ---
elif page == "Real-time Prediction Engine":
    st.header("🔮 Individual Customer Churn Predictor")
    st.markdown("Input customer parameters to get real-time churn probability and AI-driven retention recommendations.")
    
    if df.empty:
        st.error("Cannot load feature bounds. Data missing.")
    else:
        st.subheader("Customer Demographics & Account info")
        col1, col2 = st.columns(2)
        
        gender = col1.selectbox("Gender", ["Male", "Female"])
        senior = col2.selectbox("Senior Citizen", [0, 1])
        partner = col1.selectbox("Partner", ["Yes", "No"])
        dependents = col2.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        
        st.subheader("Services & Subscriptions")
        col3, col4, col5 = st.columns(3)
        phone = col3.selectbox("Phone Service", ["Yes", "No"])
        
        no_phone = (phone == "No")
        multiple_lines = col4.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], index=0 if no_phone else 1, disabled=no_phone)
        
        internet = col5.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        no_net = (internet == "No")
        idx = 2 if no_net else 0
        
        online_sec = col3.selectbox("Online Security", ["No", "Yes", "No internet service"], index=idx, disabled=no_net)
        online_bup = col4.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=idx, disabled=no_net)
        dev_prot = col5.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=idx, disabled=no_net)
        
        tech_sup = col3.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=idx, disabled=no_net)
        stream_tv = col4.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=idx, disabled=no_net)
        stream_mov = col5.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=idx, disabled=no_net)
        
        st.subheader("Billing & Charges")
        col6, col7 = st.columns(2)
        contract = col6.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = col7.selectbox("Paperless Billing", ["Yes", "No"])
        payment = col6.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure*70.0))
        
        st.markdown("---")
        # Prepare data dict
        customer_data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bup,
            'DeviceProtection': dev_prot,
            'TechSupport': tech_sup,
            'StreamingTV': stream_tv,
            'StreamingMovies': stream_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        result = predict_churn(customer_data)
        if "error" in result:
            st.error(result["error"])
        else:
            prob = result['churn_probability']
            risk = result['risk_level']
            
            rc1, rc2 = st.columns(2)
            
            # Display Metrics with corrected label
            rc1.metric("Risk Score", f"{prob * 100:.1f}%")
            
            if risk == "Low Risk":
                rc2.success(f"Risk Level: {risk}")
            elif risk == "Medium Risk":
                rc2.warning(f"Risk Level: {risk}")
            else:
                rc2.error(f"Risk Level: {risk}")
                
            # Recommendations
            st.subheader("💡 Retention Recommendation")
            rec = get_recommendation(customer_data, risk)
            st.info(rec)
