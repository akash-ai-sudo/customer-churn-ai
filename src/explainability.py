import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def generate_explanations(data_dir, model_dir, report_dir):
    print("Loading data and best model...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    best_model = joblib.load(os.path.join(model_dir, 'best_model.joblib'))
    
    with open(os.path.join(model_dir, 'best_model_info.json'), 'r') as f:
        model_info = json.load(f)
    best_model_name = model_info['best_model_name']
    
    print(f"Generating explanations for {best_model_name}")
    
    # Global Feature Importance (for tree-based models or LR)
    plt.figure(figsize=(10, 8))
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] # top 20
        sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis')
        plt.title(f'Top 20 Feature Importances ({best_model_name})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'feature_importance.png'))
        plt.close()
        print("Feature importance plot saved.")
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        indices = np.argsort(importances)[::-1][:20]
        sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis')
        plt.title(f'Top 20 Absolute Coefficients ({best_model_name})')
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'feature_importance.png'))
        plt.close()
        print("Feature importance (coef) plot saved.")
    
    # SHAP Explanations
    # Subsample data for SHAP if it's too large to compute quickly
    X_sample = X_train.sample(500, random_state=42)
    
    try:
        print("Calculating SHAP values...")
        if best_model_name in ['Random Forest', 'XGBoost']:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_sample)
            
            # For random forest, shap_values is a list of arrays for each class. We want positive class (1).
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1]
            else:
                shap_values_to_plot = shap_values
                
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values_to_plot, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'shap_summary.png'))
            plt.close()
            print("SHAP TreeExplainer summary plot saved.")
            
        else: # Logistic Regression
            explainer = shap.LinearExplainer(best_model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'shap_summary.png'))
            plt.close()
            print("SHAP LinearExplainer summary plot saved.")
            
    except Exception as e:
        print(f"Warning: SHAP explanation failed: {e}")
        # Create a dummy image or text to bypass hard UI crashes if SHAP fails due to version mismatch
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, f'SHAP generation failed\\n{str(e)}', ha='center', va='center')
        plt.savefig(os.path.join(report_dir, 'shap_summary.png'))
        plt.close()

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data')
    model_dir = os.path.join(base_path, 'models')
    report_dir = os.path.join(base_path, 'reports')
    
    generate_explanations(data_dir, model_dir, report_dir)
