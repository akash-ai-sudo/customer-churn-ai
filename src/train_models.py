import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

def train_and_evaluate(data_dir, model_dir, report_dir):
    # Load data
    print("Loading processed data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    # Initialize Models with class_weight='balanced' to improve Recall
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=(len(y_train[y_train==0])/len(y_train[y_train==1])))
    }
    
    # Metrics collection
    results = {}
    best_model_name = None
    best_roc_auc = 0.0
    best_model = None
    
    plt.figure(figsize=(10, 8)) # For ROC Curve
    
    # Grid for Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'ROC-AUC': float(roc)
        }
        
        print(f"Results for {name}:")
        print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC: {roc:.4f}")
        
        if roc > best_roc_auc:
            best_roc_auc = roc
            best_model_name = name
            best_model = model
            
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(1) # Switch back to ROC figure
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.3f})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{name} Confusion Matrix")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Save ROC curves
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'roc_curves.png'))
    plt.close(1)
    
    # Save Confusion Matrices
    plt.tight_layout()
    fig.savefig(os.path.join(report_dir, 'confusion_matrices.png'))
    plt.close(fig)
    
    # Save the best model
    print(f"\nBest Model by ROC-AUC: {best_model_name} with {best_roc_auc:.4f}")
    joblib.dump(best_model, os.path.join(model_dir, 'best_model.joblib'))
    
    # Save the model name so the app knows what was used
    with open(os.path.join(model_dir, 'best_model_info.json'), 'w') as f:
        json.dump({"best_model_name": best_model_name, "roc_auc": best_roc_auc}, f)

    # Save metrics
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(os.path.join(report_dir, 'model_comparison.csv'))
    print("\nModel evaluation completed. Models and plots saved.")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data')
    model_dir = os.path.join(base_path, 'models')
    report_dir = os.path.join(base_path, 'reports')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    train_and_evaluate(data_dir, model_dir, report_dir)
